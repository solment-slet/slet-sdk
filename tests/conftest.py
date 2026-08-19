import os

import pytest
import pytest_asyncio
import time
import sys
import subprocess
import socket
from dotenv import load_dotenv

import httpx

from tests.helpers.auth import get_auth_data, get_client
from slet_sdk.aelite.manifest import AgentManifest, ModelConfig, EmbedConfig, RerankerConfig

load_dotenv()


# ===========================================================================
# Локальный embeddings-сервер на llama-cpp-python
# ===========================================================================


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _stream_process_output(proc: subprocess.Popen, log_path: str) -> None:
    """Continuously drains proc.stdout into a file in a background thread.
    Without this, PIPE's buffer can fill up and block the subprocess once
    it's producing enough output (embedding request logs), which can look
    exactly like the server "hanging"/refusing connections - and even
    when it doesn't block, no one ever reads what a mid-run crash printed
    since _wait_for_server only drains stdout once, at the very start."""
    import threading

    def _drain():
        with open(log_path, "wb") as f:
            for line in iter(proc.stdout.readline, b""):
                f.write(line)
                f.flush()

    t = threading.Thread(target=_drain, daemon=True)
    t.start()


def _wait_for_server(
    proc: subprocess.Popen,
    url: str,
    api_key: str,
    timeout: float = 120.0,
    label: str = "Local test server",
) -> None:
    deadline = time.monotonic() + timeout
    last_exc: Exception | None = None
    headers = {"Authorization": f"Bearer {api_key}"}
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            output = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
            raise RuntimeError(
                f"{label} process exited early with code "
                f"{proc.returncode} before becoming ready at {url}.\n"
                f"--- process output ---\n{output}"
            )
        try:
            resp = httpx.get(url, headers=headers, timeout=2.0)
            if resp.status_code < 500:
                return
        except Exception as e:
            last_exc = e
        time.sleep(0.5)
    raise RuntimeError(
        f"{label} did not become ready at {url} within {timeout}s: {last_exc}"
    )


@pytest.fixture(scope="session")
def _local_embed_server(tmp_path_factory):
    """
    Launches tests/helpers/embed_server.py (fastembed-backed, ONNX
    Runtime) as a subprocess for the session.

    Two mutually exclusive ways to configure the model:
    - EMBED_FASTEMBED_MODEL_PATH: a local, pre-downloaded ONNX model
      directory. No network access at all - preferred when the test
      environment has no/unreliable egress to huggingface.co, since
      otherwise the server silently downloads several hundred MB before
      it can start listening, which looks exactly like a hung/refusing
      server to every test waiting on it.
    - EMBED_FASTEMBED_MODEL: a HuggingFace model name, downloaded and
      cached by fastembed on first use (requires network access).

    If neither is set, nothing is launched - EMBED_BASE_URL is either
    already set manually (external embeddings provider) or absent, in
    which case `embeds` will pytest.skip as before.
    """
    model_path = os.environ.get("EMBED_FASTEMBED_MODEL_PATH")
    model_name = os.environ.get("EMBED_FASTEMBED_MODEL")

    if not model_path and not model_name:
        yield
        return

    if model_path and not os.path.isdir(model_path):
        raise RuntimeError(
            f"EMBED_FASTEMBED_MODEL_PATH is set to '{model_path}' but the "
            f"directory does not exist."
        )

    port = _find_free_port()
    dummy_api_key = "sk-local-fastembed"
    resolved_name = model_name or os.environ.get("EMBED_FASTEMBED_MODEL_ALIAS", "local-embeddings")

    log_path = os.environ.get("EMBED_SERVER_LOG_PATH", "/tmp/fastembed_server.log")

    cmd = [
        sys.executable, "-m", "tests.helpers.embed_server",
        "--api-key", dummy_api_key,
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    if model_path:
        cmd += ["--model-path", model_path]
        if model_name:
            cmd += ["--model", model_name]
    else:
        cmd += ["--model", model_name]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    _stream_process_output(proc, log_path)

    try:
        # Local model_path loads should be near-instant - no need for a
        # long timeout unless we're actually downloading from HF.
        startup_timeout = float(os.environ.get(
            "EMBED_SERVER_STARTUP_TIMEOUT",
            "30" if model_path else "300",
        ))
        _wait_for_server(
            proc, f"http://127.0.0.1:{port}/v1/models", dummy_api_key,
            timeout=startup_timeout,
            label="Local embeddings server",
        )

        docker_host = os.environ.get("EMBED_DOCKER_HOST", "host.docker.internal")

        os.environ["EMBED_BASE_URL"] = f"http://{docker_host}:{port}/v1"
        os.environ["EMBED_LOCAL_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
        os.environ["EMBED_MODEL_NAME"] = resolved_name
        os.environ["EMBED_API_KEY"] = dummy_api_key

        yield proc, log_path

    finally:
        if proc.poll() is not None:
            with open(log_path, "rb") as f:
                tail = f.read()[-4000:].decode(errors="replace")
            print(
                f"[fastembed server] process died during the test session "
                f"(exit code {proc.returncode}). Last log output:\n{tail}"
            )
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture
def _embed_server_alive(_local_embed_server):
    """
    Function-scoped liveness check, run before every test that needs
    embeddings - _local_embed_server itself only launches the process
    once per session, so if it dies mid-run (e.g. blocked on a full stdout
    pipe, or OOM-killed by the OS under concurrent embedding load), every
    subsequent test would otherwise silently hit a dead port with no clear
    signal as to when/why it happened.
    """
    result = _local_embed_server
    if result is None:
        return  # external embeddings provider, nothing local to check
    proc, log_path = result
    if proc.poll() is not None:
        with open(log_path, "rb") as f:
            tail = f.read()[-4000:].decode(errors="replace")
        pytest.fail(
            f"Local embeddings server process died (exit code "
            f"{proc.returncode}) before this test ran. Last log output:\n{tail}"
        )


@pytest.fixture
def embeds(_embed_server_alive):
    """
    Mirrors the `models` fixture, but for the embeddings chain A-MEM
    requires (`AgentManifest.embeds` with `use_for_amem=True`).

    Function-scoped (not session-scoped) so that _embed_server_alive's
    per-test liveness check actually runs before every test that needs
    embeddings - a session-scoped fixture can't depend on a function-scoped
    one. Constructing the returned EmbedConfig list is cheap, so there's no
    real cost to rebuilding it per test.
    """
    base_url = os.environ.get("EMBED_BASE_URL")
    model_name = os.environ.get("EMBED_MODEL_NAME")
    api_key = os.environ.get("EMBED_API_KEY")

    if not base_url or not model_name or not api_key:
        pytest.skip(
            "EMBED_BASE_URL, EMBED_MODEL_NAME, EMBED_API_KEY must be set "
            "(directly, or via EMBED_FASTEMBED_MODEL to auto-launch a "
            "local fastembed-based server) to run A-MEM e2e tests."
        )

    return [EmbedConfig(base_url=base_url, model=model_name, api_key=api_key)]


@pytest.fixture
def embed_seed_base_url(_embed_server_alive):
    """
    Base URL for calling the embeddings server directly from the *test
    process* (used by tests/helpers/rag_redis.py to embed seed documents
    before the server ever runs), as opposed to `embeds`' EmbedConfig.base_url,
    which - when a local server was auto-launched via EMBED_FASTEMBED_MODEL(_PATH)
    - points at EMBED_DOCKER_HOST (e.g. 'host.docker.internal'), a hostname
    only resolvable from inside the server's own docker container, not from
    the host machine running pytest.

    Falls back to EMBED_BASE_URL itself when no local server was launched
    (an external embeddings provider is configured directly) - an external
    provider's URL is assumed reachable from both the server and the test
    process.
    """
    return os.environ.get("EMBED_LOCAL_BASE_URL") or os.environ.get("EMBED_BASE_URL")


# ===========================================================================
# Локальный reranker-сервер на sentence-transformers CrossEncoder
# ===========================================================================


@pytest.fixture(scope="session")
def _local_reranker_server(tmp_path_factory):
    """
    Launches tests/helpers/reranker_server.py (CrossEncoder-backed, Cohere-
    compatible /rerank) as a subprocess for the session - mirrors
    _local_embed_server exactly (same startup-wait / liveness-check /
    log-draining machinery), just pointed at a different helper module and
    a different readiness path scheme (RERANKER_* instead of EMBED_*).

    If RERANKER_MODEL is unset, nothing is launched - RERANKER_BASE_URL is
    either already set manually (external reranker provider) or absent, in
    which case `rerankers` will pytest.skip as before.
    """
    model_name = os.environ.get("RERANKER_MODEL")
    if not model_name:
        yield
        return

    port = _find_free_port()
    dummy_api_key = "sk-local-reranker"
    resolved_name = model_name

    log_path = os.environ.get("RERANKER_SERVER_LOG_PATH", "/tmp/reranker_server.log")

    cmd = [
        sys.executable, "-m", "tests.helpers.reranker_server",
        "--model", model_name,
        "--api-key", dummy_api_key,
        "--host", "0.0.0.0",
        "--port", str(port),
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    _stream_process_output(proc, log_path)

    try:
        startup_timeout = float(
            os.environ.get("RERANKER_SERVER_STARTUP_TIMEOUT", "300")
        )
        _wait_for_server(
            proc, f"http://127.0.0.1:{port}/v1/models", dummy_api_key,
            timeout=startup_timeout,
            label="Local reranker server",
        )

        docker_host = os.environ.get("EMBED_DOCKER_HOST", "host.docker.internal")

        os.environ["RERANKER_BASE_URL"] = f"http://{docker_host}:{port}"
        os.environ["RERANKER_MODEL_NAME"] = resolved_name
        os.environ["RERANKER_API_KEY"] = dummy_api_key

        yield proc, log_path

    finally:
        if proc.poll() is not None:
            with open(log_path, "rb") as f:
                tail = f.read()[-4000:].decode(errors="replace")
            print(
                f"[reranker server] process died during the test session "
                f"(exit code {proc.returncode}). Last log output:\n{tail}"
            )
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture
def _reranker_server_alive(_local_reranker_server):
    """Function-scoped liveness check, mirrors _embed_server_alive."""
    result = _local_reranker_server
    if result is None:
        return  # external reranker provider, nothing local to check
    proc, log_path = result
    if proc.poll() is not None:
        with open(log_path, "rb") as f:
            tail = f.read()[-4000:].decode(errors="replace")
        pytest.fail(
            f"Local reranker server process died (exit code "
            f"{proc.returncode}) before this test ran. Last log output:\n{tail}"
        )


@pytest.fixture
def rerankers(_reranker_server_alive):
    """
    Mirrors the `embeds` fixture, but for the reranker provider chain RAG's
    `reranker.type='cross_encoder'` requires (`AgentManifest.rerankers`
    with `use_for_rag=True`).

    Function-scoped for the same reason as `embeds` - _reranker_server_alive's
    per-test liveness check needs to run before every test that needs it.
    """
    base_url = os.environ.get("RERANKER_BASE_URL")
    model_name = os.environ.get("RERANKER_MODEL_NAME")
    api_key = os.environ.get("RERANKER_API_KEY")

    if not base_url or not model_name or not api_key:
        pytest.skip(
            "RERANKER_BASE_URL, RERANKER_MODEL_NAME, RERANKER_API_KEY must "
            "be set (directly, or via RERANKER_MODEL to auto-launch a "
            "local CrossEncoder-based server) to run RAG cross_encoder "
            "reranker e2e tests."
        )

    return [RerankerConfig(base_url=base_url, model=model_name, api_key=api_key)]


@pytest.fixture(scope="session")
def rag_redis_urls():
    local_url = os.environ.get("LOCAL_REDIS_URL")
    server_url = os.environ.get("SERVER_REDIS_URL")
    if not local_url and not server_url:
        pytest.skip("LOCAL_REDIS_URL or SERVER_REDIS_URL must be set to run RAG e2e tests against redis.")
    return local_url if local_url else server_url, server_url if server_url else local_url


@pytest_asyncio.fixture(scope="session")
async def auth_data():
    name, email, password = get_auth_data()

    async with get_client() as client:
        await client.signup(name, email, password)

    return email, password


@pytest_asyncio.fixture
async def client(auth_data):
    async with get_client() as client:
        await client.signin(*auth_data)

        yield client


@pytest.fixture(scope="session")
def models():
    models = []
    index = ""
    while True:
        base_url = os.environ.get(f"MODEL_BASE_URL{index}")
        model_name = os.environ.get(f"MODEL_NAME{index}")
        api_key = os.environ.get(f"MODEL_API_KEY{index}")

        if not base_url or not model_name or not api_key:
            if index == "":
                raise RuntimeError(
                    "MODEL_BASE_URL, MODEL_NAME, MODEL_API_KEY must be set"
                )
            break

        models.append(
            ModelConfig(
                base_url=base_url,
                model=model_name,
                api_key=api_key,
            )
        )

        index = "_2" if index == "" else f"_{int(index[1:]) + 1}"

    return models


@pytest.fixture(scope="session")
def manifest(models):
    return AgentManifest(
        id="Test_Agent",
        system_prompt="You are helpful assistant.",
        models=models,
    )


@pytest_asyncio.fixture(scope="session")
async def agent(client, manifest):
    return await client.aelite.agents.create_agent(manifest)


@pytest_asyncio.fixture
async def thread(client, agent):
    return await client.aelite.threads.create_thread(agent=agent.id)


@pytest_asyncio.fixture
async def session(client, thread):
    return await client.aelite.connect(thread.id)
