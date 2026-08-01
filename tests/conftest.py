import os

import pytest
import pytest_asyncio
from dotenv import load_dotenv

from tests.helpers.auth import get_auth_data, get_client
from slet_sdk.aelite.manifest import AgentManifest, ModelConfig

load_dotenv()


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
def model():
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
def manifest(model):
    return AgentManifest(
        id="Test_Agent",
        system_prompt="You are helpful assistant.",
        model=model,
    )


@pytest_asyncio.fixture(scope="session")
async def agent(client, manifest):
    return client.aelite.agents.create_agent(manifest)


@pytest_asyncio.fixture
async def thread(client, agent):
    return client.aelite.threads.create_thread(agent.id)


@pytest_asyncio.fixture
async def session(client, thread):
    return client.aelite.connect(thread.id)
