"""
embed_server.py - minimal OpenAI-compatible /v1/embeddings endpoint backed
by fastembed (ONNX Runtime), used as a local test double for a real
embeddings provider in e2e A-MEM tests.

Supports two ways of obtaining the model:
- --model <hf-model-name>: fastembed downloads and caches it from
  HuggingFace Hub on first use (requires network access).
- --model-path <local-dir>: loads a pre-downloaded ONNX model directory
  directly, with local_files_only=True - no network access at all. Use
  this when the test environment has no/unreliable egress to
  huggingface.co, which otherwise makes the server hang (and every test
  time out with a misleading "connection refused") while it silently
  downloads several hundred MB before it can start listening.
"""

from __future__ import annotations

import argparse

from fastapi import FastAPI, Header, HTTPException
from fastembed import TextEmbedding
from pydantic import BaseModel
import uvicorn


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str | None = None


def _load_embedder(model_name: str | None, model_path: str | None) -> tuple[TextEmbedding, str]:
    if model_path:
        if not model_name:
            raise ValueError(
                "--model-path requires --model to also be set to the real "
                "HuggingFace model name the local ONNX directory was "
                "downloaded for (must be one of "
                "TextEmbedding.list_supported_models()) - fastembed uses "
                "--model to determine the expected architecture/config, "
                "it does not infer it from the directory contents."
            )
        embedder = TextEmbedding(
            model_name=model_name,
            cache_dir=model_path,
            local_files_only=True,
        )
        return embedder, model_name

    if not model_name:
        raise ValueError("Either --model or --model-path must be provided.")

    embedder = TextEmbedding(model_name=model_name)
    return embedder, model_name


def create_app(model_name: str | None, model_path: str | None, api_key: str) -> FastAPI:
    app = FastAPI()
    embedder, resolved_name = _load_embedder(model_name, model_path)

    def _check_auth(authorization: str | None) -> None:
        if authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)):
        _check_auth(authorization)
        return {"data": [{"id": resolved_name, "object": "model"}]}

    @app.post("/v1/embeddings")
    async def create_embeddings(
        req: EmbeddingRequest,
        authorization: str | None = Header(default=None),
    ):
        _check_auth(authorization)
        texts = [req.input] if isinstance(req.input, str) else req.input

        vectors = list(embedder.embed(texts))

        return {
            "object": "list",
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": vec.tolist(),
                }
                for i, vec in enumerate(vectors)
            ],
            "model": resolved_name,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="HuggingFace model name (downloads on first use)")
    parser.add_argument("--model-path", default=None, help="Local pre-downloaded ONNX model directory (no network access)")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    app = create_app(args.model, args.model_path, args.api_key)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()