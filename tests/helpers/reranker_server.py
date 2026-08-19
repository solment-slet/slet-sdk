"""
reranker_server.py - minimal Cohere-compatible /rerank endpoint backed by
a local sentence-transformers CrossEncoder, used as a local test double
for a real reranker provider in e2e RAG tests.

Mirrors embed_server.py's structure and startup contract exactly, so it
can be launched/waited-on/torn-down the same way from conftest.py.

Contract (matches ResilientRerankerModel.rerank / RerankerConfig in
app/services/model_manager.py):
    POST /rerank
    Headers: Authorization: Bearer <api_key>
    Body:    {"model": str, "query": str, "documents": [str, ...], "top_n": int}
    Response: {"results": [{"index": int, "relevance_score": float}, ...]}
    (results sorted best-first, length <= top_n)

Only one way to obtain the model is supported (--model, a HuggingFace
cross-encoder name, e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2") -
no local-path variant, since CrossEncoder model directories are small
compared to the embedding case and network-access flakiness has not been
an issue in practice for this specific model family. Can be extended with
a --model-path option later if that changes.
"""

from __future__ import annotations

import argparse

from fastapi import FastAPI, Header, HTTPException
from sentence_transformers import CrossEncoder
from pydantic import BaseModel
import uvicorn


class RerankRequest(BaseModel):
    model: str | None = None
    query: str
    documents: list[str]
    top_n: int | None = None


def create_app(model_name: str, api_key: str) -> FastAPI:
    app = FastAPI()
    encoder = CrossEncoder(model_name)

    def _check_auth(authorization: str | None) -> None:
        if authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="Invalid API key")

    @app.get("/v1/models")
    async def list_models(authorization: str | None = Header(default=None)):
        # Mirrors embed_server.py's readiness-probe endpoint so
        # _wait_for_server (shared helper) works unchanged for both servers.
        _check_auth(authorization)
        return {"data": [{"id": model_name, "object": "model"}]}

    @app.post("/rerank")
    async def rerank(
        req: RerankRequest,
        authorization: str | None = Header(default=None),
    ):
        _check_auth(authorization)
        if not req.documents:
            return {"results": []}

        top_n = req.top_n if req.top_n is not None else len(req.documents)
        scores = encoder.predict([(req.query, d) for d in req.documents]).tolist()

        ranked = sorted(
            (
                {"index": i, "relevance_score": float(s)}
                for i, s in enumerate(scores)
            ),
            key=lambda r: r["relevance_score"],
            reverse=True,
        )[:top_n]

        return {"results": ranked}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace cross-encoder model name (downloads on first use)",
    )
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    app = create_app(args.model, args.api_key)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()