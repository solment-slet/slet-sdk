import asyncio

from slet_sdk import SletClient
from slet_sdk.aelite.manifest import AgentManifest, ModelConfig, ProviderKind
from pydantic import BaseModel


async def main():
    async with SletClient(base_url="http://localhost:8080", ssl_verify=False) as c:
        await c.signin("testesolment@gmail.com", "20132061netT%")

        manifest = AgentManifest(
            id="TestSchemasAgent",
            models=[
                ModelConfig(
                    base_url="http://host.docker.internal:8090/v1",
                    model="gemma-4-E2B-it-Q4_K_M.gguf",
                    api_key="empty",
                    kind=ProviderKind.OPENAI_COMPATIBLE,
                    thinking_enabled=True,
                    reasoning_format="deepseek",
                    native_structured_output=False,
                )
            ]
        )

        class Sentences(BaseModel):
            sentences: list[str]

        s = await c.aelite.get_session_from_manifest(manifest)
        resp = await s.chat_with_thinking(
            "Разбей текст на предложения:\nПривет, как дела? И чем могу помочь?",
            response_schema=Sentences,
        )

        print(f"Answer:\n{resp[0]}\n\nThinking:\n{resp[1]}")


asyncio.run(main())