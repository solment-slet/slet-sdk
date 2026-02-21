import asyncio
from aioconsole import ainput
from slet_sdk import SletClient
from slet_sdk.schemas.agent import AgentManifest, ToolConfig, MemoryConfig, ModelConfig


async def main():
    async with SletClient(base_url="http://localhost:8000") as client:

        # 1. Авторизация (если нужна)
        # await client.signin("user@example.com", "password")

        # 2. Описание агента (Манифест)
        manifest = AgentManifest(
            name="Neodim (male)",
            system_prompt="Ты персональный гениальный агент. У тебя есть свои подчиненные sub-агенты которые работают на тебя и помогают выполнять задачи пользователя. Ты можешь давать им задания одновременно, можешь пошагово. Твое имя {name}.\n\nВАЖНО - КОНТЕКСТ О ПРОШЛОМ: {summary}",
            model=ModelConfig(temperature=0.2, top_p=0.2, max_tokens=4096),
            memory=MemoryConfig(
                enabled=True,
                summarization=False,
                threshold=3,
                keep_last=2,
            ),
            tools=[
                ToolConfig(
                    name="programmer",
                    type="sub_agent",
                    description="Агент профессионал по программированию",
                    sub_agent_manifest=AgentManifest(
                        name="SubAgent профессионал по программированию",
                        system_prompt="Ты ИИ помогающий с программированием и IT темами, ты отлично пишешь код и разбираешься в ПО и ОС.",
                        memory=MemoryConfig(
                            enabled=False,
                        )
                    ),
                ),
                ToolConfig(
                    name="musician",
                    type="sub_agent",
                    description="Агент профессионал по музыке",
                    sub_agent_manifest=AgentManifest(
                        name="SubAgent профессионал по музыке",
                        system_prompt="Ты ИИ помогающий с музыкой, ты отлично разбираешься в музыке.",
                        memory=MemoryConfig(
                            enabled=False,
                        )
                    ),
                ),
            ],
        )

        thread_id = "sess-34363436353464"

        # 3. Деплой
        print("Deploying agent...")
        await client.deploy_agent(thread_id, manifest)
        print("Agent deployed!")

        # 4. Подключение
        agent = await client.connect_agent(thread_id)

        # Опционально: подписка на события
        agent.on_tool_start = lambda name: print(f"\n[TOOL] Executing {name}...")

        # 5. Общение
        mode = await ainput("MODE (stream or nonstream): ")

        while True:
            user_input = await ainput("YOU: ")
            if user_input == "exit":
                break

            # 5. Общение
            if mode == "stream":
                print("\nGenerating...")
                async for chunk in agent.stream(user_input):
                    print(chunk, end="", flush=True)
                print()
            else:
                print("\nGenerating...")
                print(await agent.chat(user_input))

        # 6. Триггер (если есть)
        # await agent.send_trigger("morning_alarm")


if __name__ == "__main__":
    asyncio.run(main())
