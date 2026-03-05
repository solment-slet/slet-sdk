import asyncio

from aioconsole import ainput
from slet_sdk import SletClient
from slet_sdk.aelite.manifest import AgentManifest, ToolConfig, MemoryConfig
from slet_sdk.aelite.tools import tool


SYSTEM_PROMPT = """Ты полезный ассистент по имени Aelite, 
ты можешь использовать своих подагентов для помощи пользователю"""


@tool
async def get_clipboard() -> str:
    """Возвращает текущий буфер обмена пользователя"""
    return await ainput("Clipboard: ")


@tool(broadcast=True)
async def show_notification(title: str, message: str) -> None:
    """Показывает уведомление на рабочем столе пользователя"""
    print(f"\n🔔 [{title}] {message}")


async def main():
    async with SletClient(base_url="http://localhost:8000") as client:
        await client.signin("esolment@gmail.com", "20132061esS")

        manifest = AgentManifest(
            id="MainAgent",
            system_prompt=SYSTEM_PROMPT,
            memory=MemoryConfig(enabled=True, summarization=True),
            tools=[
                ToolConfig(name="get_time", type="server"),
                get_clipboard,
                show_notification,
            ],
            sub_agents=[
                AgentManifest(
                    id="SherlokHolms",
                    system_prompt="Ты агент-поисковик по имени Шерлок Холмс",
                    sub_agents=[
                        AgentManifest(
                            id="Krosh",
                            system_prompt="Ты персонаж из мультфильма Смешарики по имени Крош.",
                        )
                    ]
                )
            ]
        )

        # Deploy + connect в одну операцию
        agent = await client.aelite.deploy_and_connect(manifest)

        # Тулы регистрируются автоматически, но можно и вручную
        agent.register_tools_from_registry()

        agent.on_message = lambda msg: print(f"Пришло сообщение!: {msg}")
        agent.on_error = lambda err: print(f"Ошибка! {str(err)}")

        # Основной цикл
        while True:
            user_input = await ainput("YOU: ")
            files: list = []
            while True:
                filepath = await ainput("FILEPATH (ok если уже все введены): ")
                if filepath == "ok":
                    break
                try:
                    file_obj = None
                    file_obj = open(filepath, "rb")
                    files.append(file_obj)
                except FileNotFoundError:
                    print("Такого файла нет.")
                finally:
                    if file_obj:
                        pass
                        #file_obj.close()


            if user_input == "exit":
                break

            async for chunk in agent.stream(user_input, files=files):
                print(chunk, end="", flush=True)
            print()
            for file in files:
                try:
                    file.close()
                except:
                    pass


asyncio.run(main())