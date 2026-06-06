import asyncio
import sys
from aioconsole import ainput
from slet_sdk import SletClient
from slet_sdk.aelite.manifest import AgentManifest, ToolConfig, MemoryConfig, BroadcastConfig
from slet_sdk.aelite.tools import tool


@tool
async def get_clipboard() -> str:
    """Возвращает текущий буфер обмена пользователя"""
    return await ainput("Clipboard: ")


@tool(broadcast=BroadcastConfig())
async def show_notification(title: str, message: str) -> None:
    """Показывает уведомление на рабочем столе пользователя"""
    print(f"\n🔔 [{title}] {message}")


manifest = AgentManifest(
    id="MainAgent",
    system_prompt="""Ты полезный ассистент по имени Aelite, 
ты можешь использовать своих подагентов для помощи пользователю. Перед тем как использовать любой инструмент добавляй текст для пользователя о том что его используешь.""",
    concurrency="parallel",
    memory=MemoryConfig(enabled=True, summarization="async"),
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
            ],
        )
    ],
)


async def main():
    async with SletClient("http://localhost:8080") as client:
        # Авторизация
        await client.signin("tester1@gmail.com", "20310482lJSD:Flsdjfls")

        # Создание чата
        new_chat = await client.aelite.chats.new_chat()

        # Deploy + connect в одну операцию
        agent = await client.aelite.deploy_and_connect(manifest, thread_id=new_chat.id)

        # Подписка на события
        agent.on_message = lambda msg: print(f"Пришло сообщение!: {msg}")
        agent.on_tool_start = lambda tool_name: print(
            f"Вызван сервеный инструмент: {tool_name}"
        )
        agent.on_error = lambda err: print(f"Ошибка! {str(err)}")

        # Основной цикл
        while True:
            # Сбрасываем буфер перед чтением
            await asyncio.get_event_loop().run_in_executor(None, sys.stdin.flush)
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
                        # file_obj.close()

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
