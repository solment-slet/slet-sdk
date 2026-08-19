import asyncio
import sys
from aioconsole import ainput
from slet_sdk import SletClient
from slet_sdk.aelite.manifest import AgentManifest, ToolConfig, MemoryConfig, BroadcastConfig, ModelConfig
from slet_sdk.aelite.tools import tool
from slet_sdk.aelite.agent import ThinkingDelta


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
    system_prompt="""Ты гениальный ассистент по имени Aelite.

    Перед тем как использовать любой инструмент добавляй текст для пользователя о том что его используешь.""",
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
                    inherit_tools_from=["SherlokHolms"],
                )
            ],
            inherit_tools_from=["MainAgent"],
        )
    ],
)


async def main():
    async with SletClient("http://localhost:8080") as client:
        # Авторизация
        await client.signin("tester1@gmail.com", "20310482lJSD:Flsdjfls")

        # Aelite
        aelite = client.aelite

        # Создание агента
        session = await aelite.agents.create_agent(manifest)

        # Создание треда
        thread = await aelite.threads.create_thread(session.id)

        # Подключение к агенту и подагентам
        session = await aelite.connect(thread.id)

        # ── Подписка на события ──
        # on_message теперь: (text, thinking, msg_id)
        session.on_message = lambda text, thinking, msg_id: print(
            f"\nПришло unsolicited-сообщение [{msg_id}]: {text}"
            + (f"\n  (thinking: {thinking})" if thinking else "")
        )

        # on_tool_start теперь: (tool_name, msg_id)
        session.on_tool_start = lambda tool_name, msg_id: print(
            f"\n🔧 Вызван инструмент: {tool_name} [{msg_id}]"
        )

        # on_error - сигнатура не изменилась
        session.on_error = lambda err: print(f"\n❌ Ошибка! {str(err)}")

        # Новый колбэк: ошибки сервера, привязанные к msg (или None)
        session.on_server_error = lambda err, msg_id: print(
            f"\n❌ Ошибка сервера [{msg_id}]: {str(err)}"
        )

        # Основной цикл (бесконечный, с прикреплением файлов) - без изменений
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

            # stream() теперь отдаёт TextDelta / ThinkingDelta вместо голых str
            thinking_open = False
            async for item in session.stream(user_input, files=files):
                if isinstance(item, ThinkingDelta):
                    if not thinking_open:
                        print("\n[thinking] ", end="", flush=True)
                        thinking_open = True
                    print(item.text, end="", flush=True)
                else:  # TextDelta
                    if thinking_open:
                        print("\n[/thinking]\n", end="", flush=True)
                        thinking_open = False
                    print(item.text, end="", flush=True)
            print()

            for file in files:
                try:
                    file.close()
                except:
                    pass


asyncio.run(main())