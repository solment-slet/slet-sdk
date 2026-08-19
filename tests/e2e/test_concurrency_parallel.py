import asyncio
import json

from slet_sdk.aelite.manifest import AgentManifest, MemoryConfig, ConcurrencyConfig


async def send_message(session, message, number):
    """Helper"""
    result = await session.chat(message)
    print(f"\n[answer_{number}]\n{result}\n[/answer_{number}]")
    return number, result


async def test_concurrency_parallel(client, models):
    manifest = AgentManifest(
        id="Test_Agent",
        models=models,
        system_prompt=(
            "You are a helpful assistant. When asked to tell a story, "
            "make one up immediately without asking clarifying questions "
            "about theme, characters, or length."
        ),
        memory=MemoryConfig(
            checkpointing=True,
            summarization="disabled",
            threshold=7,
            keep_last=4,
        ),
        concurrency=ConcurrencyConfig(
            mode="parallel",
        ),
    )

    session = await client.aelite.get_session_from_manifest(manifest)

    tasks = [
        asyncio.create_task(send_message(session, "Hello!", 1)),
        asyncio.create_task(send_message(session, "Tell me a long story before bedtime, make up your own.", 2)),
        asyncio.create_task(send_message(session, "How are you?", 3)),
    ]

    completion_order = []

    for future in asyncio.as_completed(tasks):
        number, result = await future
        assert isinstance(result, str)
        completion_order.append(number)

    assert completion_order[-1] == 2

    chat_history = (await client.aelite.threads.get_thread_with_history(session.thread_id)).messages

    data = [msg.model_dump(mode="json") for msg in chat_history]
    print(json.dumps(data, indent=4, ensure_ascii=False))

    human_count = sum(1 for m in chat_history if getattr(m, "role", None) == "user")
    ai_count = sum(1 for m in chat_history if getattr(m, "role", None) == "ai")

    assert human_count == 3, f"Expected 3 human messages, got {human_count}"
    assert ai_count == 3, f"Expected 3 ai messages, got {ai_count}"
