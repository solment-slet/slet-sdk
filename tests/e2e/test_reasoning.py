from slet_sdk.aelite.manifest import AgentManifest
from slet_sdk.aelite.agent import ThinkingDelta


async def get_reasoning_works(client, model) -> bool:
    """Helper."""
    manifest = AgentManifest(
        id="Test_Agent",
        models=model,
    )

    session = await client.aelite.get_session_from_manifest(manifest)

    prompt = (
        "On an island, knights always tell the truth and knaves always lie.\n"
        "You meet two inhabitants.\n"
        "A says: \"We are both knaves.\"\n"
        "Who is a knight and who is a knave?\n"
        "Explain your reasoning before giving the final answer."
    )

    reasoning_works = False

    thinking_open = False
    async for item in session.stream(prompt):
        if isinstance(item, ThinkingDelta):
            reasoning_works = True
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

    return reasoning_works


async def test_reasoning_enabled(client, models):
    for m in models:
        m.is_reasoning_model = True
        m.thinking_enabled = True
        m.reasoning_effort = "high"

    reasoning_works = await get_reasoning_works(client, models)

    assert reasoning_works == True


async def test_reasoning_disabled(client, models):
    for m in models:
        m.is_reasoning_model = False
        m.thinking_enabled = False
        m.reasoning_effort = None

    reasoning_works = await get_reasoning_works(client, models)

    assert reasoning_works == False
