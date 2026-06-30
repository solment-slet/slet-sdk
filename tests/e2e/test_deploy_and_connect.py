async def test_deploy_and_connect(client, manifest):
    new_thread = await client.aelite.threads.create_thread()

    agent = await client.aelite.deploy_and_connect(manifest, thread_id=new_thread.id)

    print(await agent.chat("Привет, кто ты?"))
    print(await agent.chat("Скажи 'Апельсин'"))

    assert str(new_thread.id) == agent.thread_id