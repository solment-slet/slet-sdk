from slet_sdk.aelite_client import AEliteClient
import asyncio


# ✅ Пример использования
async def main():
    client = AEliteClient(base_url="https://127.0.0.1:8000", ssl_verify=False)

    print("Авторизация...")
    await client.login("*", "*")

    print("Инициализация LLM...")
    await client.init_llm("****")

    Ошикафж