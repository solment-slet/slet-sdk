import asyncio
from slet_sdk import SletClient
from slet_sdk.exceptions import SletClientError
from slet_sdk.schemas import ErrorCode

slet = SletClient(base_url="http://localhost:8000")


async def main():
    try:
        res = await slet.signup(
            name="Walter White", email="esolment@gmail.com", password="***"
        )
        print(res)
    except SletClientError as e:
        print(e.to_dict())
        if e.to_dict()["error"] == ErrorCode.INTERNAL_SERVER_ERROR:
            print("Yea")

        print(e)


asyncio.run(main())
