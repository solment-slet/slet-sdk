import pytest
import pytest_asyncio
from dotenv import load_dotenv

from tests.helpers.auth import get_auth_data, get_client
from slet_sdk.aelite.manifest import AgentManifest, ModelConfig

load_dotenv()


@pytest_asyncio.fixture(scope="session")
async def auth_data():
    name, email, password = get_auth_data()

    async with get_client() as client:
        await client.signup(name, email, password)

    return email, password


@pytest_asyncio.fixture
async def client(auth_data):
    async with get_client() as client:
        await client.signin(*auth_data)

        yield client


@pytest.fixture(scope="session")
def manifest():
    return AgentManifest(
        id="Test_Agent",
        system_prompt="You are helpful assistant.",
        model=ModelConfig(
            base_url="https://api.groq.com/openai/v1",
            model="openai/gpt-oss-120b",
            api_key="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",
        )
    )


@pytest_asyncio.fixture
async def agent(client, manifest):
    client.aelite.deploy_and_connect(manifest)

    return client
