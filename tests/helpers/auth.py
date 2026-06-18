import os
import secrets
import string

from slet_sdk import SletClient


def random_name(length: int = 14) -> str:
    """
    Random display name.
    Example: 'Fjkslqweprtyui'
    """
    return (
        secrets.choice(string.ascii_uppercase)
        + "".join(
            secrets.choice(string.ascii_lowercase)
            for _ in range(length - 1)
        )
    )


def random_email():
    local = secrets.token_urlsafe(8).rstrip("=")
    domain = secrets.choice(["gmail", "yahoo", "outlook", "protonmail", "mailbox"])
    tld = secrets.choice(["com", "net", "org"])

    return f"{local}@{domain}.{tld}"


def random_password(length: int = 24) -> str:
    """
    Random password satisfying common complexity requirements.
    """
    alphabet = (
        string.ascii_lowercase
        + string.ascii_uppercase
        + string.digits
        + "!@#$%^&*"
    )

    while True:
        password = "".join(
            secrets.choice(alphabet)
            for _ in range(length)
        )

        if (
            any(c.islower() for c in password)
            and any(c.isupper() for c in password)
            and any(c.isdigit() for c in password)
            and any(c in "!@#$%^&*" for c in password)
        ):
            return password


def get_auth_data() -> tuple[str, str, str]:
    return random_name(), random_email(), random_password()


def get_client() -> SletClient:
    return SletClient(os.environ["BASE_URL"], ssl_verify=False)


async def create_logged_user() -> SletClient:
    name, email, password = get_auth_data()

    client = get_client()
    await client.signup(name, email, password)

    return client
