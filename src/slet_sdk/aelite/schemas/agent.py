from pydantic import Field
from slet_sdk.core.schemas.client_base import SuccessResponse

class AgentDeployResponse(SuccessResponse):
    status: int = 201
    message: str = "Agent deployed"
    thread_ids: dict[str, str] = Field(default_factory=dict)