# src/library_name/agents/resource.py

from __future__ import annotations
from typing import Any, Dict

from slet_sdk.core.base_resource import BaseResource
from slet_sdk.aelite.agent import AgentSession
from slet_sdk.aelite.schemas.agent import AgentManifest


class AeliteResource(BaseResource):
    """
    Namespace для работы с агентами.
    Доступен как client.agents
    """

    async def deploy(
        self,
        thread_id: str,
        manifest: AgentManifest | Dict[str, Any],
    ) -> dict:
        """
        Деплоит агента на сервере.

        :param thread_id: ID сессии (чата)
        :param manifest: Конфигурация агента
        """
        if not isinstance(manifest, dict):
            manifest = manifest.model_dump()

        return await self._request(
            "POST",
            f"/ae/agent/deploy/{thread_id}",
            json=manifest,
        )

    async def connect(self, thread_id: str) -> AgentSession:
        """
        Создаёт WebSocket-сессию для общения с агентом.

        :param thread_id: ID сессии
        :return: AgentSession
        """
        ws_base = self._base_url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        ws_url = f"{ws_base}/ae/agent/ws/{thread_id}"

        headers = {}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        session = AgentSession(ws_url, headers)
        await session.connect()
        return session

    async def list(self, thread_id: str) -> dict:
        """Получить список агентов в сессии."""
        return await self._request("GET", f"/ae/agent/{thread_id}")

    async def delete(self, thread_id: str, agent_id: str) -> dict:
        """Удалить агента."""
        return await self._request("DELETE", f"/ae/agent/{thread_id}/{agent_id}")
