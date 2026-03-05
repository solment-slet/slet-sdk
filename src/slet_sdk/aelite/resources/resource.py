from __future__ import annotations
from typing import Any, Dict, Sequence

from slet_sdk.core.mixins import BaseResource
from slet_sdk.aelite.agent import AgentSession
from slet_sdk.aelite.manifest import AgentManifest
from slet_sdk.aelite.schemas.agent import AgentDeployResponse
from slet_sdk.aelite.schemas.chats import ChatCreateResponse

# Resources
from .chats import ChatsResource
from .tts import TTSResource


class AeliteResource(BaseResource):
    """
    Namespace для работы с AElite.
    Доступен как client.aelite в slet_client
    """

    async def deploy(
        self,
        manifest: AgentManifest | Dict[str, Any],
        thread_id: str | None = None,
    ) -> AgentDeployResponse:
        """
        Деплой агента по манифесту на сервере.
        Возвращает thread ID каждого агента и подагента.

        :param manifest: Конфигурация агента
        :param thread_id: ID сессии (чата), если None создается новый
        """
        if not isinstance(manifest, dict):
            manifest = manifest.model_dump()

        if not thread_id:
            chat = await self.new_chat()
            thread_id = str(chat.id)

        return await self._request(
            "POST",
            f"/ae/agent/deploy/{thread_id}",
            schema=AgentDeployResponse,
            json=manifest,
        )

    async def deploy_and_connect(
            self,
            manifest: AgentManifest,
            thread_id: str | None = None,
    ) -> AgentSession:
        """
        Deploy + connect в одну операцию.
        Возвращает корневой AgentSession с деревом .sub[...].
        """
        response = await self.deploy(manifest, thread_id=thread_id)

        # Строим дерево сессий
        root = await self._build_session_tree(manifest, response.thread_ids)
        return root

    async def connect(
            self,
            thread_ids: str | Sequence[str] | AgentDeployResponse,
            manifest: AgentManifest | None = None,
    ) -> AgentSession | list[AgentSession]:
        """
        Создаёт WebSocket-сессии.

        Если передан manifest — возвращает дерево (один корневой AgentSession).
        Если передан просто список id без manifest — возвращает плоский список (обратная совместимость).
        """
        headers = {}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        if isinstance(thread_ids, AgentDeployResponse):
            id_map = thread_ids.thread_ids
            # Если есть манифест — строим дерево
            if manifest:
                return await self._build_session_tree(manifest, id_map)
            # Иначе — плоский список (backward compat)
            ids = list(id_map.values())
        elif isinstance(thread_ids, str):
            ids = [thread_ids]
        else:
            ids = list(thread_ids)

        if manifest:
            return await self._build_session_tree(manifest, {manifest.id: ids[0]})

        sessions = []
        for tid in ids:
            s = AgentSession(
                base_url=self._base_url, thread_id=tid,
                headers=headers, resource=self,
            )
            await s.connect()
            sessions.append(s)

        return sessions[0] if len(sessions) == 1 else sessions

    async def _build_session_tree(
            self,
            manifest: AgentManifest,
            thread_id_map: dict[str, str],
    ) -> AgentSession:
        """
        Рекурсивно строит дерево AgentSession по манифесту.
        thread_id_map: {"MainAgent": "abc123", "SherlokHolms": "def456", "Krosh": "ghi789"}
        """
        headers = {}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        tid = thread_id_map.get(manifest.id)
        if not tid:
            raise ValueError(
                f"No thread_id found for agent '{manifest.id}'. "
                f"Available: {list(thread_id_map.keys())}"
            )

        session = AgentSession(
            base_url=self._base_url,
            thread_id=tid,
            headers=headers,
            manifest=manifest,
            resource=self,
        )
        await session.connect()

        # Рекурсия по подагентам
        for sub_manifest in manifest.sub_agents:
            sub_session = await self._build_session_tree(sub_manifest, thread_id_map)
            session.sub[sub_manifest.id] = sub_session

        return session

    async def new_chat(self, title: str | None = None) -> ChatCreateResponse:
        """Создание нового чата (бизнес модель, а не реальный чат, нужно для сопоставления чата с его владельцем)"""
        return await self._request(
            "POST",
            "/ae/chats/",
            json={"title": title} if title else None,
            schema=ChatCreateResponse,
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chats = ChatsResource(self)
        self.tts = TTSResource(self)