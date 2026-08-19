from __future__ import annotations

from uuid import UUID

from slet_sdk.core.mixins import BaseResource
from slet_sdk.aelite.agent import AgentSession
from slet_sdk.aelite.manifest import AgentManifest
from slet_sdk.aelite.typing import StreamMode
from slet_sdk.aelite.schemas.threads import ThreadInfo, ThreadPermissions, ThreadCreateResponse
from slet_sdk.aelite.schemas.agents import AgentInfo, AgentInfoWithManifest


# Sub Resources for AeliteResource
from .agents import AgentsResource
from .threads import ThreadsResource
from .tts import TTSResource


class AeliteResource(BaseResource):
    """
    Aelite Resource.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.agents = AgentsResource(*args, **kwargs)
        self.threads = ThreadsResource(*args, **kwargs)
        self.tts = TTSResource(*args, **kwargs)

    async def connect(
        self,
        session_id: UUID | str | dict[str, str] | ThreadInfo,
        *,
        device: str | None = None,
        stream_mode: StreamMode | str = StreamMode.tokens,
        recursion: bool = True,
        manifest: AgentManifest | None = None,
    ) -> AgentSession:
        """Установить соединение с сессией.

        Строит дерево сессий для root-агента и, опционально, всех его подагентов.
        Если манифест не передан - загружается с сервера автоматически.

        Args:
            session_id: Идентификатор треда. Принимается в трёх форматах:
                - ``str`` - ID root-агента; словарь thread_id для подагентов
                  будет собран автоматически на основе манифеста.
                - ``dict[str, str]`` - готовый словарь вида ``{agent_name: thread_id}``.
                - ``ThreadInfo`` - объект ответа деплоя; словарь
                  извлекается из поля ``agent_ids``.
            manifest: Манифест, использованный при деплое агента.
                Если не передан, запрашивается у сервера по ``thread_id``.
            device: Произвольная строка-идентификатор устройства клиента.
                Позволяет агенту различать подключения одного пользователя
                с разных устройств. Если не указан - генерируется автоматически.
            stream_mode: Способ стриминга ответов агента.
            recursion: Если ``True`` (по умолчанию), подключаются root-агент
                и все подагенты. Если ``False`` - только root-агент.

        Returns:
            Сессия ``AgentSession`` для взаимодействия с агентом.
            При включённой рекурсии содержит дерево подагентов.

        Raises:
            TypeError: Если ``thread_id`` передан в неподдерживаемом типе
                (не ``str``, ``dict`` или ``ThreadInfo``).
        """

        if manifest is None:
            thread = await self.threads.get_thread(
                UUID(session_id) if not isinstance(session_id, UUID) else session_id,
            )
            manifest = thread.manifest

        if isinstance(session_id, ThreadCreateResponse):
            id_map = session_id.session_ids
        elif isinstance(session_id, ThreadInfo):
            id_map = self._collect_thread_ids(manifest, str(session_id.id))
        elif isinstance(session_id, dict):
            id_map = session_id
        elif isinstance(session_id, str):
            id_map = self._collect_thread_ids(manifest, session_id)
        elif isinstance(session_id, UUID):
            id_map = self._collect_thread_ids(manifest, str(session_id))
        else:
            raise TypeError("session_id must be str, dict or ThreadInfo")

        root = await self._build_session_tree(
            manifest,
            id_map,
            recursion=recursion,
            device=device,
            stream_mode=stream_mode,
        )

        return root

    async def get_session_from_manifest(
        self,
        manifest: AgentManifest,
        *,
        # Thread
        title: str | None = None,
        permissions: ThreadPermissions | None = None,
        # Session
        device: str | None = None,
        stream_mode: StreamMode | str = StreamMode.tokens,
        recursion: bool = True,
    ) -> AgentSession:
        # Create agent
        agent = await self.agents.create_agent(manifest)

        # Return session from agent
        return await self.get_session_from_agent(
            agent,
            title=title,
            permissions=permissions,
            device=device,
            stream_mode=stream_mode,
            recursion=recursion,
            manifest=manifest,
        )


    async def get_session_from_agent(
        self,
        agent: UUID | AgentInfo,
        *,
        # Thread
        title: str | None = None,
        permissions: ThreadPermissions | None = None,
        # Session
        device: str | None = None,
        stream_mode: StreamMode | str = StreamMode.tokens,
        recursion: bool = True,
        manifest: AgentManifest | None = None,
    ) -> AgentSession:
        if isinstance(agent, UUID):
            agent_id = agent
        elif isinstance(agent, AgentInfo):
            agent_id = agent.id
        else:
            raise TypeError(f"Agent must be an instance of UUID or AgentInfo, got {type(agent).__name__!r}")

        # Create thread
        thread = await self.threads.create_thread(
            agent=agent_id,
            title=title,
            permissions=permissions,
        )

        if manifest is None and isinstance(agent, AgentInfoWithManifest):
            manifest = agent.manifest

        # Return session from thread
        return await self.get_session_from_thread(
            thread,
            device=device,
            stream_mode=stream_mode,
            recursion=recursion,
            manifest=manifest,
        )

    async def get_session_from_thread(
        self,
        thread: UUID | ThreadInfo,
        *,
        # Session
        device: str | None = None,
        stream_mode: StreamMode | str = StreamMode.tokens,
        recursion: bool = True,
        manifest: AgentManifest | None = None,
    ) -> AgentSession:
        if isinstance(thread, ThreadInfo):
            thread_id = thread.id
        elif isinstance(thread, UUID):
            thread_id = thread
        else:
            raise TypeError(f"Thread must be an instance of UUID or ThreadInfo, got {type(thread).__name__!r}")

        return await self.connect(
            thread_id,
            device=device,
            stream_mode=stream_mode,
            recursion=recursion,
            manifest=manifest,
        )

    async def _build_session_tree(
        self,
        manifest: AgentManifest,
        session_id_map: dict[str, str],
        *,
        recursion: bool = True,
        device: str | None = None,
        stream_mode: StreamMode | str = StreamMode.tokens,
    ) -> AgentSession:
        """
        Рекурсивно строит дерево AgentSession по манифесту.
        """
        headers = {}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        tid = session_id_map.get(manifest.id)
        if not tid:
            raise ValueError(
                f"No session_id found for agent '{manifest.id}'. "
                f"Available: {list(session_id_map.keys())}"
            )

        session = AgentSession(
            thread_id=tid,
            device=device,
            stream_mode=stream_mode,
            headers=headers,
            manifest=manifest,
            resource=self,
        )
        await session.connect()

        if recursion:
            # Рекурсия по подагентам
            for sub_manifest in manifest.sub_agents:
                sub_session = await self._build_session_tree(
                    sub_manifest,
                    session_id_map,
                    device=device,
                    stream_mode=stream_mode,
                )
                session.sub[sub_manifest.id] = sub_session

        return session

    def _collect_thread_ids(
        self, manifest: AgentManifest, thread_id: str
    ) -> dict[str, str]:
        result = {manifest.id: thread_id}
        for sub in manifest.sub_agents:
            sub_thread_id = f"{thread_id}_{sub.id}"
            result.update(self._collect_thread_ids(sub, sub_thread_id))
        return result
