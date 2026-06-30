from __future__ import annotations
from typing import Any, Dict

from slet_sdk.core.mixins import BaseResource
from slet_sdk.aelite.agent import AgentSession
from slet_sdk.aelite.manifest import AgentManifest
from slet_sdk.aelite.schemas.agents import AgentDeployResponse
from slet_sdk.aelite.typing import StreamMode

# Sub Resources for AeliteResource
from .threads import ThreadsResource
from .tts import TTSResource


class AeliteResource(BaseResource):
    """
    Namespace для работы с AElite.
    Доступен как client.aelite в slet_client
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.threads = ThreadsResource(*args, **kwargs)
        self.tts = TTSResource(*args, **kwargs)

    async def deploy(
        self,
        manifest: AgentManifest | Dict[str, Any],
        thread_id: str | None = None,
    ) -> AgentDeployResponse:
        """
        Деплой агента по манифесту на сервере.
        Возвращает thread ID каждого агента и подагента.

        :param manifest: Конфигурация агента
        :param thread_id: ID треда, если None создается новый
        """
        if not isinstance(manifest, dict):
            manifest = manifest.model_dump()

        if not thread_id:
            thread = await self.threads.create_thread()
            thread_id = str(thread.id)

        return await self._request(
            "POST",
            f"/agent/{thread_id}",
            schema=AgentDeployResponse,
            json=manifest,
        )

    async def connect(
        self,
        thread_id: str | dict[str, str] | AgentDeployResponse,
        manifest: AgentManifest | None = None,
        *,
        device: str | None = None,
        stream_mode: StreamMode | str = StreamMode.tokens,
        recursion: bool = True,
        register_tools_from_register: bool = True,
    ) -> AgentSession:
        """Установить соединение с агентом и вернуть сессию.

        Строит дерево сессий для root-агента и, опционально, всех его подагентов.
        Если манифест не передан — загружается с сервера автоматически.

        Args:
            thread_id: Идентификатор треда. Принимается в трёх форматах:
                - ``str`` — ID root-агента; словарь thread_id для подагентов
                  будет собран автоматически на основе манифеста.
                - ``dict[str, str]`` — готовый словарь вида ``{agent_name: thread_id}``.
                - ``AgentDeployResponse`` — объект ответа деплоя; словарь
                  извлекается из поля ``agent_ids``.
            manifest: Манифест, использованный при деплое агента.
                Если не передан, запрашивается у сервера по ``thread_id``.
            device: Произвольная строка-идентификатор устройства клиента.
                Позволяет агенту различать подключения одного пользователя
                с разных устройств. Если не указан — генерируется автоматически.
            stream_mode: Способ стриминга ответов агента.
            recursion: Если ``True`` (по умолчанию), подключаются root-агент
                и все подагенты. Если ``False`` — только root-агент.
            register_tools_from_register: Если ``True`` (по умолчанию),
                инструменты из глобального реестра (задекорированные ``@tool``)
                регистрируются в сессии автоматически.

        Returns:
            Сессия ``AgentSession`` для взаимодействия с агентом.
            При включённой рекурсии содержит дерево подагентов.

        Raises:
            TypeError: Если ``thread_id`` передан в неподдерживаемом типе
                (не ``str``, ``dict`` или ``AgentDeployResponse``).

        Example:
            .. code-block:: python

                # Подключение по строковому ID (манифест загрузится автоматически)
                session = await client.connect("thread-abc123")

                # Подключение с готовым словарём и без рекурсии
                session = await client.connect(
                    {"root": "thread-abc123", "sub": "thread-xyz456"},
                    recursion=False,
                )

                # Подключение из ответа деплоя
                deploy = await client.deploy(...)
                session = await client.connect(deploy)
        """

        if manifest is None:
            manifest = await self._request(
                "GET",
                f"/agent/{thread_id}",
                schema=AgentManifest,
            )

        if isinstance(thread_id, AgentDeployResponse):
            id_map = thread_id.agent_ids
        elif isinstance(thread_id, dict):
            id_map = thread_id
        elif isinstance(thread_id, str):
            id_map = self._collect_thread_ids(manifest, thread_id)
        else:
            raise TypeError("thread_id must be str, dict or AgentDeployResponse")

        # Строим дерево сессий
        root = await self._build_session_tree(
            manifest,
            id_map,
            recursion=recursion,
            device=device,
            stream_mode=stream_mode,
        )
        if register_tools_from_register:
            root.register_tools_from_registry()

        return root

    async def deploy_and_connect(
        self,
        manifest: AgentManifest,
        *,
        thread_id: str | None = None,
        device: str | None = None,
        stream_mode: StreamMode | str = StreamMode.tokens,
        recursion: bool = True,
        register_tools_from_register: bool = True,
    ) -> AgentSession:
        """
        Deploy + connect в одну операцию.
        """
        response = await self.deploy(manifest, thread_id=thread_id)

        return await self.connect(
            response.agent_ids,
            manifest,
            device=device,
            stream_mode=stream_mode,
            recursion=recursion,
            register_tools_from_register=register_tools_from_register,
        )

    async def _build_session_tree(
        self,
        manifest: AgentManifest,
        thread_id_map: dict[str, str],
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

        tid = thread_id_map.get(manifest.id)
        if not tid:
            raise ValueError(
                f"No thread_id found for agent '{manifest.id}'. "
                f"Available: {list(thread_id_map.keys())}"
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
                    thread_id_map,
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
