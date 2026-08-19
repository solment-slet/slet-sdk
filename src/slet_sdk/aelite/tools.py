"""
tools.py - client-side tool declarations.

A "client tool" is a Python callable that the *server* can ask the client
to execute. Declaring one only attaches metadata to the function (name,
description, parameter schema); it does not run it and does not register
it anywhere global.

Two ways to declare tools
--------------------------

1. Stand-alone ``@tool`` - attaches metadata only. You are responsible
   for handing the function to both the manifest's ``tools=[...]`` list
   *and* ``session.register_tools([...])``::

       @tool
       async def get_clipboard() -> str:
           \"\"\"Returns the user's current clipboard contents.\"\"\"
           ...

       manifest = AgentManifest(..., tools=[get_clipboard])
       session.register_tools([get_clipboard])

2. ``ToolBelt`` - an explicit, non-global registry. Recommended for
   anything more than one or two tools, since it removes the duplication
   above and the risk of forgetting to register a tool on the session::

       my_belt = ToolBelt()

       @my_belt.tool
       async def get_clipboard() -> str:
           \"\"\"Returns the user's current clipboard contents.\"\"\"
           ...

       @my_belt.tool
       async def show_notification(title: str, message: str) -> None:
           \"\"\"Shows a desktop notification to the user.\"\"\"
           ...

       manifest = AgentManifest(..., tools=[my_belt])   # flattened automatically
       session.register_tools([my_belt])                # flattened automatically

There is intentionally no process-wide registry. Two unrelated
``ToolBelt`` instances (or a bare ``@tool`` function) never collide just
because they happen to share a tool name - a collision is only possible
*within* the same belt, where it's caught immediately at decoration time.
"""

from __future__ import annotations

import inspect
from typing import Annotated, Any, Callable, Iterator, TYPE_CHECKING, get_args, get_origin, get_type_hints

if TYPE_CHECKING:
    from slet_sdk.aelite.manifest import BroadcastConfig

# ---------------------------------------------------------------------------
# Supported parameter types
# ---------------------------------------------------------------------------

_TYPE_TO_STR: dict[type, str] = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}


def _split_annotated(annotation: Any) -> tuple[Any, str | None]:
    """Unwrap ``Annotated[T, "description"]`` -> (T, description-or-None)."""
    if get_origin(annotation) is Annotated:
        base, *metadata = get_args(annotation)
        desc = next((m for m in metadata if isinstance(m, str)), None)
        return base, desc
    return annotation, None


def _build_tool_metadata(
    fn: Callable[..., Any],
    name: str | None,
    description: str | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    """Inspect ``fn`` and build the metadata dict later attached to it."""
    tool_name = name or fn.__name__
    tool_desc = description or fn.__doc__
    if not tool_desc:
        raise ValueError(
            f"Tool '{tool_name}' has no description. Provide one via "
            f"@tool(description=...) or a docstring."
        )

    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)

    params: dict[str, dict[str, Any]] = {}
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        annotation = hints.get(param_name, str)
        py_type, param_desc = _split_annotated(annotation)

        if py_type not in _TYPE_TO_STR:
            supported = ", ".join(t.__name__ for t in _TYPE_TO_STR)
            raise TypeError(
                f"Tool '{tool_name}', parameter '{param_name}': unsupported "
                f"type {py_type!r}. Supported types: {supported}."
            )
        type_str = _TYPE_TO_STR[py_type]

        has_default = param.default is not inspect.Parameter.empty
        params[param_name] = {
            "type": type_str,
            "description": param_desc,  # None if not given via Annotated[...]
            "required": not has_default,
            "default": param.default if has_default else None,
        }

    return {
        "_tool_name": tool_name,
        "_tool_description": tool_desc.strip(),
        "_tool_params": params,
        "_tool_extra": extra,
        "_is_client_tool": True,
    }


def _attach_metadata(fn: Callable[..., Any], metadata: dict[str, Any]) -> Callable[..., Any]:
    for key, value in metadata.items():
        setattr(fn, key, value)
    return fn


# ---------------------------------------------------------------------------
# Stand-alone decorator (no registry)
# ---------------------------------------------------------------------------


def tool(
    _fn: Callable[..., Any] = None,
    *,
    name: str = None,
    description: str = None,
    timeout: float | None = None,
    broadcast: BroadcastConfig | None = None,
):
    """
    Attach client-tool metadata to a function without registering it
    anywhere. Usable as ``@tool``, ``@tool()``, or
    ``@tool(name=..., description=..., timeout=..., broaadcast=...)``.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        extra: dict[str, Any] = {}
        if timeout is not None:
            extra["timeout"] = timeout
        if broadcast is not None:
            extra["broadcast"] = broadcast

        metadata = _build_tool_metadata(fn, name, description, extra)
        return _attach_metadata(fn, metadata)

    if _fn is not None and callable(_fn):
        return decorator(_fn)
    return decorator


# ---------------------------------------------------------------------------
# ToolBelt - explicit, non-global tool registry
# ---------------------------------------------------------------------------


class ToolBelt:
    """
    A self-contained collection of client tools.

    Create as many as you like; there's no shared global state, so two
    belts (or modules, or plugins) never collide with each other just
    because they use the same tool name. Both ``AgentManifest(tools=[...])``
    and ``AgentSession.register_tools([...])`` accept ``ToolBelt``
    instances directly and flatten them into their member tools.
    """

    def __init__(self, name: str | None = None) -> None:
        self.name = name
        self._tools: dict[str, Callable[..., Any]] = {}

    def tool(
        self,
        _fn: Callable[..., Any] = None,
        *,
        name: str = None,
        description: str = None,
        timeout: float | None = None,
        broadcast: BroadcastConfig | None = None,
    ):
        """Same usage as the module-level ``tool`` decorator, but registers
        the function on this belt. Raises if the name is already taken on
        this belt."""

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            extra: dict[str, Any] = {}
            if timeout is not None:
                extra["timeout"] = timeout
            if broadcast is not None:
                extra["broadcast"] = broadcast

            metadata = _build_tool_metadata(fn, name, description, extra)
            tool_name = metadata["_tool_name"]
            if tool_name in self._tools:
                label = f" '{self.name}'" if self.name else ""
                raise ValueError(
                    f"Tool '{tool_name}' is already registered on this "
                    f"ToolBelt{label}."
                )
            _attach_metadata(fn, metadata)
            self._tools[tool_name] = fn
            return fn

        if _fn is not None and callable(_fn):
            return decorator(_fn)
        return decorator

    def __iter__(self) -> Iterator[Callable[..., Any]]:
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        label = f" '{self.name}'" if self.name else ""
        return f"<ToolBelt{label}: {list(self._tools)}>"