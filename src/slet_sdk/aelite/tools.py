import inspect
from typing import Callable, Dict, Any, get_type_hints


# Глобальный реестр клиентских тулов
_REGISTERED_CLIENT_TOOLS: Dict[str, Callable] = {}

# Маппинг Python-типов → строковых типов для ToolParam
_TYPE_TO_STR = {
    str: "str",
    int: "int",
    float: "float",
    bool: "bool",
}


def tool(name: str = None, description: str = None, **extra):
    """
    Декоратор для клиентского инструмента.
    Дополнительные kwargs (broadcast, и т.д.) попадают в манифест автоматически.

    Использование:
        @tool()
        def get_clipboard() -> str:
            '''Возвращает буфер обмена'''
            ...

        @tool(broadcast=True)
        def show_notification(title: str, message: str) -> str:
            '''Показывает уведомление на всех устройствах'''
            ...
    """
    def decorator(fn: Callable) -> Callable:
        tool_name = name or fn.__name__
        tool_desc = description or fn.__doc__

        if not tool_desc:
            raise ValueError("Missing tool description")

        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        
        params: Dict[str, Dict[str, Any]] = {}
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            py_type = hints.get(param_name, str)
            type_str = _TYPE_TO_STR.get(py_type, "str")

            has_default = param.default is not inspect.Parameter.empty
            
            params[param_name] = {
                "type": type_str,
                "description": "",
                "required": not has_default,
                "default": param.default if has_default else None,
            }

        fn._tool_name = tool_name
        fn._tool_description = tool_desc.strip()
        fn._tool_params = params
        fn._tool_extra = extra  # ← сохраняем все доп. параметры
        fn._is_client_tool = True

        _REGISTERED_CLIENT_TOOLS[tool_name] = fn
        return fn

    return decorator


def get_registered_tools() -> Dict[str, Callable]:
    """Возвращает все зарегистрированные клиентские tools."""
    return _REGISTERED_CLIENT_TOOLS.copy()