from __future__ import annotations

import locale
import platform


def get_device_string() -> str:
    """
    Формат:
        "<device_name>, <OS>, <Region>"

    Где:
    - device_name → имя хоста (кроссплатформенно)
    - OS → система + версия
    - Region → locale

    Никогда не выбрасывает исключений.
    """

    try:
        device_name = platform.node() or "Unknown"
    except Exception:
        device_name = "Unknown"

    try:
        os_name = f"{platform.system()} {platform.release()}"
    except Exception:
        os_name = "Unknown"

    try:
        region = locale.getlocale()[0] or locale.getdefaultlocale()[0] or "Unknown"
    except Exception:
        region = "Unknown"

    return f"{device_name}, {os_name}, {region}"