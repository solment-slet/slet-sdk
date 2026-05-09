from dataclasses import dataclass, fields


@dataclass(slots=True)
class ApiPrefixes:
    identify: str = "identify/v1"
    aelite: str = "aelite/v1"

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, str):
                setattr(self, field.name, self._normalize(value))

    @staticmethod
    def _normalize(value: str) -> str:
        value = value.strip()

        if value.startswith("/"):
            value = value[1:]

        if value.endswith("/"):
            value = value[:-1]

        return value