"""Minimal identity and safety differences between supported actuators."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamixelModelSpec:
    """Values that genuinely differ and are consumed by runtime code."""

    name: str
    model_number: int
    factory_baudrate: int
    max_current_limit_ma: int


XM335_T323 = DynamixelModelSpec(
    name="xm335-t323-t",
    model_number=1710,
    factory_baudrate=1_000_000,
    max_current_limit_ma=910,
)

XL330_M077 = DynamixelModelSpec(
    name="xl330-m077-t",
    model_number=1190,
    factory_baudrate=57_600,
    max_current_limit_ma=1750,
)


_MODEL_ALIASES = {
    "xm335-t323-t": XM335_T323,
    "xm335-t323": XM335_T323,
    "xl330-m077-t": XL330_M077,
    "xl330-m077": XL330_M077,
}


def normalize_model_name(name: str) -> str:
    return str(name).strip().lower().replace("_", "-")


def get_model_spec(name: str) -> DynamixelModelSpec:
    normalized = normalize_model_name(name)
    try:
        return _MODEL_ALIASES[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted({spec.name for spec in _MODEL_ALIASES.values()}))
        raise ValueError(
            f"Unsupported motor_model {name!r}; expected one of: {supported}"
        ) from exc


__all__ = [
    "DynamixelModelSpec",
    "XL330_M077",
    "XM335_T323",
    "get_model_spec",
    "normalize_model_name",
]
