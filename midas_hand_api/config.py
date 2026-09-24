"""Configuration objects for the Midas hand Dynamixel API."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field, replace
from math import pi
from typing import Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import yaml  # pyyaml

from .actuators import control_table as ct
from .actuators.models import XL330_M077, XM335_T323, DynamixelModelSpec, get_model_spec


DEFAULT_MOTOR_IDS = tuple(range(0, 13))
DEFAULT_DYNAMIXEL_BAUDRATE = 4_000_000

# Default path for persisting hand calibration across sessions.
# Lives in the user's home directory so it survives pip upgrades and
# is never inside the git working tree.
DEFAULT_CONFIG_PATH: pathlib.Path = pathlib.Path.home() / ".midas_hand" / "config.yaml"

# Indices into the 16-DOF joint-space array where passive DIP joints sit.
# Layout: thumb(0-3), index(4,DIP@5,6,7), middle(8,DIP@9,10,11), ring(12,DIP@13,14,15).
DEFAULT_PASSIVE_JOINT_INDICES: Tuple[int, ...] = (5, 9, 13)

# 0-indexed positions in the 13-motor array for the PIP motors that drive each
# passive DIP (same order as DEFAULT_PASSIVE_JOINT_INDICES).
# Motor IDs 4, 7, 10 (index/middle/ring PIP) sit at array positions 4, 7, 10.
DEFAULT_PIP_MOTOR_INDICES: Tuple[int, ...] = (4, 7, 10)
_MOTOR_VALUE_FIELDS = (
    "home_offsets",
    "joint_signs",
    "joint_lower_limits",
    "joint_upper_limits",
)
HomingEntry = Tuple[int, str, float, int]
DEFAULT_THUMB_HOMING_TABLE: Tuple[HomingEntry, ...] = (
    (0, "Thumb IP", 1.61221381, -1),
    (1, "Thumb MCP", 1.4450099, -1),
    (2, "Thumb CMC side", -0.88510692, 1),
    (3, "Thumb CMC roll", 0.0, -1),
)
DEFAULT_FINGER_HOMING_TABLE: Tuple[HomingEntry, ...] = (
    (4, "Index PIP", -0.03, 1),
    (5, "Index MCP flex", -0.05, 1),
    (6, "Index MCP abduct", 0.81, -1),
    (7, "Middle PIP", -0.03, 1),
    (8, "Middle MCP flex", -0.05, 1),
    (9, "Middle MCP abduct", 0.8, -1),
    (10, "Ring PIP", -0.03, 1),
    (11, "Ring MCP flex", -0.05, 1),
    (12, "Ring MCP abduct", -0.81, 1),
)


def _repeat(value: float, n: int) -> Tuple[float, ...]:
    return tuple(value for _ in range(n))


def _coerce_motor_values(
    values: Union[Sequence[float], Mapping[Union[int, str], float]],
    motor_ids: Sequence[int],
    name: str,
) -> Tuple[float, ...]:
    """Return motor-indexed values ordered to match ``motor_ids``.

    Saved configs may store calibration fields as ``{motor_id: value}`` for
    robustness. Runtime math uses tuples/arrays aligned with ``motor_ids``.
    """

    if isinstance(values, Mapping):
        ordered = []
        for motor_id in motor_ids:
            if motor_id in values:
                ordered.append(float(values[motor_id]))
            elif str(motor_id) in values:
                ordered.append(float(values[str(motor_id)]))
            else:
                raise ValueError(f"{name} is missing value for motor ID {motor_id}")
        return tuple(ordered)

    values_tuple = tuple(float(value) for value in values)
    if len(values_tuple) == len(motor_ids):
        return values_tuple
    if len(values_tuple) == len(DEFAULT_MOTOR_IDS):
        default_index_by_id = {
            motor_id: index for index, motor_id in enumerate(DEFAULT_MOTOR_IDS)
        }
        if all(motor_id in default_index_by_id for motor_id in motor_ids):
            return tuple(
                values_tuple[default_index_by_id[motor_id]]
                for motor_id in motor_ids
            )
        if len(set(values_tuple)) == 1:
            return tuple(values_tuple[0] for _ in motor_ids)
    return values_tuple


def _motor_value_map(
    motor_ids: Sequence[int], values: Sequence[float]
) -> dict[int, float]:
    return {int(motor_id): float(value) for motor_id, value in zip(motor_ids, values)}


def _coerce_homing_table(
    values: Sequence[Sequence[object]], name: str
) -> Tuple[HomingEntry, ...]:
    entries = []
    for value in values:
        if len(value) != 4:
            raise ValueError(
                f"each {name} entry must be [motor_id, joint_name, cad_offset_rad, direction]"
            )
        motor_id, joint_name, cad_offset_rad, direction = value
        entries.append(
            (int(motor_id), str(joint_name), float(cad_offset_rad), int(direction))
        )
    return tuple(entries)


def _filtered_passive_topology(
    source_n_motors: int,
    source_passive_joint_indices: Sequence[int],
    source_pip_motor_indices: Sequence[int],
    selected_motor_indices: Sequence[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    selected_old_to_new = {
        old_index: new_index
        for new_index, old_index in enumerate(selected_motor_indices)
    }
    passive_by_dof = dict(zip(source_passive_joint_indices, source_pip_motor_indices))

    passive_joint_indices = []
    pip_motor_indices = []
    motor_index = 0
    filtered_dof_index = 0
    for dof_index in range(source_n_motors + len(source_passive_joint_indices)):
        if dof_index in passive_by_dof:
            pip_motor_index = passive_by_dof[dof_index]
            if pip_motor_index in selected_old_to_new:
                passive_joint_indices.append(filtered_dof_index)
                pip_motor_indices.append(selected_old_to_new[pip_motor_index])
                filtered_dof_index += 1
        else:
            if motor_index in selected_old_to_new:
                filtered_dof_index += 1
            motor_index += 1

    return tuple(passive_joint_indices), tuple(pip_motor_indices)


@dataclass(frozen=True)
class HandConfig:
    """Runtime and mechanical configuration for a supported hand."""

    motor_model: str = XM335_T323.name
    motor_ids: Tuple[int, ...] = DEFAULT_MOTOR_IDS
    port: Optional[str] = None
    baudrate: int = DEFAULT_DYNAMIXEL_BAUDRATE

    # Topology: passive joints and their driving PIP motors.
    # passive_joint_indices: positions in the full n_dof joint array where
    #   passive (linkage-driven) DIP joints are inserted.
    # pip_motor_indices: index into the motor array for the PIP joint that
    #   drives each corresponding passive DIP (same order).
    passive_joint_indices: Tuple[int, ...] = DEFAULT_PASSIVE_JOINT_INDICES
    pip_motor_indices: Tuple[int, ...] = DEFAULT_PIP_MOTOR_INDICES

    position_p_gain: int = 800
    position_i_gain: int = 0
    position_d_gain: int = 500
    goal_current_limit: int = 600
    operating_mode: int = ct.OPERATING_MODE_CURRENT_BASED_POSITION

    home_offsets: Sequence[float] = field(
        default_factory=lambda: _repeat(0.0, len(DEFAULT_MOTOR_IDS))
    )
    joint_signs: Sequence[float] = field(
        default_factory=lambda: _repeat(1.0, len(DEFAULT_MOTOR_IDS))
    )
    joint_lower_limits: Sequence[float] = field(
        default_factory=lambda: _repeat(-pi, len(DEFAULT_MOTOR_IDS))
    )
    joint_upper_limits: Sequence[float] = field(
        default_factory=lambda: _repeat(pi, len(DEFAULT_MOTOR_IDS))
    )
    pip_dip_lookup: Optional[str] = None
    thumb_homing_table: Tuple[HomingEntry, ...] = DEFAULT_THUMB_HOMING_TABLE
    finger_homing_table: Tuple[HomingEntry, ...] = DEFAULT_FINGER_HOMING_TABLE
    middle_mcp_flex_pre_abduct_position_rad: float = -0.5 * pi

    def __post_init__(self) -> None:
        spec = get_model_spec(self.motor_model)
        object.__setattr__(self, "motor_model", spec.name)
        motor_ids = tuple(int(motor_id) for motor_id in self.motor_ids)
        object.__setattr__(self, "motor_ids", motor_ids)
        if (
            tuple(self.passive_joint_indices) == DEFAULT_PASSIVE_JOINT_INDICES
            and tuple(self.pip_motor_indices) == DEFAULT_PIP_MOTOR_INDICES
        ):
            selected_default_indices = [
                DEFAULT_MOTOR_IDS.index(motor_id)
                for motor_id in motor_ids
                if motor_id in DEFAULT_MOTOR_IDS
            ]
            passive_joint_indices, pip_motor_indices = _filtered_passive_topology(
                len(DEFAULT_MOTOR_IDS),
                DEFAULT_PASSIVE_JOINT_INDICES,
                DEFAULT_PIP_MOTOR_INDICES,
                selected_default_indices,
            )
            object.__setattr__(self, "passive_joint_indices", passive_joint_indices)
            object.__setattr__(self, "pip_motor_indices", pip_motor_indices)
        for name in _MOTOR_VALUE_FIELDS:
            object.__setattr__(
                self,
                name,
                _coerce_motor_values(getattr(self, name), motor_ids, name),
            )
        object.__setattr__(
            self,
            "thumb_homing_table",
            _coerce_homing_table(self.thumb_homing_table, "thumb_homing_table"),
        )
        object.__setattr__(
            self,
            "finger_homing_table",
            _coerce_homing_table(self.finger_homing_table, "finger_homing_table"),
        )
        if self.pip_dip_lookup is not None:
            object.__setattr__(
                self,
                "pip_dip_lookup",
                str(pathlib.Path(self.pip_dip_lookup).expanduser()),
            )

    @property
    def _model_spec(self) -> DynamixelModelSpec:
        return get_model_spec(self.motor_model)

    @property
    def expected_model_number(self) -> int:
        return self._model_spec.model_number

    @property
    def max_current_limit_ma(self) -> int:
        return self._model_spec.max_current_limit_ma

    @property
    def n_motors(self) -> int:
        """Number of active motors (= len(motor_ids))."""
        return len(self.motor_ids)

    @property
    def n_dof(self) -> int:
        """Total joint-space DOF, including passive linkage joints."""
        return self.n_motors + len(self.passive_joint_indices)

    @property
    def position_scale(self) -> float:
        """Radians per encoder count."""

        return ct.X_SERIES_POSITION_UNIT_RAD

    @property
    def velocity_scale(self) -> float:
        """Radians/sec per raw velocity unit."""

        return ct.X_SERIES_VELOCITY_UNIT_RAD_S

    @property
    def current_unit_ma(self) -> float:
        """mA per raw current unit."""

        return ct.X_SERIES_CURRENT_UNIT_MA

    @property
    def home_offsets_array(self) -> np.ndarray:
        return np.asarray(self.home_offsets, dtype=np.float64)

    @property
    def joint_signs_array(self) -> np.ndarray:
        return np.asarray(self.joint_signs, dtype=np.float64)

    @property
    def joint_lower_limits_array(self) -> np.ndarray:
        return np.asarray(self.joint_lower_limits, dtype=np.float64)

    @property
    def joint_upper_limits_array(self) -> np.ndarray:
        return np.asarray(self.joint_upper_limits, dtype=np.float64)

    def validate(self) -> None:
        n = len(self.motor_ids)
        if n == 0:
            raise ValueError("motor_ids must not be empty")
        if len(set(self.motor_ids)) != n:
            raise ValueError("motor_ids must be unique")
        if any(motor_id < 0 or motor_id > 252 for motor_id in self.motor_ids):
            raise ValueError("motor_ids must be in the range 0..252")
        if self.baudrate <= 0:
            raise ValueError("baudrate must be greater than zero")
        for name in _MOTOR_VALUE_FIELDS:
            values = getattr(self, name)
            if len(values) != n:
                raise ValueError(
                    f"{name} must have {n} values (one per motor), got {len(values)}"
                )
            if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
                raise ValueError(f"{name} must contain only finite values")
        if not np.all(np.isin(self.joint_signs_array, (-1.0, 1.0))):
            raise ValueError("joint_signs values must be -1 or 1")
        if np.any(self.joint_lower_limits_array >= self.joint_upper_limits_array):
            raise ValueError("each joint lower limit must be below its upper limit")

        spec = self._model_spec
        if self.operating_mode not in ct.X_SERIES_SUPPORTED_OPERATING_MODES:
            raise ValueError(
                f"operating_mode {self.operating_mode} is unsupported by {spec.name}"
            )
        for name in ("position_p_gain", "position_i_gain", "position_d_gain"):
            value = getattr(self, name)
            if value != int(value) or not 0 <= value <= ct.X_SERIES_MAX_GAIN:
                raise ValueError(
                    f"{name} must be an integer in 0..{ct.X_SERIES_MAX_GAIN}"
                )
        if not 0 <= self.goal_current_limit <= spec.max_current_limit_ma:
            raise ValueError(
                "goal_current_limit must be between 0 and "
                f"{spec.max_current_limit_ma} mA for {spec.name}"
            )

        if len(self.passive_joint_indices) != len(self.pip_motor_indices):
            raise ValueError(
                "passive_joint_indices and pip_motor_indices must have the same length"
            )
        for idx in self.pip_motor_indices:
            if idx < 0 or idx >= n:
                raise ValueError(
                    "pip_motor_indices contains "
                    f"{idx}, but only {n} motors are configured"
                )
        if any(idx < 0 or idx >= self.n_dof for idx in self.passive_joint_indices):
            raise ValueError("passive_joint_indices contains an invalid DOF index")

        homing_entries = self.thumb_homing_table + self.finger_homing_table
        if len(self.thumb_homing_table) not in (0, 4):
            raise ValueError("homing.thumb must contain 4 entries or be empty")
        if len(self.finger_homing_table) not in (0, 9):
            raise ValueError("homing.fingers must contain 9 entries or be empty")
        entry_ids = [entry[0] for entry in homing_entries]
        if len(set(entry_ids)) != len(entry_ids):
            raise ValueError("homing tables must contain unique motor IDs")
        for motor_id, joint_name, cad_offset, direction in homing_entries:
            if not 0 <= motor_id <= 252:
                raise ValueError("homing motor IDs must be in the range 0..252")
            if direction not in (-1, 1):
                raise ValueError("homing direction must be -1 or 1")
            if not joint_name or not np.isfinite(cad_offset):
                raise ValueError("homing names must be non-empty and offsets finite")
        if not np.isfinite(self.middle_mcp_flex_pre_abduct_position_rad):
            raise ValueError("middle MCP flex pre-abduction position must be finite")
        if self.pip_dip_lookup is not None and not pathlib.Path(self.pip_dip_lookup).exists():
            raise ValueError(f"pip_dip_lookup does not exist: {self.pip_dip_lookup}")

    @classmethod
    def xm335_t323(
        cls,
        motor_ids: Tuple[int, ...] = DEFAULT_MOTOR_IDS,
        port: Optional[str] = None,
        baudrate: int = DEFAULT_DYNAMIXEL_BAUDRATE,
        **overrides,
    ) -> "HandConfig":
        """Backward-compatible constructor; ``HandConfig()`` is XM335-only."""

        for key in (
            "motor_model",
            "expected_model_number",
            "max_goal_current_limit",
            "counts_per_rev",
            "velocity_unit_rpm",
            "current_unit_ma",
        ):
            overrides.pop(key, None)
        values = {
            "motor_model": XM335_T323.name,
            "motor_ids": motor_ids,
            "port": port,
            "baudrate": baudrate,
        }
        values.update(overrides)
        return cls(**values)

    @classmethod
    def xl330_m077(
        cls,
        motor_ids: Tuple[int, ...] = DEFAULT_MOTOR_IDS,
        port: Optional[str] = None,
        baudrate: int = XL330_M077.factory_baudrate,
        **overrides,
    ) -> "HandConfig":
        """Construct an XL330 config with conservative, explicit tuning defaults."""

        overrides.pop("motor_model", None)
        values = {
            "motor_model": XL330_M077.name,
            "motor_ids": motor_ids,
            "port": port,
            "baudrate": baudrate,
            "position_p_gain": 400,
            "position_i_gain": 0,
            "position_d_gain": 400,
            "goal_current_limit": 300,
            # Hard-stop directions and CAD offsets are hand-specific.
            "thumb_homing_table": (),
            "finger_homing_table": (),
        }
        values.update(overrides)
        return cls(**values)

    def subset(self, motor_ids: Sequence[int]) -> "HandConfig":
        """Return this config narrowed to ``motor_ids`` while preserving calibration."""

        requested_ids = set(int(motor_id) for motor_id in motor_ids)
        selected_ids = tuple(
            motor_id for motor_id in self.motor_ids if motor_id in requested_ids
        )
        selected_indices = [
            index
            for index, motor_id in enumerate(self.motor_ids)
            if motor_id in requested_ids
        ]
        if len(selected_ids) != len(requested_ids):
            missing = sorted(requested_ids - set(selected_ids))
            raise ValueError(f"Cannot subset config for unknown motor IDs: {missing}")

        def select(values):
            return tuple(values[index] for index in selected_indices)

        passive_joint_indices, pip_motor_indices = _filtered_passive_topology(
            self.n_motors,
            self.passive_joint_indices,
            self.pip_motor_indices,
            selected_indices,
        )
        return replace(
            self,
            motor_ids=selected_ids,
            passive_joint_indices=passive_joint_indices,
            pip_motor_indices=pip_motor_indices,
            **{name: select(getattr(self, name)) for name in _MOTOR_VALUE_FIELDS},
        )

    def save(self, path: Union[str, pathlib.Path] = DEFAULT_CONFIG_PATH) -> None:
        """Save runtime, mechanical, and calibration configuration to YAML."""
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "motor_model": self.motor_model,
            "motor_ids": list(self.motor_ids),
            "port": self.port,
            "baudrate": self.baudrate,
            "operating_mode": self.operating_mode,
            "position_p_gain": self.position_p_gain,
            "position_i_gain": self.position_i_gain,
            "position_d_gain": self.position_d_gain,
            "goal_current_limit_ma": self.goal_current_limit,
            "passive_joint_indices": list(self.passive_joint_indices),
            "pip_motor_indices": list(self.pip_motor_indices),
            "home_offsets": _motor_value_map(self.motor_ids, self.home_offsets),
            "joint_signs": _motor_value_map(self.motor_ids, self.joint_signs),
            "joint_lower_limits": _motor_value_map(
                self.motor_ids, self.joint_lower_limits
            ),
            "joint_upper_limits": _motor_value_map(
                self.motor_ids, self.joint_upper_limits
            ),
            "pip_dip_lookup": self.pip_dip_lookup,
            "homing": {
                "thumb": [list(entry) for entry in self.thumb_homing_table],
                "fingers": [list(entry) for entry in self.finger_homing_table],
                "middle_mcp_flex_pre_abduct_position_rad": (
                    self.middle_mcp_flex_pre_abduct_position_rad
                ),
            },
        }
        with open(path, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

    def save_merged_offsets(
        self,
        motor_indices: Sequence[int],
        path: Union[str, pathlib.Path] = DEFAULT_CONFIG_PATH,
    ) -> None:
        """Merge selected offsets without discarding any unrelated YAML fields."""

        path = pathlib.Path(path)
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        else:
            self.save(path)
            with open(path) as f:
                data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError("configuration YAML root must be a mapping")
        saved_ids_value = data.get("motor_ids")
        if saved_ids_value is None:
            offset_value = data.get("home_offsets")
            if isinstance(offset_value, Mapping):
                saved_ids_value = list(offset_value)
            elif isinstance(offset_value, list):
                saved_ids_value = DEFAULT_MOTOR_IDS[: len(offset_value)]
            else:
                saved_ids_value = ()
        saved_motor_ids = [int(motor_id) for motor_id in saved_ids_value]
        existing_ids = list(saved_motor_ids)
        for motor_id in self.motor_ids:
            if int(motor_id) not in existing_ids:
                existing_ids.append(int(motor_id))

        merged_data = dict(data)
        merged_data.setdefault("motor_model", self.motor_model)
        merged_data["motor_ids"] = existing_ids

        for key in _MOTOR_VALUE_FIELDS:
            values = getattr(self, key)
            existing = data.get(key, {})
            if isinstance(existing, list):
                existing = {
                    int(motor_id): float(value)
                    for motor_id, value in zip(saved_motor_ids, existing)
                }
            elif np.isscalar(existing) and existing is not None:
                existing = {
                    int(motor_id): float(existing) for motor_id in saved_motor_ids
                }
            else:
                existing = {
                    int(motor_id): float(value)
                    for motor_id, value in (existing or {}).items()
                }

            if key == "home_offsets":
                indices_to_write = motor_indices
                for index in indices_to_write:
                    existing[int(self.motor_ids[index])] = float(values[index])
            else:
                for index, motor_id in enumerate(self.motor_ids):
                    existing.setdefault(int(motor_id), float(values[index]))
            merged_data[key] = existing

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(merged_data, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: Union[str, pathlib.Path] = DEFAULT_CONFIG_PATH) -> "HandConfig":
        """Load YAML as an overlay on the selected motor-model defaults."""
        path = pathlib.Path(path)
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError("configuration YAML root must be a mapping")

        model = get_model_spec(data.get("motor_model", XM335_T323.name))
        motor_ids = tuple(
            int(motor_id) for motor_id in data.get("motor_ids", DEFAULT_MOTOR_IDS)
        )
        factory = cls.xl330_m077 if model is XL330_M077 else cls.xm335_t323
        config = factory(motor_ids=motor_ids)
        updates = {"motor_model": model.name}
        direct_fields = (
            "port",
            "baudrate",
            "passive_joint_indices",
            "pip_motor_indices",
            "position_p_gain",
            "position_i_gain",
            "position_d_gain",
            "operating_mode",
        ) + _MOTOR_VALUE_FIELDS
        for key in direct_fields:
            if key in data:
                value = data[key]
                if key in ("passive_joint_indices", "pip_motor_indices"):
                    value = tuple(int(item) for item in value)
                updates[key] = value
        if "goal_current_limit_ma" in data:
            updates["goal_current_limit"] = data["goal_current_limit_ma"]
        elif "goal_current_limit" in data:
            updates["goal_current_limit"] = data["goal_current_limit"]
        homing = data.get("homing")
        if homing is not None:
            if not isinstance(homing, Mapping):
                raise ValueError("homing must be a mapping")
            if "thumb" in homing:
                updates["thumb_homing_table"] = homing["thumb"]
            if "fingers" in homing:
                updates["finger_homing_table"] = homing["fingers"]
            if "middle_mcp_flex_pre_abduct_position_rad" in homing:
                updates["middle_mcp_flex_pre_abduct_position_rad"] = float(
                    homing["middle_mcp_flex_pre_abduct_position_rad"]
                )
        if data.get("pip_dip_lookup") is not None:
            lookup = pathlib.Path(str(data["pip_dip_lookup"])).expanduser()
            if not lookup.is_absolute():
                lookup = path.parent / lookup
            updates["pip_dip_lookup"] = str(lookup.resolve())

        config = replace(config, **updates)
        config.validate()
        return config
