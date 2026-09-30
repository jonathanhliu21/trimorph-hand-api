from dataclasses import replace

import pytest
import yaml

from midas_hand_api.actuators import control_table as ct
from midas_hand_api.actuators.models import (
    XL330_M077,
    XM335_T323,
    get_model_spec,
)
from midas_hand_api.config import HandConfig
from midas_hand_api.hand import MidasHand


def _write_yaml(path, data) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_supported_model_constants_and_aliases() -> None:
    assert (
        XM335_T323.model_number,
        XM335_T323.factory_baudrate,
        XM335_T323.max_current_limit_ma,
    ) == (1710, 1_000_000, 910)
    assert (
        XL330_M077.model_number,
        XL330_M077.factory_baudrate,
        XL330_M077.max_current_limit_ma,
    ) == (1190, 57_600, 1750)

    for alias in ("xl330-m077-t", "xl330-m077", "XL330_M077_T", "XL330_M077"):
        assert get_model_spec(alias) is XL330_M077
    for alias in ("xm335-t323-t", "xm335-t323", "XM335_T323_T", "XM335_T323"):
        assert get_model_spec(alias) is XM335_T323

    with pytest.raises(ValueError, match="Unsupported motor_model"):
        get_model_spec("not-a-motor")


def test_shared_x_series_units_are_used_by_both_configs() -> None:
    xm = HandConfig()
    xl = HandConfig.xl330_m077()

    assert xm.position_scale == xl.position_scale == ct.X_SERIES_POSITION_UNIT_RAD
    assert xm.velocity_scale == xl.velocity_scale == ct.X_SERIES_VELOCITY_UNIT_RAD_S
    assert xm.current_unit_ma == xl.current_unit_ma == ct.X_SERIES_CURRENT_UNIT_MA


def test_xl330_constructor_uses_safe_model_specific_defaults() -> None:
    config = HandConfig.xl330_m077()

    assert config.motor_model == "xl330-m077-t"
    assert config.expected_model_number == 1190
    assert config.max_current_limit_ma == 1750
    assert config.baudrate == 57_600
    assert (
        config.position_p_gain,
        config.position_i_gain,
        config.position_d_gain,
    ) == (400, 0, 400)
    assert config.goal_current_limit == 300
    assert config.thumb_homing_table == ()
    assert config.finger_homing_table == ()


def test_existing_xm335_defaults_remain_unchanged() -> None:
    config = HandConfig()

    assert config.motor_model == "xm335-t323-t"
    assert config.expected_model_number == 1710
    assert config.max_current_limit_ma == 910
    assert config.baudrate == 4_000_000
    assert (
        config.position_p_gain,
        config.position_i_gain,
        config.position_d_gain,
    ) == (800, 0, 500)
    assert config.goal_current_limit == 600
    assert len(config.thumb_homing_table) == 4
    assert len(config.finger_homing_table) == 9


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (HandConfig(motor_ids=(0, 0)), "unique"),
        (HandConfig(motor_ids=(253,)), "0..252"),
        (HandConfig(motor_ids=(0,), joint_signs=(0,)), "-1 or 1"),
        (
            HandConfig(
                motor_ids=(0,),
                joint_lower_limits=(1,),
                joint_upper_limits=(0,),
            ),
            "lower limit",
        ),
        (HandConfig(motor_ids=(0,), position_p_gain=-1), "position_p_gain"),
        (HandConfig(motor_ids=(0,), operating_mode=2), "operating_mode"),
        (
            HandConfig.xl330_m077(motor_ids=(0,), goal_current_limit=1751),
            "1750",
        ),
        (HandConfig(motor_ids=(0,), goal_current_limit=911), "910"),
        (
            HandConfig(motor_ids=(0, 1), home_offsets=(0.0,)),
            "home_offsets",
        ),
        (
            HandConfig(motor_ids=(0,), home_offsets=(float("nan"),)),
            "finite",
        ),
    ],
)
def test_config_validation_rejects_invalid_values(
    config: HandConfig, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config.validate()


def test_legacy_calibration_yaml_still_loads_as_xm335(tmp_path) -> None:
    path = tmp_path / "legacy.yaml"
    _write_yaml(path, {"home_offsets": [0.1] * 13})

    config = HandConfig.load(path)

    assert config.motor_model == XM335_T323.name
    assert config.home_offsets == (0.1,) * 13
    assert config.position_p_gain == 800
    assert config.goal_current_limit == 600


def test_xl330_yaml_loads_runtime_mechanical_and_homing_fields(tmp_path) -> None:
    lookup = tmp_path / "lookup.csv"
    lookup.write_text(
        "q_pip,q_dip,q_linkage_base,q_linkage_distal\n"
        "0,0,0,0\n1,2,0,0\n2,4,0,0\n",
        encoding="utf-8",
    )
    xm = HandConfig()
    path = tmp_path / "xl330.yaml"
    _write_yaml(
        path,
        {
            "motor_model": "XL330_M077",
            "motor_ids": list(range(13)),
            "port": "/dev/example",
            "baudrate": 1_000_000,
            "position_p_gain": 410,
            "position_i_gain": 1,
            "position_d_gain": 390,
            "goal_current_limit_ma": 250,
            "home_offsets": {motor_id: motor_id / 100 for motor_id in range(13)},
            "joint_signs": [1, -1] + [1] * 11,
            "joint_lower_limits": [-1.0] * 13,
            "joint_upper_limits": [1.0] * 13,
            "pip_dip_lookup": "lookup.csv",
            "homing": {
                "thumb": [list(entry) for entry in xm.thumb_homing_table],
                "fingers": [list(entry) for entry in xm.finger_homing_table],
                "middle_mcp_flex_pre_abduct_position_rad": -1.2,
            },
        },
    )

    config = HandConfig.load(path)

    assert config.motor_model == XL330_M077.name
    assert config.expected_model_number == 1190
    assert config.port == "/dev/example"
    assert config.baudrate == 1_000_000
    assert (
        config.position_p_gain,
        config.position_i_gain,
        config.position_d_gain,
    ) == (410, 1, 390)
    assert config.goal_current_limit == 250
    assert config.home_offsets[12] == pytest.approx(0.12)
    assert config.joint_signs[:2] == (1.0, -1.0)
    assert config.joint_lower_limits == (-1.0,) * 13
    assert config.joint_upper_limits == (1.0,) * 13
    assert config.pip_dip_lookup == str(lookup.resolve())
    assert config.thumb_homing_table == xm.thumb_homing_table
    assert config.finger_homing_table == xm.finger_homing_table
    assert config.middle_mcp_flex_pre_abduct_position_rad == -1.2


def test_xl330_save_load_round_trip(tmp_path) -> None:
    path = tmp_path / "roundtrip.yaml"
    original = HandConfig.xl330_m077(
        motor_ids=(2, 7),
        baudrate=1_000_000,
        home_offsets=(0.2, 0.7),
        joint_signs=(-1, 1),
        joint_lower_limits=(-0.5, -0.7),
        joint_upper_limits=(0.8, 0.9),
    )

    original.save(path)
    loaded = HandConfig.load(path)

    assert loaded == original


def test_offset_merge_preserves_unrelated_yaml_and_subset_preserves_model(
    tmp_path,
) -> None:
    path = tmp_path / "merge.yaml"
    _write_yaml(
        path,
        {
            "motor_model": "xl330-m077-t",
            "motor_ids": [0, 1],
            "position_p_gain": 420,
            "goal_current_limit_ma": 275,
            "home_offsets": {0: 0.0, 1: 0.1},
            "custom_metadata": {"keep": True},
        },
    )
    config = HandConfig.load(path)
    changed = replace(config, home_offsets=(0.25, 0.1))

    changed.save_merged_offsets([0], path)
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    subset = changed.subset((1,))

    assert saved["custom_metadata"] == {"keep": True}
    assert saved["home_offsets"][0] == pytest.approx(0.25)
    assert saved["home_offsets"][1] == pytest.approx(0.1)
    assert subset.motor_model == XL330_M077.name
    assert subset.motor_ids == (1,)
    assert subset.position_p_gain == 420
    assert subset.goal_current_limit == 275


@pytest.mark.parametrize(
    ("factory", "expected_model", "other_model"),
    [
        (HandConfig.xm335_t323, 1710, 1190),
        (HandConfig.xl330_m077, 1190, 1710),
    ],
    ids=["xm335", "xl330"],
)
def test_hand_model_verification_uses_selected_config(
    factory, expected_model: int, other_model: int
) -> None:
    class PingClient:
        def ping(self, motor_ids, **kwargs):
            return {motor_ids[0]: expected_model, motor_ids[1]: other_model}

    hand = MidasHand(factory(motor_ids=(3, 4)), autoconnect=False)
    hand.dxl_client = PingClient()

    assert hand.verify_models() == {4: other_model}
