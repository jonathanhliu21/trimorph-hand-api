import pytest

from midas_hand_api import homing
from midas_hand_api.actuators import control_table as ct
from midas_hand_api.config import HandConfig
from midas_hand_api.hand import MidasHand


class RecordingClient:
    def __init__(self, config: HandConfig) -> None:
        self.cur_scale = config.current_unit_ma
        self.vel_scale = config.velocity_scale
        self.writes = []
        self.torque_commands = []

    def sync_write(self, motor_ids, values, address, size) -> None:
        self.writes.append((list(motor_ids), list(values), address, size))

    def set_torque_enabled(self, motor_ids, enabled, **kwargs) -> None:
        self.torque_commands.append((list(motor_ids), enabled, kwargs))


@pytest.mark.parametrize(
    "config",
    [
        HandConfig(),
        HandConfig.xl330_m077(
            thumb_homing_table=HandConfig().thumb_homing_table,
            finger_homing_table=HandConfig().finger_homing_table,
        ),
    ],
    ids=["xm335", "xl330"],
)
def test_finger_homing_keeps_existing_sequence(
    config: HandConfig, monkeypatch
) -> None:
    observed_steps = []

    def record_entries(hand, table, label, offsets, **kwargs):
        observed_steps.append(label)
        return offsets, []

    monkeypatch.setattr(homing, "_home_entries", record_entries)
    monkeypatch.setattr(
        homing,
        "_finish_homing",
        lambda hand, *args, **kwargs: hand.config,
    )
    monkeypatch.setattr(
        homing,
        "_set_middle_mcp_flex_for_abduction",
        lambda *args: None,
    )

    homing.home_fingers(MidasHand(config, autoconnect=False), save=False)

    assert observed_steps == [
        "index finger",
        "middle finger",
        "middle finger abduction",
        "ring finger",
    ]


def test_xl330_without_homing_table_is_a_safe_noop() -> None:
    hand = MidasHand(HandConfig.xl330_m077(), autoconnect=False)

    result = homing.home_fingers(hand, save=False)

    assert result == hand.config


def test_restore_position_mode_uses_xl330_configured_tuning() -> None:
    config = HandConfig.xl330_m077(
        motor_ids=(0,),
        position_p_gain=401,
        position_i_gain=2,
        position_d_gain=403,
        goal_current_limit=301,
    )
    hand = MidasHand(config, autoconnect=False)
    client = RecordingClient(config)
    hand.dxl_client = client

    homing._restore_position_mode(hand, 0)

    writes_by_address = {write[2]: write[1] for write in client.writes}
    assert writes_by_address[ct.ADDR_OPERATING_MODE] == [config.operating_mode]
    assert writes_by_address[ct.ADDR_POSITION_P_GAIN] == [401]
    assert writes_by_address[ct.ADDR_POSITION_I_GAIN] == [2]
    assert writes_by_address[ct.ADDR_POSITION_D_GAIN] == [403]
    assert writes_by_address[ct.ADDR_GOAL_CURRENT] == [301]


def test_homing_configuration_validates_table_size_and_direction() -> None:
    with pytest.raises(ValueError, match="4 entries or be empty"):
        HandConfig.xl330_m077(
            thumb_homing_table=((0, "thumb", 0.0, 1),)
        ).validate()

    invalid_direction = list(HandConfig().thumb_homing_table)
    invalid_direction[0] = (0, "Thumb IP", 0.0, 0)
    with pytest.raises(ValueError, match="direction"):
        HandConfig.xl330_m077(
            thumb_homing_table=tuple(invalid_direction)
        ).validate()
