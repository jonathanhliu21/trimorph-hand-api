import pytest

from midas_hand_api.actuators import control_table as ct
from midas_hand_api.config import HandConfig
from midas_hand_api.hand import MidasHand


class FakeDynamixelClient:
    def __init__(self, config: HandConfig) -> None:
        self.vel_scale = config.velocity_scale
        self.writes = []
        self.torque_commands = []

    def sync_write(self, motor_ids, values, address, size) -> None:
        self.writes.append((list(motor_ids), list(values), address, size))

    def set_torque_enabled(self, motor_ids, enabled, **kwargs) -> None:
        self.torque_commands.append((list(motor_ids), enabled, kwargs))


def _hand_with_fake_client(
    config: HandConfig,
) -> tuple[MidasHand, FakeDynamixelClient]:
    hand = MidasHand(config, autoconnect=False)
    client = FakeDynamixelClient(config)
    hand.dxl_client = client
    return hand, client


@pytest.fixture(params=[HandConfig(), HandConfig.xl330_m077()], ids=["xm335", "xl330"])
def hand_config(request) -> HandConfig:
    return request.param


def test_motion_profile_converts_acceleration_rad_s2_to_x_series_raw(
    hand_config: HandConfig,
) -> None:
    hand, client = _hand_with_fake_client(hand_config)
    acceleration = ct.X_SERIES_PROFILE_ACCELERATION_UNIT_RAD_S2 * 300

    hand.set_motion_profile(
        profile_velocity_rad_s=1.5,
        profile_acceleration_rad_s2=acceleration,
        motor_ids=[0, 2],
    )

    assert client.writes[0] == (
        [0, 2],
        [300, 300],
        ct.ADDR_PROFILE_ACCELERATION,
        ct.LEN_PROFILE_ACCELERATION,
    )
    expected_velocity = int(1.5 / client.vel_scale)
    assert client.writes[1] == (
        [0, 2],
        [expected_velocity, expected_velocity],
        ct.ADDR_PROFILE_VELOCITY,
        ct.LEN_PROFILE_VELOCITY,
    )


def test_motion_profile_zero_acceleration_and_unlimited_velocity_write_zero(
    hand_config: HandConfig,
) -> None:
    hand, client = _hand_with_fake_client(hand_config)

    hand.set_motion_profile(
        profile_velocity_rad_s=0.0,
        profile_acceleration_rad_s2=None,
        motor_ids=[1],
    )

    assert client.writes == [
        ([1], [0], ct.ADDR_PROFILE_ACCELERATION, ct.LEN_PROFILE_ACCELERATION),
        ([1], [0], ct.ADDR_PROFILE_VELOCITY, ct.LEN_PROFILE_VELOCITY),
    ]


def test_motion_profile_rejects_negative_acceleration(
    hand_config: HandConfig,
) -> None:
    hand, _ = _hand_with_fake_client(hand_config)

    with pytest.raises(ValueError, match="profile_acceleration_rad_s2"):
        hand.set_motion_profile(profile_acceleration_rad_s2=-1.0)


def test_configure_writes_model_tuning_in_shared_register_order(
    hand_config: HandConfig,
) -> None:
    hand, client = _hand_with_fake_client(hand_config)

    hand.configure(enable_torque=False)

    assert client.torque_commands == [(hand.motor_ids, False, {})]
    assert client.writes[0] == (
        hand.motor_ids,
        [ct.DRIVE_MODE_VELOCITY_BASED_PROFILE] * len(hand.motor_ids),
        ct.ADDR_DRIVE_MODE,
        ct.LEN_DRIVE_MODE,
    )
    assert [write[2] for write in client.writes] == [
        ct.ADDR_DRIVE_MODE,
        ct.ADDR_OPERATING_MODE,
        ct.ADDR_POSITION_P_GAIN,
        ct.ADDR_POSITION_I_GAIN,
        ct.ADDR_POSITION_D_GAIN,
        ct.ADDR_GOAL_CURRENT,
    ]
    assert client.writes[-1] == (
        hand.motor_ids,
        [hand_config.goal_current_limit] * len(hand.motor_ids),
        ct.ADDR_GOAL_CURRENT,
        ct.LEN_GOAL_CURRENT,
    )


def test_current_limit_uses_selected_model_ceiling() -> None:
    xm_hand, _ = _hand_with_fake_client(HandConfig(motor_ids=(0,)))
    xl_hand, xl_client = _hand_with_fake_client(
        HandConfig.xl330_m077(motor_ids=(0,))
    )

    with pytest.raises(ValueError, match="910"):
        xm_hand.set_current_limit(1000)

    xl_hand.set_current_limit(1000)
    assert xl_client.writes == [
        ([0], [1000], ct.ADDR_GOAL_CURRENT, ct.LEN_GOAL_CURRENT)
    ]
