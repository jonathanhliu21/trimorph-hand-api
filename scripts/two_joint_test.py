"""Hardware bring-up CLI for a two-joint XL330 test fixture.

The commands mirror ``xl330_hand_test.py``, but every bus operation is limited
to exactly two selected motor IDs.  By default those are the index-finger PIP
and MCP-flex motors (IDs 4 and 5 in the full-hand configuration).

Run from the repository root, for example:

    .venv/bin/python scripts/two_joint_test.py record
    .venv/bin/python scripts/two_joint_test.py profile_move_sequential
    .venv/bin/python scripts/two_joint_test.py profile_move --motors 4
    .venv/bin/python scripts/two_joint_test.py stream_move --motors 4,5
    .venv/bin/python scripts/two_joint_test.py home

Use ``--motor-ids`` before the command if the fixture uses different IDs:

    .venv/bin/python scripts/two_joint_test.py --motor-ids 7,8 record

Review the YAML and the motion constants below before running ``home`` or a
motion command. Every command attempts to disable torque before closing the
serial port.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from midas_hand_api import HandConfig, MidasHand, home_hand  # noqa: E402


DEFAULT_CONFIG_PATH = Path(__file__).with_name("xl330_test_config.yaml")
DEFAULT_MOTOR_IDS = (4, 5)

# Review these calibrated targets before moving hardware. Positive joint angle
# is provisionally treated as closing; joint_signs in the YAML must make that
# convention true for both selected motors.
OPEN_ANGLE_RAD = 0.0
CLOSE_ANGLE_RAD = -1.0

MOTOR_LABELS = {
    0: "Thumb IP",
    1: "Thumb MCP",
    2: "Thumb CMC side",
    3: "Thumb CMC roll",
    4: "Index PIP",
    5: "Index MCP flex",
    6: "Index MCP abduct",
    7: "Middle PIP",
    8: "Middle MCP flex",
    9: "Middle MCP abduct",
    10: "Ring PIP",
    11: "Ring MCP flex",
    12: "Ring MCP abduct",
}

PROFILE_MOVE_VELOCITY_RAD_S = 3.0
PROFILE_MOVE_ACCELERATION_RAD_S2 = 50.0
POSITION_TOLERANCE_RAD = 0.1
PROFILE_MOVE_TIMEOUT_S = 4.0
STREAM_MOVE_RATE_HZ = 30.0
STREAM_MOVE_DURATION_S = 2.0
STREAM_MOVE_SETTLE_S = 0.5


class _CurrentRecorder:
    """Collect selected motor currents only when explicitly sampled."""

    def __init__(self, hand: MidasHand, motor_ids: Sequence[int]) -> None:
        self._hand = hand
        self.motor_ids = tuple(motor_ids)
        self._indices = tuple(hand.motor_ids.index(motor_id) for motor_id in motor_ids)
        self._start_time: Optional[float] = None
        self._trial = 1
        self.trials: list[int] = []
        self.times_s: list[float] = []
        self.currents_ma: list[list[float]] = []
        self.target_positions_rad: list[list[float]] = []
        self.actual_positions_rad: list[list[float]] = []

    def start_trial(self, trial: int) -> None:
        self._trial = trial
        self._start_time = None

    def sample(
        self,
        target_positions_rad: Sequence[float],
        elapsed_s: Optional[float] = None,
    ) -> np.ndarray:
        now = time.monotonic()
        if self._start_time is None:
            self._start_time = now
        actual_positions, _, currents = self._hand.read_pos_vel_cur()
        self.trials.append(self._trial)
        self.times_s.append(now - self._start_time if elapsed_s is None else elapsed_s)
        self.currents_ma.append([float(currents[index]) for index in self._indices])
        self.target_positions_rad.append(
            [float(target_positions_rad[index]) for index in self._indices]
        )
        self.actual_positions_rad.append(
            [float(actual_positions[index]) for index in self._indices]
        )
        return actual_positions


def _output_current_csv(recorder: _CurrentRecorder, output: Path) -> None:
    if not recorder.times_s:
        raise RuntimeError("No movement current samples were recorded")

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    header = ["trial", "elapsed_s"]
    for motor_id in recorder.motor_ids:
        header.extend(
            (
                f"motor_{motor_id}_current_ma",
                f"motor_{motor_id}_target_position_rad",
                f"motor_{motor_id}_actual_position_rad",
            )
        )

    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for sample_index, elapsed_s in enumerate(recorder.times_s):
            row = [recorder.trials[sample_index], elapsed_s]
            for motor_index in range(len(recorder.motor_ids)):
                row.extend(
                    (
                        recorder.currents_ma[sample_index][motor_index],
                        recorder.target_positions_rad[sample_index][motor_index],
                        recorder.actual_positions_rad[sample_index][motor_index],
                    )
                )
            writer.writerow(row)
    print(f"Saved movement telemetry CSV to {output}")


def _parse_motor_ids(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "motor IDs must be comma-separated integers"
        ) from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one motor ID is required")
    if len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("motor IDs must be unique")
    return result


def _parse_two_motor_ids(value: str) -> tuple[int, int]:
    result = _parse_motor_ids(value)
    if len(result) != 2:
        raise argparse.ArgumentTypeError("exactly two motor IDs are required")
    return result


def _load_config(path: Path, motor_ids: Sequence[int]) -> HandConfig:
    full_config = HandConfig.load(path)
    if full_config.motor_model != "xl330-m077-t":
        raise ValueError(
            f"This script requires XL330-M077-T, but {path} selects "
            f"{full_config.motor_model}"
        )
    return full_config.subset(motor_ids)


def _require_reviewed_config(path: Path) -> None:
    with path.open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if data.get("placeholders_unreviewed", True):
        raise RuntimeError(
            "Motion refused: review joint limits, joint signs, homing directions, "
            "CAD offsets, and OPEN_ANGLE_RAD/CLOSE_ANGLE_RAD in this script, then "
            "set placeholders_unreviewed: false in the YAML."
        )


def _require_homing_entries(config: HandConfig) -> None:
    homing_ids = {
        entry[0]
        for entry in config.thumb_homing_table + config.finger_homing_table
    }
    missing = sorted(set(config.motor_ids) - homing_ids)
    if missing:
        raise ValueError(
            "No homing entry is configured for selected motor IDs "
            f"{missing}; refusing to move toward an unknown hard stop"
        )


def _verify_bus(hand: MidasHand) -> None:
    models = hand.ping()
    missing = sorted(set(hand.motor_ids) - set(models))
    wrong_models = {
        motor_id: model_number
        for motor_id, model_number in models.items()
        if model_number != hand.config.expected_model_number
    }
    if missing:
        raise OSError(f"Configured motors did not respond: {missing}")
    if wrong_models:
        raise OSError(f"Unexpected Dynamixel model numbers: {wrong_models}")


def _confirm_motion(action: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    response = input(
        f"{action} will move physical hardware. Type MOVE to continue: "
    ).strip()
    if response != "MOVE":
        raise RuntimeError("Motion cancelled")


def _run_home(args: argparse.Namespace, config: HandConfig) -> None:
    _require_reviewed_config(args.config)
    _require_homing_entries(config)
    _confirm_motion("Homing", args.yes)
    hand = MidasHand(config, autoconnect=False)
    try:
        hand.connect()
        _verify_bus(hand)
        hand.configure(enable_torque=False)
        calibrated = home_hand(
            hand,
            save=False,
            current_threshold_ma=args.current_threshold_ma,
            homing_velocity_rad_s=args.velocity_rad_s,
        )
        calibrated.save_merged_offsets(
            range(calibrated.n_motors),
            args.config,
        )
        print(
            "Homing complete; measured offsets for motor IDs "
            f"{list(calibrated.motor_ids)} merged into {args.config}"
        )
    finally:
        if hand.is_connected:
            hand.shutdown()


def _wait_for_motor(
    hand: MidasHand,
    motor_id: int,
    target_rad: float,
    timeout_s: float,
    sample_motion: Optional[Callable[[], Sequence[float]]] = None,
    skip_target_safety: bool = False,
) -> bool:
    motor_index = hand.motor_ids.index(motor_id)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        positions = hand.read_pos() if sample_motion is None else sample_motion()
        measured = float(positions[motor_index])
        if abs(measured - target_rad) <= POSITION_TOLERANCE_RAD:
            return True
        time.sleep(0.02)
    measured = float(hand.read_pos()[motor_index])
    message = (
        f"Motor {motor_id} did not reach {target_rad:.3f} rad; "
        f"last reading was {measured:.3f} rad"
    )
    if skip_target_safety:
        print(f"WARNING: {message}; continuing because --skip-target-safety is set")
        return False
    raise TimeoutError(message)


def _check_move_target(config: HandConfig, motor_id: int, target: float) -> None:
    index = config.motor_ids.index(motor_id)
    lower = float(config.joint_lower_limits[index])
    upper = float(config.joint_upper_limits[index])
    if not lower <= target <= upper:
        raise ValueError(
            f"Motor {motor_id} target {target:.3f} is outside configured "
            f"limits [{lower:.3f}, {upper:.3f}]"
        )


def _selected_motor_ids(
    args: argparse.Namespace,
    config: HandConfig,
) -> tuple[int, ...]:
    selected = config.motor_ids if args.motors is None else args.motors
    unknown_ids = sorted(set(selected) - set(config.motor_ids))
    if unknown_ids:
        raise ValueError(
            f"Motion motor IDs {unknown_ids} are not in the two-joint fixture "
            f"{list(config.motor_ids)}"
        )
    return tuple(selected)


def _run_profile_move(args: argparse.Namespace, config: HandConfig) -> None:
    _require_reviewed_config(args.config)
    # if not any(abs(offset) > 1e-9 for offset in config.home_offsets):
    #     raise RuntimeError(
    #         "All selected home_offsets are zero; run and review homing first"
    #     )

    test_motor_ids = _selected_motor_ids(args, config)
    for motor_id in test_motor_ids:
        _check_move_target(config, motor_id, CLOSE_ANGLE_RAD)
        _check_move_target(config, motor_id, OPEN_ANGLE_RAD)
    print(f"Selected motor IDs = {list(config.motor_ids)}")
    print(f"OPEN_ANGLE_RAD     = {OPEN_ANGLE_RAD}")
    print(f"CLOSE_ANGLE_RAD    = {CLOSE_ANGLE_RAD}")
    if args.skip_target_safety:
        print("WARNING: target timeout failures will not stop this movement sequence")
    _confirm_motion("Profile move testing", args.yes)

    hand = MidasHand(config, autoconnect=False)
    current_recorder: Optional[_CurrentRecorder] = None
    try:
        hand.connect()
        _verify_bus(hand)
        hand.configure(enable_torque=False)
        if args.output_current is not None:
            current_recorder = _CurrentRecorder(hand, test_motor_ids)

        for trial in range(1, args.trials + 1):
            print(f"Trial {trial}/{args.trials}")
            if current_recorder is not None:
                current_recorder.start_trial(trial)
            for motor_id in test_motor_ids:
                motor_index = hand.motor_ids.index(motor_id)
                label = MOTOR_LABELS.get(motor_id, "unlabelled joint")
                print(
                    f"Motor {motor_id} ({label}): closing to "
                    f"{CLOSE_ANGLE_RAD:.3f} rad, then opening to "
                    f"{OPEN_ANGLE_RAD:.3f} rad"
                )
                try:
                    # Establish a non-jumping goal before enabling this motor alone.
                    hold_position = hand.read_pos()
                    hand.set_positions(hold_position, clip=True)
                    hand.set_motion_profile(
                        profile_velocity_rad_s=args.profile_velocity_rad_s,
                        profile_acceleration_rad_s2=args.profile_acceleration_rad_s2,
                        motor_ids=[motor_id],
                    )
                    hand.enable_torque([motor_id])

                    target = np.array(hold_position, copy=True)
                    target[motor_index] = CLOSE_ANGLE_RAD
                    hand.set_positions(target)
                    _wait_for_motor(
                        hand,
                        motor_id,
                        CLOSE_ANGLE_RAD,
                        args.timeout_s,
                        None
                        if current_recorder is None
                        else lambda: current_recorder.sample(target),
                        args.skip_target_safety,
                    )
                    time.sleep(args.pause_s)

                    target[motor_index] = OPEN_ANGLE_RAD
                    hand.set_positions(target)
                    _wait_for_motor(
                        hand,
                        motor_id,
                        OPEN_ANGLE_RAD,
                        args.timeout_s,
                        None
                        if current_recorder is None
                        else lambda: current_recorder.sample(target),
                        args.skip_target_safety,
                    )
                    time.sleep(args.pause_s)
                finally:
                    hand.disable_torque([motor_id])
    finally:
        if hand.is_connected:
            hand.shutdown()
    if current_recorder is not None:
        _output_current_csv(current_recorder, args.output_current)


def _run_stream_move(args: argparse.Namespace, config: HandConfig) -> None:
    """Stream equally spaced targets from open to closed and back to open."""

    _require_reviewed_config(args.config)
    test_motor_ids = _selected_motor_ids(args, config)
    for motor_id in test_motor_ids:
        _check_move_target(config, motor_id, OPEN_ANGLE_RAD)
        _check_move_target(config, motor_id, CLOSE_ANGLE_RAD)
    sample_count = max(2, int(round(args.duration_s * args.rate_hz)) + 1)
    leg_times = np.linspace(0.0, args.duration_s, sample_count)
    close_samples = np.linspace(OPEN_ANGLE_RAD, CLOSE_ANGLE_RAD, sample_count)
    open_samples = np.linspace(CLOSE_ANGLE_RAD, OPEN_ANGLE_RAD, sample_count)

    # Do not send the closed endpoint twice at the turnaround. Keeping one copy
    # also preserves the requested setpoint interval across both legs.
    sample_times = np.concatenate((leg_times, args.duration_s + leg_times[1:]))
    samples = np.concatenate((close_samples, open_samples[1:]))

    print(f"Selected motor IDs = {list(test_motor_ids)}")
    print(f"OPEN_ANGLE_RAD     = {OPEN_ANGLE_RAD}")
    print(f"CLOSE_ANGLE_RAD    = {CLOSE_ANGLE_RAD}")
    if args.skip_target_safety:
        print("WARNING: target timeout failures will not stop this movement sequence")
    print(
        f"Stream             = open -> close -> open, {sample_count} points per "
        f"leg over {args.duration_s:.3f}s per leg at approximately "
        f"{args.rate_hz:.1f}Hz"
    )
    _confirm_motion("Stream move testing", args.yes)

    hand = MidasHand(config, autoconnect=False)
    current_recorder: Optional[_CurrentRecorder] = None
    try:
        hand.connect()
        _verify_bus(hand)
        hand.configure(enable_torque=False)
        if args.output_current is not None:
            current_recorder = _CurrentRecorder(hand, test_motor_ids)

        selected_indices = [
            hand.motor_ids.index(motor_id) for motor_id in test_motor_ids
        ]
        hold_position = hand.read_pos()
        hand.set_positions(hold_position, clip=False)
        hand.enable_torque(test_motor_ids)

        open_target = np.array(hold_position, copy=True)
        open_target[selected_indices] = OPEN_ANGLE_RAD
        for trial in range(1, args.trials + 1):
            print(f"Trial {trial}/{args.trials}")
            hand.set_motion_profile(
                profile_velocity_rad_s=PROFILE_MOVE_VELOCITY_RAD_S,
                profile_acceleration_rad_s2=PROFILE_MOVE_ACCELERATION_RAD_S2,
                motor_ids=test_motor_ids,
            )
            hand.set_positions(open_target)
            for motor_id in test_motor_ids:
                _wait_for_motor(
                    hand,
                    motor_id,
                    OPEN_ANGLE_RAD,
                    args.preposition_timeout_s,
                    skip_target_safety=args.skip_target_safety,
                )
            if args.settle_s > 0:
                time.sleep(args.settle_s)

            # set motion profile before streaming
            hand.set_motion_profile(
                profile_velocity_rad_s=PROFILE_MOVE_VELOCITY_RAD_S,
                profile_acceleration_rad_s2=PROFILE_MOVE_ACCELERATION_RAD_S2,
                motor_ids=test_motor_ids,
            )

            if current_recorder is not None:
                current_recorder.start_trial(trial)
            target = np.array(open_target, copy=True)
            start_t = time.monotonic()
            for tick, (sample_t, angle_rad) in enumerate(zip(sample_times, samples)):
                target[selected_indices] = angle_rad
                hand.set_positions(target)
                if current_recorder is not None:
                    current_recorder.sample(target, float(sample_t))

                if tick + 1 < len(samples):
                    next_tick_t = start_t + float(sample_times[tick + 1])
                    sleep_s = next_tick_t - time.monotonic()
                    if sleep_s > 0:
                        time.sleep(sleep_s)

            print(f"Trial {trial} stream move complete.")
    finally:
        if hand.is_connected:
            hand.shutdown()
    if current_recorder is not None:
        _output_current_csv(current_recorder, args.output_current)


def _run_record(args: argparse.Namespace, config: HandConfig) -> None:
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not any(abs(offset) > 1e-9 for offset in config.home_offsets):
        print(
            "WARNING: all selected home_offsets are zero; recorded values will not "
            "be meaningful calibrated joint angles until homing is complete."
        )

    hand = MidasHand(config, autoconnect=False)
    sample_period = 1.0 / args.rate_hz
    sample_count = 0
    start_monotonic = time.monotonic()

    try:
        hand.connect()
        _verify_bus(hand)
        hand.disable_torque()
        print("Torque is disabled. Move the joints manually; press Ctrl-C to stop.")
        print(f"Recording calibrated actuator angles to {output}")

        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["elapsed_s", "unix_time_s"]
                + [f"motor_{motor_id}_rad" for motor_id in hand.motor_ids]
            )
            next_sample = time.monotonic()
            while (
                args.duration_s is None
                or time.monotonic() - start_monotonic < args.duration_s
            ):
                now = time.monotonic()
                if now < next_sample:
                    time.sleep(next_sample - now)
                elapsed = time.monotonic() - start_monotonic
                positions = hand.read_pos()
                writer.writerow([elapsed, time.time(), *positions.tolist()])
                stream.flush()
                sample_count += 1
                if args.print_every and sample_count % args.print_every == 0:
                    formatted = np.array2string(
                        positions,
                        precision=3,
                        suppress_small=True,
                    )
                    print(f"{elapsed:8.3f}s {formatted}")
                next_sample += sample_period
    except KeyboardInterrupt:
        print("\nRecording stopped by user.")
    finally:
        if hand.is_connected:
            hand.shutdown()
    print(f"Saved {sample_count} samples to {output}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"XL330 YAML configuration (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--motor-ids",
        type=_parse_two_motor_ids,
        default=DEFAULT_MOTOR_IDS,
        help="Two comma-separated fixture motor IDs (default: 4,5)",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    home = commands.add_parser("home", help="Home both configured joints")
    home.add_argument("--current-threshold-ma", type=float, default=200.0)
    home.add_argument("--velocity-rad-s", type=float, default=1.5)
    home.add_argument("--yes", action="store_true", help="Skip the MOVE prompt")

    profile_move = commands.add_parser(
        "profile_move",
        help="Move one or both fixture motors closed then open",
    )
    profile_move.add_argument(
        "--motors",
        type=_parse_motor_ids,
        default=None,
        help="Fixture motor IDs to test (default: both)",
    )
    _add_profile_move_arguments(profile_move)

    profile_move_sequential = commands.add_parser(
        "profile_move_sequential",
        help="Profile-move both fixture motors sequentially to verify mapping",
    )
    profile_move_sequential.set_defaults(motors=None)
    _add_profile_move_arguments(profile_move_sequential)

    stream_move = commands.add_parser(
        "stream_move",
        help="Stream equally spaced targets open to closed and back to open",
    )
    stream_move.add_argument(
        "--motors",
        type=_parse_motor_ids,
        default=None,
        help="Fixture motor IDs to stream together (default: both)",
    )
    stream_move.add_argument(
        "--rate-hz",
        type=float,
        default=STREAM_MOVE_RATE_HZ,
        help=f"Setpoint streaming rate (default: {STREAM_MOVE_RATE_HZ:g} Hz)",
    )
    stream_move.add_argument(
        "--duration-s",
        type=float,
        default=STREAM_MOVE_DURATION_S,
        help=f"Duration of each close/open leg (default: {STREAM_MOVE_DURATION_S:g}s)",
    )
    stream_move.add_argument(
        "--preposition-timeout-s",
        type=float,
        default=PROFILE_MOVE_TIMEOUT_S,
        help="Timeout while moving to OPEN_ANGLE_RAD before streaming",
    )
    stream_move.add_argument(
        "--settle-s",
        type=float,
        default=STREAM_MOVE_SETTLE_S,
        help=f"Pause at the open pose before streaming (default: {STREAM_MOVE_SETTLE_S:g}s)",
    )
    stream_move.add_argument(
        "--yes",
        action="store_true",
        help="Skip the MOVE prompt",
    )
    _add_output_current_argument(stream_move)
    _add_skip_target_safety_argument(stream_move)
    _add_trials_argument(stream_move)

    record = commands.add_parser(
        "record",
        help="Disable torque and record both calibrated actuator angles to CSV",
    )
    record.add_argument(
        "--output",
        type=Path,
        default=Path("two_joint_angles.csv"),
    )
    record.add_argument("--rate-hz", type=float, default=20.0)
    record.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Stop after this duration; otherwise record until Ctrl-C",
    )
    record.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print every Nth sample; use 0 to disable",
    )
    return parser


def _add_profile_move_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile-velocity-rad-s",
        type=float,
        default=PROFILE_MOVE_VELOCITY_RAD_S,
    )
    parser.add_argument(
        "--profile-acceleration-rad-s2",
        type=float,
        default=PROFILE_MOVE_ACCELERATION_RAD_S2,
    )
    parser.add_argument("--timeout-s", type=float, default=PROFILE_MOVE_TIMEOUT_S)
    parser.add_argument("--pause-s", type=float, default=0.5)
    parser.add_argument("--yes", action="store_true", help="Skip the MOVE prompt")
    _add_output_current_argument(parser)
    _add_skip_target_safety_argument(parser)
    _add_trials_argument(parser)


def _add_trials_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of movement trials to run sequentially (default: 1)",
    )


def _add_skip_target_safety_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--skip-target-safety",
        action="store_true",
        help=(
            "Continue after a target is not reached by its timeout; communication "
            "and other hardware errors remain fatal"
        ),
    )


def _add_output_current_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-current",
        nargs="?",
        const=Path("output/current.csv"),
        type=Path,
        metavar="PATH",
        help=(
            "Write movement-only current and position telemetry to CSV; "
            "optionally provide an output path (default: output/current.csv)"
        ),
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.config.is_file():
        parser.error(f"configuration file does not exist: {args.config}")
    if args.command in ("profile_move", "profile_move_sequential", "stream_move"):
        if args.trials <= 0:
            parser.error("--trials must be positive")
    if args.command == "record":
        if args.rate_hz <= 0:
            parser.error("--rate-hz must be positive")
        if args.duration_s is not None and args.duration_s <= 0:
            parser.error("--duration-s must be positive")
        if args.print_every < 0:
            parser.error("--print-every must be non-negative")
    elif args.command == "stream_move":
        if args.rate_hz <= 0:
            parser.error("--rate-hz must be positive")
        if args.duration_s <= 0:
            parser.error("--duration-s must be positive")
        if args.preposition_timeout_s <= 0:
            parser.error("--preposition-timeout-s must be positive")
        if args.settle_s < 0:
            parser.error("--settle-s must be non-negative")
    elif args.command in ("profile_move", "profile_move_sequential"):
        if args.timeout_s <= 0:
            parser.error("--timeout-s must be positive")
        if args.pause_s < 0:
            parser.error("--pause-s must be non-negative")

    try:
        config = _load_config(args.config, args.motor_ids)
        if args.command == "home":
            _run_home(args, config)
        elif args.command in ("profile_move", "profile_move_sequential"):
            _run_profile_move(args, config)
        elif args.command == "stream_move":
            _run_stream_move(args, config)
        else:
            _run_record(args, config)
    except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
