"""Generate one side-by-side movement graph image from telemetry CSV files.

Example:

    python scripts/gen_graphs_from_csvs.py output/profile.csv output/stream.csv \
        --titles "Profile move" "Stream move" \
        --output output/comparison.png
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Sequence

import numpy as np


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
CURRENT_COLUMN = re.compile(r"^motor_(\d+)_current_ma$")


def _load_telemetry(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, np.ndarray]]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "elapsed_s" not in reader.fieldnames:
            raise ValueError(f"{path}: missing elapsed_s column")

        motor_ids = []
        for field in reader.fieldnames:
            match = CURRENT_COLUMN.fullmatch(field)
            if match:
                motor_ids.append(int(match.group(1)))
        if not motor_ids:
            raise ValueError(f"{path}: no motor current columns found")

        required_fields = []
        for motor_id in motor_ids:
            required_fields.extend(
                (
                    f"motor_{motor_id}_current_ma",
                    f"motor_{motor_id}_target_position_rad",
                    f"motor_{motor_id}_actual_position_rad",
                )
            )
        missing = [field for field in required_fields if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(missing)}")

        rows = list(reader)
        if not rows:
            raise ValueError(f"{path}: contains no telemetry samples")

    try:
        trials = np.asarray(
            [int(row["trial"]) for row in rows]
            if "trial" in reader.fieldnames
            else [1] * len(rows)
        )
        times = np.asarray([float(row["elapsed_s"]) for row in rows])
        motors = {
            motor_id: {
                "current": np.asarray(
                    [float(row[f"motor_{motor_id}_current_ma"]) for row in rows]
                ),
                "target": np.asarray(
                    [
                        float(row[f"motor_{motor_id}_target_position_rad"])
                        for row in rows
                    ]
                ),
                "actual": np.asarray(
                    [
                        float(row[f"motor_{motor_id}_actual_position_rad"])
                        for row in rows
                    ]
                ),
            }
            for motor_id in motor_ids
        }
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: contains a non-numeric telemetry value") from exc
    if np.any(trials <= 0):
        raise ValueError(f"{path}: trial numbers must be positive integers")
    return trials, times, motors


def _sequential_trials(
    trials: np.ndarray,
    times: np.ndarray,
    motors: dict[int, dict[str, np.ndarray]],
) -> tuple[np.ndarray, dict[int, dict[str, np.ndarray]]]:
    combined_times = []
    combined = {
        motor_id: {metric: [] for metric in values}
        for motor_id, values in motors.items()
    }
    offset = 0.0
    for trial in sorted(np.unique(trials)):
        mask = trials == trial
        trial_times = times[mask] - times[mask][0]
        combined_times.append(trial_times + offset)
        for motor_id, values in motors.items():
            for metric, samples in values.items():
                combined[motor_id][metric].append(samples[mask])
        offset += float(trial_times[-1])
    return np.concatenate(combined_times), {
        motor_id: {
            metric: np.concatenate(trial_values)
            for metric, trial_values in values.items()
        }
        for motor_id, values in combined.items()
    }


def _average_trials(
    trials: np.ndarray,
    times: np.ndarray,
    motors: dict[int, dict[str, np.ndarray]],
) -> tuple[
    np.ndarray,
    dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
]:
    trial_ids = sorted(np.unique(trials))
    trial_times = [
        times[trials == trial] - times[trials == trial][0] for trial in trial_ids
    ]
    duration_s = float(np.median([values[-1] for values in trial_times]))
    sample_count = max(len(values) for values in trial_times)
    common_times = np.linspace(0.0, duration_s, sample_count)

    averaged = {}
    for motor_id, values in motors.items():
        averaged[motor_id] = {}
        for metric, samples in values.items():
            interpolated = np.vstack(
                [
                    np.interp(
                        common_times,
                        trial_time,
                        samples[trials == trial],
                    )
                    for trial, trial_time in zip(trial_ids, trial_times)
                ]
            )
            averaged[motor_id][metric] = (
                np.mean(interpolated, axis=0),
                np.std(interpolated, axis=0),
            )
    return common_times, averaged


def _generate_graph(
    csv_paths: Sequence[Path],
    output: Path,
    titles: Sequence[str],
    combine_mode: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Graph generation requires the graph extra; install with "
            "`pip install -e '.[graph]'`"
        ) from exc

    telemetry = [_load_telemetry(path) for path in csv_paths]
    figure, axes = plt.subplots(
        2,
        len(csv_paths),
        squeeze=False,
        figsize=(8 * len(csv_paths), 7),
        sharex="col",
    )

    for column, (trials, times, motors) in enumerate(telemetry):
        current_axis = axes[0, column]
        position_axis = axes[1, column]
        if combine_mode == "sequential":
            plot_times, plot_motors = _sequential_trials(trials, times, motors)
            for motor_id, values in plot_motors.items():
                label = MOTOR_LABELS.get(motor_id, f"Motor {motor_id}")
                line = current_axis.plot(
                    plot_times,
                    values["current"],
                    label=f"{label} (ID {motor_id})",
                )[0]
                position_axis.plot(
                    plot_times,
                    values["target"],
                    color=line.get_color(),
                    linestyle="--",
                    alpha=0.45,
                    label=f"{label} (ID {motor_id}) target",
                )
                position_axis.plot(
                    plot_times,
                    values["actual"],
                    color=line.get_color(),
                    linewidth=1.8,
                    label=f"{label} (ID {motor_id}) actual",
                )
        else:
            plot_times, plot_motors = _average_trials(trials, times, motors)
            for motor_id, values in plot_motors.items():
                label = MOTOR_LABELS.get(motor_id, f"Motor {motor_id}")
                current_mean, current_std = values["current"]
                line = current_axis.plot(
                    plot_times,
                    current_mean,
                    label=f"{label} (ID {motor_id}) mean",
                )[0]
                current_axis.fill_between(
                    plot_times,
                    current_mean - current_std,
                    current_mean + current_std,
                    color=line.get_color(),
                    alpha=0.2,
                    label=f"{label} ±1 SD",
                )
                target_mean, _ = values["target"]
                actual_mean, actual_std = values["actual"]
                position_axis.plot(
                    plot_times,
                    target_mean,
                    color=line.get_color(),
                    linestyle="--",
                    alpha=0.45,
                    label=f"{label} (ID {motor_id}) target mean",
                )
                position_axis.plot(
                    plot_times,
                    actual_mean,
                    color=line.get_color(),
                    linewidth=1.8,
                    label=f"{label} (ID {motor_id}) actual mean",
                )
                position_axis.fill_between(
                    plot_times,
                    actual_mean - actual_std,
                    actual_mean + actual_std,
                    color=line.get_color(),
                    alpha=0.2,
                    label=f"{label} actual ±1 SD",
                )

        current_axis.set_title(
            f"{titles[column]}\nMotor current during movement",
            fontweight="semibold",
        )
        current_axis.set_ylabel("Current (mA)")
        current_axis.grid(True, alpha=0.3)
        current_axis.legend()
        position_axis.set_title("Commanded and actual position")
        position_axis.set_xlabel("Time since movement began (s)")
        position_axis.set_ylabel("Position (rad)")
        position_axis.grid(True, alpha=0.3)
        position_axis.legend()

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)
    print(f"Saved graph to {output}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csvs",
        nargs="+",
        type=Path,
        help="Movement telemetry CSVs, ordered left to right",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/movement_graphs.png"),
        help="Output PNG path (default: output/movement_graphs.png)",
    )
    parser.add_argument(
        "--titles",
        nargs="+",
        metavar="TITLE",
        help=(
            "One title per CSV, ordered left to right; defaults to each CSV "
            "filename without its extension"
        ),
    )
    parser.add_argument(
        "--combine-mode",
        choices=("sequential", "average"),
        default="sequential",
        help=(
            "Plot trials end-to-end or plot their interpolated mean ± standard "
            "deviation (default: sequential)"
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    missing = [path for path in args.csvs if not path.is_file()]
    if missing:
        parser.error(f"CSV file does not exist: {missing[0]}")
    titles = args.titles or [path.stem for path in args.csvs]
    if len(titles) != len(args.csvs):
        parser.error(
            f"--titles requires exactly one title per CSV "
            f"({len(args.csvs)} expected, {len(titles)} provided)"
        )
    try:
        _generate_graph(args.csvs, args.output, titles, args.combine_mode)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
