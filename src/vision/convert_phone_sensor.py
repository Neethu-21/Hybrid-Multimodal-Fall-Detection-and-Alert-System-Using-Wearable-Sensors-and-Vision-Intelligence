from pathlib import Path
import json
import csv
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outdoor"
    / "phone_sensor"
    / "recording_20260824_153014.jsonl"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "outdoor"
    / "phone_sensor"
    / "recording_20260824_153014_aligned.csv"
)


# ============================================================
# SENSOR EXTRACTION
# ============================================================

def collect_sensor_data():

    accelerometer = []
    gravity = []
    gyroscope = []

    payload_count = 0

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue

            payload = message.get("payload", [])

            if not isinstance(payload, list):
                continue

            payload_count += 1

            for item in payload:

                name = item.get("name")
                values = item.get("values")
                timestamp = item.get("time")

                if (
                    values is None
                    or timestamp is None
                ):
                    continue

                try:

                    x = float(values["x"])
                    y = float(values["y"])
                    z = float(values["z"])
                    t = int(timestamp)

                except (
                    KeyError,
                    TypeError,
                    ValueError
                ):
                    continue

                row = [
                    t,
                    x,
                    y,
                    z
                ]

                if name == "accelerometer":
                    accelerometer.append(row)

                elif name == "gravity":
                    gravity.append(row)

                elif name == "gyroscope":
                    gyroscope.append(row)

    return (
        payload_count,
        np.asarray(accelerometer, dtype=np.float64),
        np.asarray(gravity, dtype=np.float64),
        np.asarray(gyroscope, dtype=np.float64)
    )


# ============================================================
# REMOVE DUPLICATE TIMESTAMPS
# ============================================================

def remove_duplicate_timestamps(data):

    if len(data) == 0:
        return data

    data = data[
        np.argsort(data[:, 0])
    ]

    timestamps = data[:, 0]

    _, indices = np.unique(
        timestamps,
        return_index=True
    )

    return data[
        np.sort(indices)
    ]


# ============================================================
# INTERPOLATION
# ============================================================

def interpolate_sensor(
    reference_times,
    sensor_data
):

    sensor_data = remove_duplicate_timestamps(
        sensor_data
    )

    if len(sensor_data) < 2:
        raise ValueError(
            "Not enough sensor samples for interpolation."
        )

    times = sensor_data[:, 0]

    x = sensor_data[:, 1]
    y = sensor_data[:, 2]
    z = sensor_data[:, 3]

    result_x = np.interp(
        reference_times,
        times,
        x
    )

    result_y = np.interp(
        reference_times,
        times,
        y
    )

    result_z = np.interp(
        reference_times,
        times,
        z
    )

    return np.column_stack(
        [
            result_x,
            result_y,
            result_z
        ]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("PHONE SENSOR TIME-ALIGNED CONVERTER")
    print("=" * 60)

    print()
    print("Input:")
    print(INPUT_FILE)

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"\nRecording not found:\n{INPUT_FILE}"
        )

    (
        payload_count,
        accelerometer,
        gravity,
        gyroscope
    ) = collect_sensor_data()

    print()
    print(
        f"Payloads processed : "
        f"{payload_count}"
    )

    print(
        f"Accelerometer raw  : "
        f"{len(accelerometer)}"
    )

    print(
        f"Gravity raw        : "
        f"{len(gravity)}"
    )

    print(
        f"Gyroscope raw      : "
        f"{len(gyroscope)}"
    )

    if len(accelerometer) < 2:
        raise ValueError(
            "Not enough accelerometer data."
        )

    if len(gravity) < 2:
        raise ValueError(
            "Not enough gravity data."
        )

    if len(gyroscope) < 2:
        raise ValueError(
            "Not enough gyroscope data."
        )

    # --------------------------------------------------------
    # Accelerometer is our reference timeline
    # --------------------------------------------------------

    accelerometer = remove_duplicate_timestamps(
        accelerometer
    )

    reference_times = accelerometer[:, 0]

    acc_values = accelerometer[:, 1:4]

    # --------------------------------------------------------
    # Interpolate other sensors
    # --------------------------------------------------------

    gravity_values = interpolate_sensor(
        reference_times,
        gravity
    )

    gyro_values = interpolate_sensor(
        reference_times,
        gyroscope
    )

    # --------------------------------------------------------
    # Combine 9 features
    # --------------------------------------------------------

    combined = np.column_stack(
        [
            reference_times,

            acc_values[:, 0],
            acc_values[:, 1],
            acc_values[:, 2],

            gravity_values[:, 0],
            gravity_values[:, 1],
            gravity_values[:, 2],

            gyro_values[:, 0],
            gyro_values[:, 1],
            gyro_values[:, 2]
        ]
    )

    # --------------------------------------------------------
    # Keep only timestamps within all sensor ranges
    # --------------------------------------------------------

    start_time = max(
        accelerometer[0, 0],
        gravity[0, 0],
        gyroscope[0, 0]
    )

    end_time = min(
        accelerometer[-1, 0],
        gravity[-1, 0],
        gyroscope[-1, 0]
    )

    valid = (
        (combined[:, 0] >= start_time)
        &
        (combined[:, 0] <= end_time)
    )

    combined = combined[valid]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    header = [
        "timestamp",

        "acc_x",
        "acc_y",
        "acc_z",

        "gravity_x",
        "gravity_y",
        "gravity_z",

        "gyro_x",
        "gyro_y",
        "gyro_z"
    ]

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)

        writer.writerows(
            combined
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("ALIGNED DATASET")
    print("=" * 60)

    print(
        f"Aligned samples : "
        f"{len(combined)}"
    )

    if len(combined) >= 2:

        timestamps = combined[:, 0]

        duration = (
            timestamps[-1]
            - timestamps[0]
        ) / 1_000_000_000

        print(
            f"Duration        : "
            f"{duration:.2f} seconds"
        )

        if duration > 0:

            rate = (
                len(combined)
                / duration
            )

            print(
                f"Effective rate  : "
                f"{rate:.2f} Hz"
            )

    print()
    print("Features:")

    print(
        "1  acc_x"
    )
    print(
        "2  acc_y"
    )
    print(
        "3  acc_z"
    )
    print(
        "4  gravity_x"
    )
    print(
        "5  gravity_y"
    )
    print(
        "6  gravity_z"
    )
    print(
        "7  gyro_x"
    )
    print(
        "8  gyro_y"
    )
    print(
        "9  gyro_z"
    )

    print()
    print("Output:")
    print(OUTPUT_FILE)

    print()
    print("=" * 60)
    print("TIME ALIGNMENT COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()