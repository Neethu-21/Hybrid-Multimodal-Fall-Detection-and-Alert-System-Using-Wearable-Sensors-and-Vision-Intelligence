from pathlib import Path
import re
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SISFALL_DIR = (
    PROJECT_ROOT
    / "sisfall"
    / "SisFall_dataset"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "outdoor"
    / "phone_compatible_sequences"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 200
STRIDE = 100

NUM_FEATURES = 9


# ============================================================
# LABEL
# ============================================================

def get_label(file_path):

    name = file_path.name.upper()

    if re.match(r"^F\d{2}_", name):
        return 1

    if re.match(r"^D\d{2}_", name):
        return 0

    return None


# ============================================================
# LOAD SISFALL
# ============================================================

def load_sensor_file(file_path):

    rows = []

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            line = line.rstrip(";")

            parts = line.split(",")

            values = []

            for part in parts:

                part = part.strip()

                if not part:
                    continue

                try:
                    values.append(
                        float(part)
                    )

                except ValueError:
                    continue

            if len(values) >= 9:

                rows.append(
                    values[:9]
                )

    if not rows:

        raise ValueError(
            f"No valid sensor data:\n{file_path}"
        )

    return np.asarray(
        rows,
        dtype=np.float32
    )


# ============================================================
# SISFALL → PHONE REPRESENTATION
# ============================================================

def convert_to_phone_features(signal):

    # --------------------------------------------------------
    # SisFall:
    #
    # 0-2 = Accelerometer 1
    # 3-5 = Gyroscope
    # 6-8 = Accelerometer 2
    #
    # Phone:
    #
    # 0-2 = Accelerometer
    # 3-5 = Gravity
    # 6-8 = Gyroscope
    # --------------------------------------------------------

    acceleration = signal[:, 0:3]

    gyroscope = signal[:, 3:6]

    # --------------------------------------------------------
    # Estimate gravity from the low-frequency component
    #
    # A moving-average approximation is used to create
    # a gravity-like component from the accelerometer.
    # --------------------------------------------------------

    gravity = np.zeros_like(
        acceleration
    )

    window_size = 25

    for axis in range(3):

        padded = np.pad(
            acceleration[:, axis],
            (
                window_size // 2,
                window_size // 2
            ),
            mode="edge"
        )

        kernel = np.ones(
            window_size,
            dtype=np.float32
        ) / window_size

        gravity[:, axis] = np.convolve(
            padded,
            kernel,
            mode="valid"
        )[:len(acceleration)]

    phone_features = np.column_stack(
        [
            acceleration[:, 0],
            acceleration[:, 1],
            acceleration[:, 2],

            gravity[:, 0],
            gravity[:, 1],
            gravity[:, 2],

            gyroscope[:, 0],
            gyroscope[:, 1],
            gyroscope[:, 2]
        ]
    )

    return phone_features.astype(
        np.float32
    )


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows(
    signal,
    sequence_length,
    stride
):

    windows = []

    total = len(signal)

    if total < sequence_length:

        return windows

    for start in range(
        0,
        total - sequence_length + 1,
        stride
    ):

        end = (
            start
            + sequence_length
        )

        windows.append(
            signal[start:end]
        )

    return windows


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(file_path):

    label = get_label(
        file_path
    )

    if label is None:

        return 0, None

    signal = load_sensor_file(
        file_path
    )

    phone_signal = (
        convert_to_phone_features(
            signal
        )
    )

    windows = create_windows(
        phone_signal,
        SEQUENCE_LENGTH,
        STRIDE
    )

    if not windows:

        return 0, None

    X = np.asarray(
        windows,
        dtype=np.float32
    )

    y = np.full(
        len(X),
        label,
        dtype=np.int64
    )

    output_file = (
        OUTPUT_DIR
        / f"{file_path.stem}.npz"
    )

    np.savez_compressed(
        output_file,
        X=X,
        y=y
    )

    label_name = (
        "FALL"
        if label == 1
        else "ADL"
    )

    print(
        f"{file_path.name:30s} "
        f"{label_name:4s} "
        f"windows={len(X)}"
    )

    return len(X), label


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print(
        "SISFALL → PHONE-COMPATIBLE "
        "SEQUENCE PREPARATION"
    )
    print("=" * 70)

    print()
    print(
        f"Dataset:\n{SISFALL_DIR}"
    )

    files = sorted(
        SISFALL_DIR.rglob("*.txt")
    )

    print()
    print(
        f"Sensor files found: {len(files)}"
    )

    if not files:

        raise FileNotFoundError(
            "No SisFall .txt files found."
        )

    # --------------------------------------------------------
    # Remove old generated files
    # --------------------------------------------------------

    old_files = list(
        OUTPUT_DIR.glob("*.npz")
    )

    print(
        f"Removing old generated "
        f"files: {len(old_files)}"
    )

    for file in old_files:

        file.unlink()

    # --------------------------------------------------------
    # Process all files
    # --------------------------------------------------------

    total_windows = 0
    total_adl = 0
    total_fall = 0

    processed_files = 0

    for file_path in files:

        try:

            windows, label = process_file(
                file_path
            )

            if label is not None:

                processed_files += 1

            total_windows += windows

            if label == 0:

                total_adl += windows

            elif label == 1:

                total_fall += windows

        except Exception as error:

            print()
            print(
                f"ERROR: "
                f"{file_path.name}"
            )

            print(error)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print(
        "FINAL PHONE-COMPATIBLE "
        "SISFALL DATASET"
    )
    print("=" * 70)

    print(
        f"Files processed : "
        f"{processed_files}"
    )

    print(
        f"Total windows   : "
        f"{total_windows}"
    )

    print(
        f"ADL windows     : "
        f"{total_adl}"
    )

    print(
        f"Fall windows    : "
        f"{total_fall}"
    )

    print(
        f"Features        : "
        f"{NUM_FEATURES}"
    )

    print(
        f"Sequence size   : "
        f"{SEQUENCE_LENGTH}"
    )

    print()
    print(
        f"Saved to:\n"
        f"{OUTPUT_DIR}"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()