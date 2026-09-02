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
    / "sequences"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLING_RATE = 200

# 1 second sequence
SEQUENCE_LENGTH = 200

# 50% overlap
STRIDE = 100

# ------------------------------------------------------------
# SisFall sensor channels
#
# We use 9 channels:
#
# 0  Accelerometer 1 - X
# 1  Accelerometer 1 - Y
# 2  Accelerometer 1 - Z
#
# 3  Gyroscope - X
# 4  Gyroscope - Y
# 5  Gyroscope - Z
#
# 6  Accelerometer 2 - X
# 7  Accelerometer 2 - Y
# 8  Accelerometer 2 - Z
# ------------------------------------------------------------

NUM_FEATURES = 9


# ============================================================
# LABEL
# ============================================================

def get_label(file_path):

    name = file_path.name.upper()

    # F01, F02, ...
    if re.match(
        r"^F\d{2}_",
        name
    ):
        return 1

    # D01, D02, ...
    if re.match(
        r"^D\d{2}_",
        name
    ):
        return 0

    return None


# ============================================================
# LOAD SENSOR FILE
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

            # Remove trailing semicolon
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

            # ------------------------------------------------
            # We need at least 9 values.
            # ------------------------------------------------

            if len(values) >= NUM_FEATURES:

                rows.append(
                    values[:NUM_FEATURES]
                )

    if not rows:

        raise ValueError(
            f"No valid sensor data found:\n"
            f"{file_path}"
        )

    signal = np.asarray(
        rows,
        dtype=np.float32
    )

    return signal


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
            start +
            sequence_length
        )

        window = (
            signal[start:end]
        )

        windows.append(
            window
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

        print(
            f"Skipping unknown file: "
            f"{file_path.name}"
        )

        return 0, None

    print()
    print(
        f"Processing: "
        f"{file_path.name}"
    )

    # --------------------------------------------------------
    # Load raw sensor data
    # --------------------------------------------------------

    signal = load_sensor_file(
        file_path
    )

    print(
        f"  Raw shape: "
        f"{signal.shape}"
    )

    # --------------------------------------------------------
    # Verify feature count
    # --------------------------------------------------------

    if signal.shape[1] != NUM_FEATURES:

        raise ValueError(
            f"Expected "
            f"{NUM_FEATURES} features but "
            f"got {signal.shape[1]} "
            f"in {file_path.name}"
        )

    # --------------------------------------------------------
    # Create temporal windows
    # --------------------------------------------------------

    windows = create_windows(
        signal,
        SEQUENCE_LENGTH,
        STRIDE
    )

    if not windows:

        print(
            "  Too short - skipped."
        )

        return 0, None

    X = np.asarray(
        windows,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # DO NOT normalize each individual window here.
    #
    # The training script will calculate the mean/std using
    # TRAINING DATA ONLY and then apply the same scaler to
    # validation and test data.
    # --------------------------------------------------------

    y = np.full(
        len(X),
        label,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        f"{file_path.stem}.npz"
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
        f"  Type: {label_name}"
    )

    print(
        f"  Windows: {len(X)}"
    )

    print(
        f"  Output shape: {X.shape}"
    )

    print(
        f"  Saved: {output_file}"
    )

    return (
        len(X),
        label
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "SISFALL OUTDOOR SEQUENCE PREPARATION"
    )
    print("=" * 60)

    print(
        f"\nDataset:\n"
        f"{SISFALL_DIR}"
    )

    if not SISFALL_DIR.exists():

        raise FileNotFoundError(
            "\nSisFall dataset folder not found.\n\n"
            f"Expected:\n"
            f"{SISFALL_DIR}\n\n"
            "Check the folder location before "
            "continuing."
        )

    # --------------------------------------------------------
    # Find all files
    # --------------------------------------------------------

    files = sorted(
        SISFALL_DIR.rglob("*.txt")
    )

    print(
        f"\nSensor files found: "
        f"{len(files)}"
    )

    if len(files) == 0:

        print(
            "No .txt files found."
        )

        return

    # --------------------------------------------------------
    # DELETE OLD GENERATED SEQUENCES
    #
    # This is IMPORTANT because your current folder contains
    # old 3-channel .npz files.
    # --------------------------------------------------------

    old_files = list(
        OUTPUT_DIR.glob("*.npz")
    )

    print()
    print(
        f"Removing old sequence files: "
        f"{len(old_files)}"
    )

    for old_file in old_files:

        old_file.unlink()

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    total_windows = 0

    total_adl = 0

    total_fall = 0

    for file_path in files:

        windows, label = (
            process_file(
                file_path
            )
        )

        total_windows += windows

        if label == 0:

            total_adl += windows

        elif label == 1:

            total_fall += windows

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 60)
    print(
        "FINAL OUTDOOR SEQUENCE DATASET"
    )
    print("=" * 60)

    print(
        f"Total windows : "
        f"{total_windows}"
    )

    print(
        f"ADL windows   : "
        f"{total_adl}"
    )

    print(
        f"Fall windows  : "
        f"{total_fall}"
    )

    print(
        f"Features      : "
        f"{NUM_FEATURES}"
    )

    print(
        f"Sequence size : "
        f"{SEQUENCE_LENGTH}"
    )

    print()
    print(
        f"Saved to:\n"
        f"{OUTPUT_DIR}"
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()