from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PHONE_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "outdoor"
    / "phone_sensor"
)

ADL_DIR = PHONE_DATA_DIR / "adl"
FALL_DIR = PHONE_DATA_DIR / "fall"

OUTPUT_DIR = (
    PHONE_DATA_DIR
    / "sequences"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 200

# 50% overlap
STRIDE = 100

FEATURE_COLUMNS = [
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

NUM_FEATURES = 9


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(file_path):

    df = pd.read_csv(file_path)

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"\nMissing columns in {file_path.name}:\n"
            f"{missing}"
        )

    data = df[
        FEATURE_COLUMNS
    ].astype(
        np.float32
    ).values

    return data


# ============================================================
# NORMALIZE
# ============================================================

def normalize_data(data):

    mean = np.mean(
        data,
        axis=0,
        keepdims=True
    )

    std = np.std(
        data,
        axis=0,
        keepdims=True
    )

    std[
        std < 1e-6
    ] = 1.0

    normalized = (
        data - mean
    ) / std

    return normalized.astype(
        np.float32
    )


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows(
    data,
    sequence_length,
    stride
):

    windows = []

    total_samples = len(data)

    if total_samples < sequence_length:

        return windows

    for start in range(
        0,
        total_samples - sequence_length + 1,
        stride
    ):

        end = (
            start
            + sequence_length
        )

        windows.append(
            data[start:end]
        )

    return windows


# ============================================================
# PROCESS ONE FILE
# ============================================================

def process_file(
    file_path,
    label
):

    print()
    print(
        f"Processing: "
        f"{file_path.name}"
    )

    data = load_csv(
        file_path
    )

    print(
        f"  Samples: "
        f"{len(data)}"
    )

    print(
        f"  Features: "
        f"{data.shape[1]}"
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    normalized = normalize_data(
        data
    )

    # --------------------------------------------------------
    # Windows
    # --------------------------------------------------------

    windows = create_windows(
        normalized,
        SEQUENCE_LENGTH,
        STRIDE
    )

    if len(windows) == 0:

        print(
            "  Too short for "
            f"{SEQUENCE_LENGTH}-sample window."
        )

        return 0

    X = np.asarray(
        windows,
        dtype=np.float32
    )

    y = np.full(
        len(X),
        label,
        dtype=np.int64
    )

    # --------------------------------------------------------
    # Output filename
    # --------------------------------------------------------

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
        f"  Label: {label_name}"
    )

    print(
        f"  Windows: "
        f"{len(X)}"
    )

    print(
        f"  Shape: "
        f"{X.shape}"
    )

    print(
        f"  Saved: "
        f"{output_file}"
    )

    return len(X)


# ============================================================
# PROCESS DIRECTORY
# ============================================================

def process_directory(
    directory,
    label
):

    if not directory.exists():

        print(
            f"\nDirectory not found:"
            f"\n{directory}"
        )

        return 0

    files = sorted(
        directory.glob("*.csv")
    )

    print()
    print(
        f"{directory.name.upper()} "
        f"FILES: {len(files)}"
    )

    total = 0

    for file_path in files:

        total += process_file(
            file_path,
            label
        )

    return total


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("PHONE SENSOR SEQUENCE PREPARATION")
    print("=" * 60)

    print()
    print(
        f"Sequence length : "
        f"{SEQUENCE_LENGTH}"
    )

    print(
        f"Stride          : "
        f"{STRIDE}"
    )

    print(
        f"Features        : "
        f"{NUM_FEATURES}"
    )

    # --------------------------------------------------------
    # ADL
    # --------------------------------------------------------

    total_adl = process_directory(
        ADL_DIR,
        label=0
    )

    # --------------------------------------------------------
    # FALL
    # --------------------------------------------------------

    total_fall = process_directory(
        FALL_DIR,
        label=1
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL PHONE SEQUENCE DATASET")
    print("=" * 60)

    print(
        f"Total windows : "
        f"{total_adl + total_fall}"
    )

    print(
        f"ADL windows   : "
        f"{total_adl}"
    )

    print(
        f"Fall windows  : "
        f"{total_fall}"
    )

    print()
    print(
        f"Saved to:"
    )

    print(
        OUTPUT_DIR
    )

    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()