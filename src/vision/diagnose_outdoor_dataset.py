from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "outdoor"
    / "sequences"
)


def get_subject(file_path):

    for part in file_path.stem.split("_"):

        if part.startswith("SA"):
            return part

    return "UNKNOWN"


def main():

    files = sorted(
        SEQUENCE_DIR.glob("*.npz")
    )

    print("=" * 70)
    print("OUTDOOR DATASET DIAGNOSTIC")
    print("=" * 70)

    subjects = {}

    for file_path in files:

        subject = get_subject(
            file_path
        )

        data = np.load(
            file_path
        )

        X = data["X"]
        y = data["y"]

        subjects.setdefault(
            subject,
            {
                "files": 0,
                "windows": 0,
                "adl": 0,
                "fall": 0,
                "mean": [],
                "std": [],
                "minimum": [],
                "maximum": []
            }
        )

        s = subjects[subject]

        s["files"] += 1
        s["windows"] += len(X)

        s["adl"] += int(
            np.sum(y == 0)
        )

        s["fall"] += int(
            np.sum(y == 1)
        )

        s["mean"].append(
            np.mean(X, axis=(0, 1))
        )

        s["std"].append(
            np.std(X, axis=(0, 1))
        )

        s["minimum"].append(
            np.min(X, axis=(0, 1))
        )

        s["maximum"].append(
            np.max(X, axis=(0, 1))
        )

    for subject, s in subjects.items():

        print()
        print("-" * 70)

        print(
            f"SUBJECT: {subject}"
        )

        print(
            f"Files   : {s['files']}"
        )

        print(
            f"Windows : {s['windows']}"
        )

        print(
            f"ADL     : {s['adl']}"
        )

        print(
            f"FALL    : {s['fall']}"
        )

        mean = np.mean(
            s["mean"],
            axis=0
        )

        std = np.mean(
            s["std"],
            axis=0
        )

        minimum = np.min(
            s["minimum"],
            axis=0
        )

        maximum = np.max(
            s["maximum"],
            axis=0
        )

        print()
        print(
            "Channel statistics:"
        )

        for i in range(9):

            print(
                f"Channel {i}: "
                f"mean={mean[i]:.4f}, "
                f"std={std[i]:.4f}, "
                f"min={minimum[i]:.4f}, "
                f"max={maximum[i]:.4f}"
            )

    print()
    print("=" * 70)
    print(
        "DIAGNOSTIC COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()