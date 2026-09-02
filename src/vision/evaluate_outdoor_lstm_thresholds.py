from pathlib import Path
import random

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "outdoor"
    / "sequences"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "outdoor"
    / "best_outdoor_lstm.keras"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "outdoor"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_PATH = (
    RESULT_DIR
    / "outdoor_lstm_threshold_results.txt"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

SEQUENCE_LENGTH = 200
NUM_FEATURES = 9


# ============================================================
# SUBJECT ID
# ============================================================

def get_subject_id(file_path):

    parts = file_path.stem.split("_")

    for part in parts:

        if part.startswith("SA"):

            return part

    return None


# ============================================================
# CREATE SAME SPLIT USED DURING TRAINING
# ============================================================

def create_test_split():

    files = sorted(
        SEQUENCE_DIR.glob("*.npz")
    )

    if not files:

        raise FileNotFoundError(
            f"No sequence files found:\n"
            f"{SEQUENCE_DIR}"
        )

    groups = {}

    for file_path in files:

        subject = get_subject_id(
            file_path
        )

        if subject is None:

            raise ValueError(
                f"Cannot identify subject "
                f"from {file_path.name}"
            )

        groups.setdefault(
            subject,
            []
        ).append(file_path)

    subjects = sorted(
        groups.keys()
    )

    print(
        "Subjects found:",
        subjects
    )

    if len(subjects) != 2:

        raise ValueError(
            "This threshold script expects "
            "the same 2-subject setup used "
            "during training."
        )

    # Same setup as training:
    #
    # SA01 = development
    # SA02 = independent test
    #
    test_subject = subjects[1]

    test_files = sorted(
        groups[test_subject]
    )

    return test_files


# ============================================================
# LOAD DATA
# ============================================================

def load_files(files):

    X_list = []
    y_list = []

    for file_path in files:

        data = np.load(
            file_path,
            allow_pickle=True
        )

        X = data["X"]
        y = data["y"]

        if (
            X.shape[1] != SEQUENCE_LENGTH
            or X.shape[2] != NUM_FEATURES
        ):

            raise ValueError(
                f"Unexpected shape "
                f"{X.shape} in "
                f"{file_path.name}"
            )

        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(
        X_list,
        axis=0
    )

    y = np.concatenate(
        y_list,
        axis=0
    )

    return (
        X.astype(np.float32),
        y.astype(np.int32)
    )


# ============================================================
# RECREATE TRAINING FILE SPLIT
# ============================================================

def create_training_files():

    files = sorted(
        SEQUENCE_DIR.glob("*.npz")
    )

    groups = {}

    for file_path in files:

        subject = get_subject_id(
            file_path
        )

        groups.setdefault(
            subject,
            []
        ).append(file_path)

    subjects = sorted(
        groups.keys()
    )

    if len(subjects) != 2:

        raise ValueError(
            "Expected exactly 2 subjects."
        )

    development_subject = subjects[0]

    development_files = sorted(
        groups[
            development_subject
        ]
    )

    # EXACT SAME RANDOM SPLIT AS TRAINING
    rng = random.Random(
        SEED
    )

    development_files = (
        development_files.copy()
    )

    rng.shuffle(
        development_files
    )

    validation_count = max(
        1,
        int(
            len(
                development_files
            ) * 0.20
        )
    )

    train_files = (
        development_files[
            validation_count:
        ]
    )

    return sorted(
        train_files
    )


# ============================================================
# CALCULATE TRAINING SCALER
# ============================================================

def calculate_scaler(X_train):

    mean = np.mean(
        X_train,
        axis=(0, 1),
        keepdims=True
    )

    std = np.std(
        X_train,
        axis=(0, 1),
        keepdims=True
    )

    std[
        std < 1e-6
    ] = 1.0

    return mean, std


# ============================================================
# APPLY SCALER
# ============================================================

def apply_scaler(
    X,
    mean,
    std
):

    return (
        (
            X - mean
        ) / std
    ).astype(
        np.float32
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "OUTDOOR LSTM THRESHOLD EVALUATION"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{MODEL_PATH}"
        )

    print()
    print(
        "Loading trained outdoor LSTM..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded."
    )

    # --------------------------------------------------------
    # Recreate training data
    # --------------------------------------------------------

    print()
    print(
        "Recreating training split..."
    )

    train_files = (
        create_training_files()
    )

    print(
        f"Training files: "
        f"{len(train_files)}"
    )

    X_train, y_train = load_files(
        train_files
    )

    # --------------------------------------------------------
    # Test data
    # --------------------------------------------------------

    test_files = (
        create_test_split()
    )

    print(
        f"Test files: "
        f"{len(test_files)}"
    )

    X_test, y_test = load_files(
        test_files
    )

    print()
    print(
        "Training shape:",
        X_train.shape
    )

    print(
        "Test shape:",
        X_test.shape
    )

    # --------------------------------------------------------
    # Same scaler used during training
    # --------------------------------------------------------

    print()
    print(
        "Calculating scaler from training data..."
    )

    mean, std = calculate_scaler(
        X_train
    )

    X_test = apply_scaler(
        X_test,
        mean,
        std
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    print()
    print(
        "Generating test probabilities..."
    )

    probabilities = (
        model.predict(
            X_test,
            verbose=0
        ).ravel()
    )

    # --------------------------------------------------------
    # Thresholds
    # --------------------------------------------------------

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90
    ]

    results = []

    print()
    print(
        "=" * 75
    )

    print(
        "Threshold | Accuracy | Precision | Recall | F1"
    )

    print(
        "-" * 75
    )

    # --------------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------------

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(
            np.int32
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        results.append(
            (
                threshold,
                accuracy,
                precision,
                recall,
                f1
            )
        )

        print(
            f"{threshold:8.2f} | "
            f"{accuracy:8.4f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:6.4f}"
        )

    # ========================================================
    # BEST F1
    # ========================================================

    best_f1 = max(
        results,
        key=lambda x: x[4]
    )

    # ========================================================
    # BEST RECALL
    # ========================================================

    best_recall = max(
        results,
        key=lambda x: x[3]
    )

    # ========================================================
    # BEST BALANCED THRESHOLD
    #
    # Prefer the highest F1.
    # ========================================================

    print()
    print("=" * 60)
    print(
        "BEST F1 THRESHOLD"
    )
    print("=" * 60)

    print(
        f"Threshold : "
        f"{best_f1[0]:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{best_f1[1]:.4f}"
    )

    print(
        f"Precision : "
        f"{best_f1[2]:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_f1[3]:.4f}"
    )

    print(
        f"F1-score  : "
        f"{best_f1[4]:.4f}"
    )

    print()
    print("=" * 60)
    print(
        "BEST RECALL THRESHOLD"
    )
    print("=" * 60)

    print(
        f"Threshold : "
        f"{best_recall[0]:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{best_recall[1]:.4f}"
    )

    print(
        f"Precision : "
        f"{best_recall[2]:.4f}"
    )

    print(
        f"Recall    : "
        f"{best_recall[3]:.4f}"
    )

    print(
        f"F1-score  : "
        f"{best_recall[4]:.4f}"
    )

    # ========================================================
    # CONFUSION MATRIX FOR BEST F1
    # ========================================================

    best_predictions = (
        probabilities >= best_f1[0]
    ).astype(
        np.int32
    )

    best_cm = confusion_matrix(
        y_test,
        best_predictions
    )

    print()
    print(
        "Confusion Matrix "
        "(Best F1 Threshold):"
    )

    print(
        best_cm
    )

    # ========================================================
    # PROBABILITY INFORMATION
    # ========================================================

    fall_probabilities = (
        probabilities[
            y_test == 1
        ]
    )

    adl_probabilities = (
        probabilities[
            y_test == 0
        ]
    )

    print()
    print(
        "=" * 60
    )

    print(
        "PROBABILITY DISTRIBUTION"
    )

    print(
        "=" * 60
    )

    print(
        "ADL probability:"
    )

    print(
        f"  Min    : "
        f"{np.min(adl_probabilities):.4f}"
    )

    print(
        f"  Mean   : "
        f"{np.mean(adl_probabilities):.4f}"
    )

    print(
        f"  Median : "
        f"{np.median(adl_probabilities):.4f}"
    )

    print(
        f"  Max    : "
        f"{np.max(adl_probabilities):.4f}"
    )

    print()
    print(
        "FALL probability:"
    )

    print(
        f"  Min    : "
        f"{np.min(fall_probabilities):.4f}"
    )

    print(
        f"  Mean   : "
        f"{np.mean(fall_probabilities):.4f}"
    )

    print(
        f"  Median : "
        f"{np.median(fall_probabilities):.4f}"
    )

    print(
        f"  Max    : "
        f"{np.max(fall_probabilities):.4f}"
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "OUTDOOR SISFALL LSTM "
            "THRESHOLD RESULTS\n"
        )

        file.write(
            "=" * 60 +
            "\n\n"
        )

        file.write(
            "Threshold | Accuracy | "
            "Precision | Recall | F1\n"
        )

        file.write(
            "-" * 60 +
            "\n"
        )

        for row in results:

            file.write(
                f"{row[0]:.2f} | "
                f"{row[1]:.4f} | "
                f"{row[2]:.4f} | "
                f"{row[3]:.4f} | "
                f"{row[4]:.4f}\n"
            )

        file.write(
            "\n"
        )

        file.write(
            "BEST F1 THRESHOLD\n"
        )

        file.write(
            f"Threshold : "
            f"{best_f1[0]:.2f}\n"
        )

        file.write(
            f"Accuracy  : "
            f"{best_f1[1]:.4f}\n"
        )

        file.write(
            f"Precision : "
            f"{best_f1[2]:.4f}\n"
        )

        file.write(
            f"Recall    : "
            f"{best_f1[3]:.4f}\n"
        )

        file.write(
            f"F1-score  : "
            f"{best_f1[4]:.4f}\n\n"
        )

        file.write(
            "BEST RECALL THRESHOLD\n"
        )

        file.write(
            f"Threshold : "
            f"{best_recall[0]:.2f}\n"
        )

        file.write(
            f"Accuracy  : "
            f"{best_recall[1]:.4f}\n"
        )

        file.write(
            f"Precision : "
            f"{best_recall[2]:.4f}\n"
        )

        file.write(
            f"Recall    : "
            f"{best_recall[3]:.4f}\n"
        )

        file.write(
            f"F1-score  : "
            f"{best_recall[4]:.4f}\n\n"
        )

        file.write(
            "CONFUSION MATRIX - BEST F1\n"
        )

        file.write(
            str(best_cm)
        )

        file.write(
            "\n\n"
        )

        file.write(
            "ADL PROBABILITY DISTRIBUTION\n"
        )

        file.write(
            f"Min    : "
            f"{np.min(adl_probabilities):.4f}\n"
        )

        file.write(
            f"Mean   : "
            f"{np.mean(adl_probabilities):.4f}\n"
        )

        file.write(
            f"Median : "
            f"{np.median(adl_probabilities):.4f}\n"
        )

        file.write(
            f"Max    : "
            f"{np.max(adl_probabilities):.4f}\n\n"
        )

        file.write(
            "FALL PROBABILITY DISTRIBUTION\n"
        )

        file.write(
            f"Min    : "
            f"{np.min(fall_probabilities):.4f}\n"
        )

        file.write(
            f"Mean   : "
            f"{np.mean(fall_probabilities):.4f}\n"
        )

        file.write(
            f"Median : "
            f"{np.median(fall_probabilities):.4f}\n"
        )

        file.write(
            f"Max    : "
            f"{np.max(fall_probabilities):.4f}\n"
        )

    print()
    print(
        "Results saved:"
    )

    print(
        RESULT_PATH
    )

    print()
    print("=" * 60)
    print(
        "THRESHOLD EVALUATION COMPLETED"
    )
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()