from pathlib import Path
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
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

SEQUENCE_DIR = PROJECT_ROOT / "data" / "indoor" / "sequences"

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vision"
    / "best_gru.keras"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "vision"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70
]


# ============================================================
# GET SEQUENCES
# ============================================================

def get_sequence_files():

    files = sorted(SEQUENCE_DIR.glob("*.npz"))

    fall_files = [
        file for file in files
        if file.stem.lower().startswith("fall")
    ]

    adl_files = [
        file for file in files
        if file.stem.lower().startswith("adl")
    ]

    return fall_files, adl_files


# ============================================================
# SAME SPLIT USED DURING TRAINING
# ============================================================

def split_sequences(fall_files, adl_files):

    fall_train, fall_test = train_test_split(
        fall_files,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    fall_train, fall_val = train_test_split(
        fall_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    adl_train, adl_test = train_test_split(
        adl_files,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    adl_train, adl_val = train_test_split(
        adl_train,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE
    )

    test_files = fall_test + adl_test

    return test_files


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data(test_files):

    X_list = []
    y_list = []

    for file in test_files:

        data = np.load(file)

        X_list.append(data["X"])
        y_list.append(data["y"])

    X = np.concatenate(
        X_list,
        axis=0
    )

    y = np.concatenate(
        y_list,
        axis=0
    )

    # (samples, 20, 17, 2)
    # →
    # (samples, 20, 34)

    X = X.reshape(
        X.shape[0],
        X.shape[1],
        17 * 2
    )

    return X.astype(np.float32), y


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("GRU THRESHOLD EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading GRU model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded.")

    # --------------------------------------------------------
    # Get test sequences
    # --------------------------------------------------------

    fall_files, adl_files = get_sequence_files()

    test_files = split_sequences(
        fall_files,
        adl_files
    )

    print(
        f"\nTest sequences: {len(test_files)}"
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    X_test, y_test = load_test_data(
        test_files
    )

    print(
        f"Test samples: {len(y_test)}"
    )

    # --------------------------------------------------------
    # Get probabilities ONCE
    # --------------------------------------------------------

    probabilities = model.predict(
        X_test,
        verbose=0
    ).ravel()

    # --------------------------------------------------------
    # Evaluate thresholds
    # --------------------------------------------------------

    results = []

    print("\n")
    print(
        "Threshold | Accuracy | Precision | Recall | F1"
    )
    print("-" * 58)

    for threshold in THRESHOLDS:

        predictions = (
            probabilities >= threshold
        ).astype(int)

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
            f"{threshold:9.2f} | "
            f"{accuracy:8.4f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:6.4f}"
        )

    # --------------------------------------------------------
    # Best F1
    # --------------------------------------------------------

    best_f1 = max(
        results,
        key=lambda x: x[4]
    )

    # --------------------------------------------------------
    # Best Recall
    # --------------------------------------------------------

    best_recall = max(
        results,
        key=lambda x: x[3]
    )

    print("\n")
    print("=" * 60)
    print("BEST F1 THRESHOLD")
    print("=" * 60)

    print(
        f"Threshold : {best_f1[0]:.2f}"
    )

    print(
        f"Accuracy  : {best_f1[1]:.4f}"
    )

    print(
        f"Precision : {best_f1[2]:.4f}"
    )

    print(
        f"Recall    : {best_f1[3]:.4f}"
    )

    print(
        f"F1-score  : {best_f1[4]:.4f}"
    )

    print("\n")
    print("=" * 60)
    print("BEST RECALL THRESHOLD")
    print("=" * 60)

    print(
        f"Threshold : {best_recall[0]:.2f}"
    )

    print(
        f"Accuracy  : {best_recall[1]:.4f}"
    )

    print(
        f"Precision : {best_recall[2]:.4f}"
    )

    print(
        f"Recall    : {best_recall[3]:.4f}"
    )

    print(
        f"F1-score  : {best_recall[4]:.4f}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_file = (
        RESULT_DIR
        / "gru_threshold_results.txt"
    )

    with open(output_file, "w") as file:

        file.write(
            "GRU Threshold Evaluation\n"
        )

        file.write(
            "========================\n\n"
        )

        file.write(
            "Threshold,Accuracy,Precision,Recall,F1\n"
        )

        for row in results:

            file.write(
                f"{row[0]:.2f},"
                f"{row[1]:.6f},"
                f"{row[2]:.6f},"
                f"{row[3]:.6f},"
                f"{row[4]:.6f}\n"
            )

        file.write(
            "\nBest F1 threshold: "
            f"{best_f1[0]:.2f}\n"
        )

        file.write(
            "Best Recall threshold: "
            f"{best_recall[0]:.2f}\n"
        )

    print(
        f"\nSaved results to:\n{output_file}"
    )


if __name__ == "__main__":
    main()