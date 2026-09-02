from pathlib import Path
import re
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "outdoor"
    / "phone_compatible_sequences"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "outdoor"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "outdoor"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


MODEL_PATH = (
    MODEL_DIR
    / "phone_compatible_lstm.keras"
)

SCALER_PATH = (
    MODEL_DIR
    / "phone_compatible_lstm_scaler.npz"
)

RESULT_PATH = (
    RESULT_DIR
    / "phone_compatible_lstm_results.txt"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 200
NUM_FEATURES = 9

RANDOM_STATE = 42

EPOCHS = 30
BATCH_SIZE = 64

VALIDATION_SIZE = 0.20
TEST_SIZE = 0.20


# ============================================================
# LABEL FROM FILE NAME
# ============================================================

def get_label_from_name(name):

    name = name.upper()

    if re.match(r"^F\d{2}_", name):
        return 1

    if re.match(r"^D\d{2}_", name):
        return 0

    return None


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    files = sorted(
        DATA_DIR.glob("*.npz")
    )

    if not files:

        raise FileNotFoundError(
            f"No .npz files found in:\n{DATA_DIR}"
        )

    X_list = []
    y_list = []
    groups = []

    print()
    print(
        f"Sequence files found: "
        f"{len(files)}"
    )

    for file_path in files:

        data = np.load(
            file_path
        )

        X = data["X"]
        y = data["y"]

        if X.ndim != 3:

            raise ValueError(
                f"Invalid X shape "
                f"{X.shape} in "
                f"{file_path.name}"
            )

        if X.shape[1] != SEQUENCE_LENGTH:

            raise ValueError(
                f"Expected sequence length "
                f"{SEQUENCE_LENGTH}, got "
                f"{X.shape[1]} in "
                f"{file_path.name}"
            )

        if X.shape[2] != NUM_FEATURES:

            raise ValueError(
                f"Expected {NUM_FEATURES} "
                f"features, got "
                f"{X.shape[2]} in "
                f"{file_path.name}"
            )

        X_list.append(X)
        y_list.append(y)

        # Every window from one original
        # recording gets the same group.
        groups.extend(
            [file_path.stem] * len(X)
        )

    X = np.concatenate(
        X_list,
        axis=0
    )

    y = np.concatenate(
        y_list,
        axis=0
    )

    groups = np.asarray(
        groups
    )

    return X, y, groups


# ============================================================
# GROUP SPLIT
# ============================================================

def create_group_split(
    X,
    y,
    groups
):

    unique_groups = np.unique(
        groups
    )

    print()
    print(
        f"Unique recordings: "
        f"{len(unique_groups)}"
    )

    if len(unique_groups) < 3:

        raise ValueError(
            "Need at least 3 recordings."
        )

    # --------------------------------------------------------
    # First: separate test set
    # --------------------------------------------------------

    splitter_test = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    train_val_idx, test_idx = next(
        splitter_test.split(
            X,
            y,
            groups
        )
    )

    # --------------------------------------------------------
    # Second: validation from remaining data
    # --------------------------------------------------------

    train_val_groups = groups[
        train_val_idx
    ]

    splitter_val = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE
            / (1.0 - TEST_SIZE),
        random_state=RANDOM_STATE
    )

    train_idx_rel, val_idx_rel = next(
        splitter_val.split(
            X[train_val_idx],
            y[train_val_idx],
            train_val_groups
        )
    )

    train_idx = train_val_idx[
        train_idx_rel
    ]

    val_idx = train_val_idx[
        val_idx_rel
    ]

    return (
        train_idx,
        val_idx,
        test_idx
    )


# ============================================================
# SCALER
# ============================================================

def calculate_scaler(X_train):

    mean = np.mean(
        X_train,
        axis=(0, 1)
    )

    std = np.std(
        X_train,
        axis=(0, 1)
    )

    std[
        std < 1e-6
    ] = 1.0

    return mean, std


def apply_scaler(
    X,
    mean,
    std
):

    return (
        (
            X - mean
        )
        / std
    ).astype(
        np.float32
    )


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = tf.keras.Sequential(

        [

            tf.keras.layers.Input(
                shape=(
                    SEQUENCE_LENGTH,
                    NUM_FEATURES
                )
            ),

            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    64,
                    return_sequences=True
                )
            ),

            tf.keras.layers.Dropout(
                0.30
            ),

            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    32
                )
            ),

            tf.keras.layers.Dropout(
                0.30
            ),

            tf.keras.layers.Dense(
                32,
                activation="relu"
            ),

            tf.keras.layers.Dropout(
                0.20
            ),

            tf.keras.layers.Dense(
                1,
                activation="sigmoid"
            )
        ]
    )

    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def evaluate_thresholds(
    y_true,
    probabilities
):

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    print()
    print(
        "Validation threshold search:"
    )

    print(
        "Threshold | Accuracy | "
        "Precision | Recall | F1"
    )

    print("-" * 65)

    best_threshold = 0.50
    best_f1 = -1.0

    for threshold in thresholds:

        predictions = (
            probabilities
            >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_true,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_true,
            predictions,
            zero_division=0
        )

        print(
            f"{threshold:9.2f} | "
            f"{accuracy:8.4f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:6.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    return (
        best_threshold,
        best_f1
    )


# ============================================================
# FINAL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test,
    threshold
):

    probabilities = (
        model.predict(
            X_test,
            verbose=0
        ).ravel()
    )

    predictions = (
        probabilities
        >= threshold
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

    cm = confusion_matrix(
        y_test,
        predictions
    )

    return (
        accuracy,
        precision,
        recall,
        f1,
        cm
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "PHONE-COMPATIBLE SISFALL LSTM TRAINING"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y, groups = load_dataset()

    print()
    print(
        f"Dataset shape : {X.shape}"
    )

    print(
        f"Labels shape  : {y.shape}"
    )

    print(
        f"ADL windows   : "
        f"{np.sum(y == 0)}"
    )

    print(
        f"FALL windows  : "
        f"{np.sum(y == 1)}"
    )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    (
        train_idx,
        val_idx,
        test_idx
    ) = create_group_split(
        X,
        y,
        groups
    )

    X_train = X[
        train_idx
    ]

    y_train = y[
        train_idx
    ]

    X_val = X[
        val_idx
    ]

    y_val = y[
        val_idx
    ]

    X_test = X[
        test_idx
    ]

    y_test = y[
        test_idx
    ]

    print()
    print(
        "TRAINING:"
    )

    print(
        f"  Windows: "
        f"{len(X_train)}"
    )

    print(
        f"  Recordings: "
        f"{len(np.unique(groups[train_idx]))}"
    )

    print()
    print(
        "VALIDATION:"
    )

    print(
        f"  Windows: "
        f"{len(X_val)}"
    )

    print(
        f"  Recordings: "
        f"{len(np.unique(groups[val_idx]))}"
    )

    print()
    print(
        "TEST:"
    )

    print(
        f"  Windows: "
        f"{len(X_test)}"
    )

    print(
        f"  Recordings: "
        f"{len(np.unique(groups[test_idx]))}"
    )

    # --------------------------------------------------------
    # Scale using TRAINING data only
    # --------------------------------------------------------

    print()
    print(
        "Calculating scaler from "
        "training data..."
    )

    mean, std = calculate_scaler(
        X_train
    )

    X_train = apply_scaler(
        X_train,
        mean,
        std
    )

    X_val = apply_scaler(
        X_val,
        mean,
        std
    )

    X_test = apply_scaler(
        X_test,
        mean,
        std
    )

    np.savez(
        SCALER_PATH,
        mean=mean,
        std=std
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print()
    print(
        "Building phone-compatible LSTM..."
    )

    model = build_model()

    model.summary()

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------

    negative = np.sum(
        y_train == 0
    )

    positive = np.sum(
        y_train == 1
    )

    total = negative + positive

    class_weight = {

        0:
            total
            / (2.0 * negative),

        1:
            total
            / (2.0 * positive)
    }

    print()
    print(
        "Class weights:"
    )

    print(
        class_weight
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "TRAINING"
    )

    print(
        "=" * 60
    )

    model.fit(

        X_train,
        y_train,

        validation_data=(
            X_val,
            y_val
        ),

        epochs=EPOCHS,

        batch_size=BATCH_SIZE,

        class_weight=class_weight,

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print()
    print(
        "Loading best saved model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Validation threshold
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "VALIDATION THRESHOLD SELECTION"
    )

    print(
        "=" * 60
    )

    val_probabilities = (
        model.predict(
            X_val,
            verbose=0
        ).ravel()
    )

    (
        threshold,
        validation_f1
    ) = evaluate_thresholds(
        y_val,
        val_probabilities
    )

    print()
    print(
        "SELECTED THRESHOLD"
    )

    print(
        f"Threshold : "
        f"{threshold:.2f}"
    )

    print(
        f"Validation F1 : "
        f"{validation_f1:.4f}"
    )

    # --------------------------------------------------------
    # Final TEST
    # --------------------------------------------------------

    print()
    print(
        "=" * 60
    )

    print(
        "FINAL PHONE LSTM TEST RESULTS"
    )

    print(
        "=" * 60
    )

    (
        accuracy,
        precision,
        recall,
        f1,
        cm
    ) = evaluate_model(
        model,
        X_test,
        y_test,
        threshold
    )

    print()
    print(
        f"Threshold : "
        f"{threshold:.2f}"
    )

    print(
        f"Accuracy  : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1-score  : "
        f"{f1:.4f}"
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        cm
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "PHONE-COMPATIBLE SISFALL "
            "LSTM RESULTS\n"
        )

        file.write(
            "=" * 60
            + "\n\n"
        )

        file.write(
            f"Sequence length : "
            f"{SEQUENCE_LENGTH}\n"
        )

        file.write(
            f"Features        : "
            f"{NUM_FEATURES}\n"
        )

        file.write(
            f"Threshold       : "
            f"{threshold:.4f}\n\n"
        )

        file.write(
            f"Accuracy        : "
            f"{accuracy:.4f}\n"
        )

        file.write(
            f"Precision       : "
            f"{precision:.4f}\n"
        )

        file.write(
            f"Recall          : "
            f"{recall:.4f}\n"
        )

        file.write(
            f"F1-score        : "
            f"{f1:.4f}\n\n"
        )

        file.write(
            "Confusion Matrix:\n"
        )

        file.write(
            str(cm)
            + "\n"
        )

    print()
    print(
        f"Model saved:\n"
        f"{MODEL_PATH}"
    )

    print()
    print(
        f"Scaler saved:\n"
        f"{SCALER_PATH}"
    )

    print()
    print(
        f"Results saved:\n"
        f"{RESULT_PATH}"
    )

    print()
    print("=" * 60)
    print(
        "PHONE LSTM TRAINING COMPLETED"
    )
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()