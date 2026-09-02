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
    / "best_outdoor_lstm.keras"
)

RESULT_PATH = (
    RESULT_DIR
    / "outdoor_lstm_results.txt"
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

SEQUENCE_LENGTH = 200
NUM_FEATURES = 9

BATCH_SIZE = 64
EPOCHS = 50

LEARNING_RATE = 0.001


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# GET RECORDING ID
# ============================================================

def get_recording_id(file_path):

    # Example:
    #
    # D01_SA01_R01.npz
    #
    # D01_SA01_R01 is the complete recording.
    #
    # We use the complete filename as the group so that
    # overlapping windows from one recording can never
    # appear in different splits.

    return file_path.stem


# ============================================================
# GET SUBJECT
# ============================================================

def get_subject_id(file_path):

    for part in file_path.stem.split("_"):

        if part.startswith("SA"):

            return part

    return None


# ============================================================
# GET LABEL
# ============================================================

def get_label(file_path):

    name = file_path.stem.upper()

    if name.startswith("F"):

        return 1

    if name.startswith("D"):

        return 0

    return None


# ============================================================
# LOAD FILES
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

        if X.ndim != 3:

            raise ValueError(
                f"Invalid X shape "
                f"{X.shape} in "
                f"{file_path.name}"
            )

        if (
            X.shape[1] != SEQUENCE_LENGTH
            or X.shape[2] != NUM_FEATURES
        ):

            raise ValueError(
                f"Expected "
                f"(N,{SEQUENCE_LENGTH},"
                f"{NUM_FEATURES}) but got "
                f"{X.shape} in "
                f"{file_path.name}"
            )

        X_list.append(
            X
        )

        y_list.append(
            y
        )

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
# RECORDING-WISE SPLIT
# ============================================================

def create_recording_split():

    files = sorted(
        SEQUENCE_DIR.glob("*.npz")
    )

    if not files:

        raise FileNotFoundError(
            f"No sequence files found:\n"
            f"{SEQUENCE_DIR}"
        )

    print()
    print(
        f"Total sequence files: "
        f"{len(files)}"
    )

    # --------------------------------------------------------
    # Group by subject
    # --------------------------------------------------------

    subject_groups = {}

    for file_path in files:

        subject = get_subject_id(
            file_path
        )

        if subject is None:

            raise ValueError(
                f"Could not identify subject "
                f"from {file_path.name}"
            )

        subject_groups.setdefault(
            subject,
            []
        ).append(file_path)

    subjects = sorted(
        subject_groups.keys()
    )

    print(
        "Subjects:",
        subjects
    )

    # --------------------------------------------------------
    # Deterministic recording split
    #
    # 70% train
    # 15% validation
    # 15% test
    #
    # SPLIT IS BY RECORDING.
    # --------------------------------------------------------

    rng = random.Random(
        SEED
    )

    train_files = []
    val_files = []
    test_files = []

    for subject in subjects:

        subject_files = (
            subject_groups[
                subject
            ].copy()
        )

        rng.shuffle(
            subject_files
        )

        total = len(
            subject_files
        )

        if total < 3:

            raise ValueError(
                f"Subject {subject} has only "
                f"{total} recordings. "
                f"At least 3 are required "
                f"for train/validation/test."
            )

        test_count = max(
            1,
            int(
                round(
                    total * 0.15
                )
            )
        )

        val_count = max(
            1,
            int(
                round(
                    total * 0.15
                )
            )
        )

        # Guarantee at least one training recording
        if (
            test_count +
            val_count
            >= total
        ):

            test_count = 1
            val_count = 1

        test_part = (
            subject_files[
                :test_count
            ]
        )

        val_part = (
            subject_files[
                test_count:
                test_count + val_count
            ]
        )

        train_part = (
            subject_files[
                test_count + val_count:
            ]
        )

        train_files.extend(
            train_part
        )

        val_files.extend(
            val_part
        )

        test_files.extend(
            test_part
        )

        print()
        print(
            f"{subject}:"
        )

        print(
            f"  Train recordings : "
            f"{len(train_part)}"
        )

        print(
            f"  Val recordings   : "
            f"{len(val_part)}"
        )

        print(
            f"  Test recordings  : "
            f"{len(test_part)}"
        )

    return (
        sorted(train_files),
        sorted(val_files),
        sorted(test_files)
    )


# ============================================================
# BALANCE TRAINING DATA
# ============================================================

def balance_training_data(
    X,
    y
):

    adl_indices = np.where(
        y == 0
    )[0]

    fall_indices = np.where(
        y == 1
    )[0]

    print()
    print(
        "Before balancing:"
    )

    print(
        f"ADL  : {len(adl_indices)}"
    )

    print(
        f"FALL : {len(fall_indices)}"
    )

    if (
        len(adl_indices) == 0
        or len(fall_indices) == 0
    ):

        raise ValueError(
            "Training data must contain "
            "both ADL and FALL classes."
        )

    target = min(
        len(adl_indices),
        len(fall_indices)
    )

    rng = np.random.default_rng(
        SEED
    )

    adl_selected = rng.choice(
        adl_indices,
        size=target,
        replace=False
    )

    fall_selected = rng.choice(
        fall_indices,
        size=target,
        replace=False
    )

    selected = np.concatenate(
        [
            adl_selected,
            fall_selected
        ]
    )

    rng.shuffle(
        selected
    )

    X_balanced = X[
        selected
    ]

    y_balanced = y[
        selected
    ]

    print()
    print(
        "After balancing:"
    )

    print(
        f"ADL  : "
        f"{np.sum(y_balanced == 0)}"
    )

    print(
        f"FALL : "
        f"{np.sum(y_balanced == 1)}"
    )

    return (
        X_balanced,
        y_balanced
    )


# ============================================================
# CALCULATE SCALER
# ============================================================

def calculate_scaler(
    X_train
):

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

    return (
        mean,
        std
    )


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
# BUILD MODEL
# ============================================================

def build_model():

    inputs = tf.keras.Input(
        shape=(
            SEQUENCE_LENGTH,
            NUM_FEATURES
        )
    )

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            return_sequences=True
        )
    )(inputs)

    x = tf.keras.layers.Dropout(
        0.30
    )(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            32
        )
    )(x)

    x = tf.keras.layers.Dropout(
        0.30
    )(x)

    x = tf.keras.layers.Dense(
        32,
        activation="relu"
    )(x)

    x = tf.keras.layers.Dropout(
        0.20
    )(x)

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs
    )

    optimizer = (
        tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        )
    )

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            "accuracy"
        ]
    )

    return model


# ============================================================
# FIND BEST VALIDATION THRESHOLD
# ============================================================

def find_best_threshold(
    y_true,
    probabilities
):

    thresholds = np.arange(
        0.10,
        0.91,
        0.05
    )

    best = None

    print()
    print(
        "Validation threshold search:"
    )

    print(
        "Threshold | Accuracy | "
        "Precision | Recall | F1"
    )

    print(
        "-" * 65
    )

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(
            np.int32
        )

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
            f"{threshold:8.2f} | "
            f"{accuracy:8.4f} | "
            f"{precision:9.4f} | "
            f"{recall:6.4f} | "
            f"{f1:6.4f}"
        )

        if (
            best is None
            or f1 > best["f1"]
        ):

            best = {
                "threshold": threshold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }

    return best


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print(
        "OUTDOOR SISFALL - RECORDING-WISE LSTM"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Split by recording
    # --------------------------------------------------------

    (
        train_files,
        val_files,
        test_files
    ) = create_recording_split()

    print()
    print(
        "FINAL RECORDING SPLIT:"
    )

    print(
        f"Training recordings   : "
        f"{len(train_files)}"
    )

    print(
        f"Validation recordings : "
        f"{len(val_files)}"
    )

    print(
        f"Testing recordings    : "
        f"{len(test_files)}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X_train, y_train = load_files(
        train_files
    )

    X_val, y_val = load_files(
        val_files
    )

    X_test, y_test = load_files(
        test_files
    )

    print()
    print(
        "Before balancing:"
    )

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_val  :",
        X_val.shape
    )

    print(
        "X_test :",
        X_test.shape
    )

    print()
    print(
        "Training classes:"
    )

    print(
        "ADL  :",
        np.sum(y_train == 0)
    )

    print(
        "FALL :",
        np.sum(y_train == 1)
    )

    # --------------------------------------------------------
    # Balance training only
    # --------------------------------------------------------

    (
        X_train,
        y_train
    ) = balance_training_data(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Scale
    #
    # Calculate scaler ONLY from training data.
    # --------------------------------------------------------

    mean, std = (
        calculate_scaler(
            X_train
        )
    )
    SCALER_PATH = (
    MODEL_DIR
    / "outdoor_lstm_scaler.npz"
)

    np.savez(
    SCALER_PATH,
    mean=mean,
    std=std
)

    print()
    print("Scaler saved:")
    print(SCALER_PATH)
    
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

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    print()
    print(
        "MODEL ARCHITECTURE"
    )

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),

        tf.keras.callbacks.ModelCheckpoint(
            MODEL_PATH,
            monitor="val_loss",
            save_best_only=True,
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
        "Starting training..."
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

        callbacks=callbacks,

        verbose=1
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    print()
    print(
        "Loading best model..."
    )

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Validation probabilities
    # --------------------------------------------------------

    val_probabilities = (
        model.predict(
            X_val,
            verbose=0
        ).ravel()
    )

    # --------------------------------------------------------
    # Select threshold ONLY from validation
    # --------------------------------------------------------

    best = find_best_threshold(
        y_val,
        val_probabilities
    )

    threshold = (
        best["threshold"]
    )

    print()
    print("=" * 60)
    print(
        "SELECTED VALIDATION THRESHOLD"
    )
    print("=" * 60)

    print(
        f"Threshold : "
        f"{threshold:.2f}"
    )

    print(
        f"Validation F1 : "
        f"{best['f1']:.4f}"
    )

    print(
        f"Validation Recall : "
        f"{best['recall']:.4f}"
    )

    # --------------------------------------------------------
    # FINAL TEST
    # --------------------------------------------------------

    test_probabilities = (
        model.predict(
            X_test,
            verbose=0
        ).ravel()
    )

    test_predictions = (
        test_probabilities >= threshold
    ).astype(
        np.int32
    )

    accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        test_predictions
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print(
        "FINAL OUTDOOR LSTM TEST RESULTS"
    )
    print("=" * 60)

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
            "OUTDOOR SISFALL "
            "RECORDING-WISE LSTM RESULTS\n"
        )

        file.write(
            "=" * 60 +
            "\n\n"
        )

        file.write(
            f"Sequence length : "
            f"{SEQUENCE_LENGTH}\n"
        )

        file.write(
            f"Features        : "
            f"{NUM_FEATURES}\n\n"
        )

        file.write(
            "Recording split\n"
        )

        file.write(
            f"Training files   : "
            f"{len(train_files)}\n"
        )

        file.write(
            f"Validation files : "
            f"{len(val_files)}\n"
        )

        file.write(
            f"Testing files    : "
            f"{len(test_files)}\n\n"
        )

        file.write(
            "Selected validation threshold\n"
        )

        file.write(
            f"Threshold : "
            f"{threshold:.2f}\n"
        )

        file.write(
            f"Validation F1 : "
            f"{best['f1']:.4f}\n"
        )

        file.write(
            f"Validation Recall : "
            f"{best['recall']:.4f}\n\n"
        )

        file.write(
            "FINAL TEST RESULTS\n"
        )

        file.write(
            f"Accuracy  : "
            f"{accuracy:.4f}\n"
        )

        file.write(
            f"Precision : "
            f"{precision:.4f}\n"
        )

        file.write(
            f"Recall    : "
            f"{recall:.4f}\n"
        )

        file.write(
            f"F1-score  : "
            f"{f1:.4f}\n\n"
        )

        file.write(
            "Confusion Matrix:\n"
        )

        file.write(
            str(cm)
        )

    print()
    print(
        "Model saved:"
    )

    print(
        MODEL_PATH
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
        "OUTDOOR LSTM TRAINING COMPLETED"
    )
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()