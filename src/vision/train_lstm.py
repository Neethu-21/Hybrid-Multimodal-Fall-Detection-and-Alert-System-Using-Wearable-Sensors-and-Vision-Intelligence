from pathlib import Path
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "indoor"
    / "sequences"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "vision"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "vision"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

EPOCHS = 50
BATCH_SIZE = 16


# ============================================================
# LOAD DATA AT SEQUENCE LEVEL
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


def split_sequences(fall_files, adl_files):

    # --------------------------------------------------------
    # Split FALL sequences
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Split ADL sequences
    # --------------------------------------------------------

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

    train_files = fall_train + adl_train
    val_files = fall_val + adl_val
    test_files = fall_test + adl_test

    return train_files, val_files, test_files


# ============================================================
# LOAD WINDOWS FROM SELECTED SEQUENCES
# ============================================================

def load_sequences(files):

    X_list = []
    y_list = []

    for file in files:

        data = np.load(file)

        X = data["X"]
        y = data["y"]

        X_list.append(X)
        y_list.append(y)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y


# ============================================================
# PREPARE INPUT
# ============================================================

def reshape_input(X):

    # Current shape:
    #
    # (samples, 20, 17, 2)
    #
    # Convert every frame's 17 keypoints × 2 coordinates
    # into one feature vector:
    #
    # (samples, 20, 34)

    samples = X.shape[0]
    frames = X.shape[1]

    X = X.reshape(
        samples,
        frames,
        17 * 2
    )

    return X.astype(np.float32)


# ============================================================
# BUILD LSTM MODEL
# ============================================================

def build_model(input_shape):

    model = Sequential(
        [
            Input(shape=input_shape),

            LSTM(
                64,
                return_sequences=True
            ),

            Dropout(0.30),

            LSTM(
                32,
                return_sequences=False
            ),

            Dropout(0.30),

            Dense(
                32,
                activation="relu"
            ),

            Dropout(0.20),

            Dense(
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
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    probabilities = model.predict(
        X_test,
        verbose=0
    ).ravel()

    predictions = (
        probabilities >= 0.5
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

    print("\n" + "=" * 60)
    print("LSTM TEST RESULTS")
    print("=" * 60)

    print(
        f"Accuracy  : {accuracy:.4f}"
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1-score  : {f1:.4f}"
    )

    print("\nConfusion Matrix:")
    print(cm)

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    result_file = (
        RESULT_DIR
        / "lstm_metrics.txt"
    )

    with open(result_file, "w") as file:

        file.write(
            "Indoor Vision - LSTM\n"
        )

        file.write(
            "====================\n\n"
        )

        file.write(
            f"Accuracy: {accuracy:.6f}\n"
        )

        file.write(
            f"Precision: {precision:.6f}\n"
        )

        file.write(
            f"Recall: {recall:.6f}\n"
        )

        file.write(
            f"F1-score: {f1:.6f}\n"
        )

        file.write(
            "\nConfusion Matrix:\n"
        )

        file.write(
            str(cm)
        )

    return accuracy, precision, recall, f1


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("INDOOR VISION - LSTM TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Find sequences
    # --------------------------------------------------------

    fall_files, adl_files = get_sequence_files()

    print(
        f"\nFall sequences : {len(fall_files)}"
    )

    print(
        f"ADL sequences  : {len(adl_files)}"
    )

    if len(fall_files) == 0 or len(adl_files) == 0:

        raise RuntimeError(
            "Fall or ADL sequence files were not found."
        )

    # --------------------------------------------------------
    # Sequence-level split
    # --------------------------------------------------------

    train_files, val_files, test_files = split_sequences(
        fall_files,
        adl_files
    )

    print("\nSequence split:")
    print(
        f"Training   : {len(train_files)}"
    )
    print(
        f"Validation : {len(val_files)}"
    )
    print(
        f"Testing    : {len(test_files)}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    X_train, y_train = load_sequences(
        train_files
    )

    X_val, y_val = load_sequences(
        val_files
    )

    X_test, y_test = load_sequences(
        test_files
    )

    print("\nBefore reshaping:")
    print(
        "X_train:",
        X_train.shape
    )

    # --------------------------------------------------------
    # Reshape
    # --------------------------------------------------------

    X_train = reshape_input(X_train)
    X_val = reshape_input(X_val)
    X_test = reshape_input(X_test)

    print("\nAfter reshaping:")
    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_val:",
        X_val.shape
    )

    print(
        "X_test:",
        X_test.shape
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model(
        input_shape=X_train.shape[1:]
    )

    print("\nModel architecture:")
    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "best_lstm.keras"
    )

    callbacks = [

        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True
        ),

        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nStarting training...\n")

    history = model.fit(

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
    # Save final model
    # --------------------------------------------------------

    final_model_path = (
        MODEL_DIR
        / "lstm_final.keras"
    )

    model.save(
        final_model_path
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\n")
    print("=" * 60)
    print("LSTM TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()