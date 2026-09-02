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
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    GlobalAveragePooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SEQUENCE_DIR = PROJECT_ROOT / "data" / "indoor" / "sequences"
MODEL_DIR = PROJECT_ROOT / "models" / "vision"
RESULT_DIR = PROJECT_ROOT / "results" / "vision"

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
# FIND ORIGINAL SEQUENCES
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
# SEQUENCE-LEVEL SPLIT
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

    train_files = fall_train + adl_train
    val_files = fall_val + adl_val
    test_files = fall_test + adl_test

    return train_files, val_files, test_files


# ============================================================
# LOAD WINDOWS
# ============================================================

def load_sequences(files):

    X_list = []
    y_list = []

    for file in files:

        data = np.load(file)

        X_list.append(data["X"])
        y_list.append(data["y"])

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    return X, y


# ============================================================
# RESHAPE
# ============================================================

def reshape_input(X):

    # Original:
    # (samples, 20, 17, 2)
    #
    # CNN input:
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
# BUILD 1D CNN
# ============================================================

def build_model(input_shape):

    model = Sequential(
        [
            Input(shape=input_shape),

            Conv1D(
                filters=64,
                kernel_size=3,
                padding="same",
                activation="relu"
            ),

            MaxPooling1D(
                pool_size=2
            ),

            Dropout(0.25),

            Conv1D(
                filters=128,
                kernel_size=3,
                padding="same",
                activation="relu"
            ),

            MaxPooling1D(
                pool_size=2
            ),

            Dropout(0.25),

            GlobalAveragePooling1D(),

            Dense(
                64,
                activation="relu"
            ),

            Dropout(0.30),

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
        metrics=["accuracy"]
    )

    return model


# ============================================================
# EVALUATE
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
    print("CNN TEST RESULTS")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    result_file = RESULT_DIR / "cnn_metrics.txt"

    with open(result_file, "w") as file:

        file.write("Indoor Vision - CNN\n")
        file.write("===================\n\n")

        file.write(f"Accuracy: {accuracy:.6f}\n")
        file.write(f"Precision: {precision:.6f}\n")
        file.write(f"Recall: {recall:.6f}\n")
        file.write(f"F1-score: {f1:.6f}\n")

        file.write("\nConfusion Matrix:\n")
        file.write(str(cm))

    return accuracy, precision, recall, f1


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("INDOOR VISION - CNN TRAINING")
    print("=" * 60)

    fall_files, adl_files = get_sequence_files()

    print(f"\nFall sequences : {len(fall_files)}")
    print(f"ADL sequences  : {len(adl_files)}")

    if len(fall_files) == 0 or len(adl_files) == 0:
        raise RuntimeError(
            "Fall or ADL sequence files were not found."
        )

    train_files, val_files, test_files = split_sequences(
        fall_files,
        adl_files
    )

    print("\nSequence split:")
    print(f"Training   : {len(train_files)}")
    print(f"Validation : {len(val_files)}")
    print(f"Testing    : {len(test_files)}")

    X_train, y_train = load_sequences(train_files)
    X_val, y_val = load_sequences(val_files)
    X_test, y_test = load_sequences(test_files)

    print("\nBefore reshaping:")
    print("X_train:", X_train.shape)

    X_train = reshape_input(X_train)
    X_val = reshape_input(X_val)
    X_test = reshape_input(X_test)

    print("\nAfter reshaping:")
    print("X_train:", X_train.shape)
    print("X_val:", X_val.shape)
    print("X_test:", X_test.shape)

    model = build_model(
        input_shape=X_train.shape[1:]
    )

    print("\nModel architecture:")
    model.summary()

    model_path = MODEL_DIR / "best_cnn.keras"

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

    print("\nStarting training...\n")

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    final_model_path = MODEL_DIR / "cnn_final.keras"

    model.save(final_model_path)

    evaluate_model(
        model,
        X_test,
        y_test
    )

    print("\n")
    print("=" * 60)
    print("CNN TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()