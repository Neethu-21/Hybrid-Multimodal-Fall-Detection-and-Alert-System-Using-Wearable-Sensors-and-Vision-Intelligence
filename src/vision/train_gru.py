from pathlib import Path
import numpy as np
import tensorflow as tf

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    GRU,
    Dense,
    Dropout,
    Bidirectional
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


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
    / "best_gru.keras"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 20
FEATURES = 34

EPOCHS = 60
BATCH_SIZE = 16


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    files = sorted(
        SEQUENCE_DIR.glob("*.npz")
    )

    if not files:

        raise FileNotFoundError(
            f"No sequence files found:\n"
            f"{SEQUENCE_DIR}"
        )

    X_list = []
    y_list = []
    groups = []

    print()
    print("=" * 60)
    print("LOADING SEQUENCE DATA")
    print("=" * 60)

    for file in files:

        data = np.load(
            file,
            allow_pickle=True
        )

        X = data["X"]
        y = data["y"]

        X = np.asarray(
            X,
            dtype=np.float32
        )

        y = np.asarray(
            y,
            dtype=np.int32
        )

        # ----------------------------------------------------
        # Validate shape
        # ----------------------------------------------------

        if X.ndim != 4:

            raise ValueError(
                f"Invalid X shape in "
                f"{file.name}: {X.shape}"
            )

        if (
            X.shape[1] != SEQUENCE_LENGTH
            or
            X.shape[2] != 17
            or
            X.shape[3] != 2
        ):

            raise ValueError(
                f"Unexpected X shape in "
                f"{file.name}: {X.shape}"
            )

        # ----------------------------------------------------
        # Flatten 17 x 2 -> 34
        # ----------------------------------------------------

        X = X.reshape(
            X.shape[0],
            SEQUENCE_LENGTH,
            FEATURES
        )

        X_list.append(X)
        y_list.append(y)

        # IMPORTANT:
        # All windows from the same original recording
        # belong to the same group.
        groups.extend(
            [file.stem] * len(X)
        )

        print(
            f"{file.stem:35s}"
            f" windows={len(X):4d}"
            f" fall={int(np.sum(y == 1)):4d}"
            f" adl={int(np.sum(y == 0)):4d}"
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

    print()
    print(
        "Total samples:",
        len(X)
    )

    print(
        "ADL:",
        int(np.sum(y == 0))
    )

    print(
        "Fall:",
        int(np.sum(y == 1))
    )

    return X, y, groups


# ============================================================
# GROUP SPLIT
# ============================================================

def split_data(
    X,
    y,
    groups
):

    # --------------------------------------------------------
    # 80% training
    # 20% temporary
    # --------------------------------------------------------

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED
    )

    train_idx, temp_idx = next(
        splitter.split(
            X,
            y,
            groups
        )
    )

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_temp = X[temp_idx]
    y_temp = y[temp_idx]

    groups_temp = groups[temp_idx]

    # --------------------------------------------------------
    # Temporary -> validation + test
    # --------------------------------------------------------

    splitter2 = GroupShuffleSplit(
        n_splits=1,
        test_size=0.50,
        random_state=SEED + 1
    )

    val_idx, test_idx = next(
        splitter2.split(
            X_temp,
            y_temp,
            groups_temp
        )
    )

    X_val = X_temp[val_idx]
    y_val = y_temp[val_idx]

    X_test = X_temp[test_idx]
    y_test = y_temp[test_idx]

    print()
    print("=" * 60)
    print("GROUPED DATA SPLIT")
    print("=" * 60)

    print(
        "Training   :",
        len(X_train)
    )

    print(
        "Validation :",
        len(X_val)
    )

    print(
        "Testing    :",
        len(X_test)
    )

    print()
    print(
        "Training ADL  :",
        int(np.sum(y_train == 0))
    )

    print(
        "Training fall :",
        int(np.sum(y_train == 1))
    )

    print()
    print(
        "Validation ADL :",
        int(np.sum(y_val == 0))
    )

    print(
        "Validation fall:",
        int(np.sum(y_val == 1))
    )

    print()
    print(
        "Testing ADL:",
        int(np.sum(y_test == 0))
    )

    print(
        "Testing fall:",
        int(np.sum(y_test == 1))
    )

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    )


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = Sequential(
        [

            Input(
                shape=(
                    SEQUENCE_LENGTH,
                    FEATURES
                )
            ),

            Bidirectional(
                GRU(
                    64,
                    return_sequences=True
                )
            ),

            Dropout(0.30),

            Bidirectional(
                GRU(
                    32,
                    return_sequences=False
                )
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
            "accuracy",
            tf.keras.metrics.Precision(
                name="precision"
            ),
            tf.keras.metrics.Recall(
                name="recall"
            )
        ]
    )

    return model


# ============================================================
# VALIDATION F1 CALLBACK
# ============================================================

class ValidationF1Callback(
    tf.keras.callbacks.Callback
):

    def __init__(
        self,
        X_val,
        y_val
    ):

        super().__init__()

        self.X_val = X_val
        self.y_val = y_val

        self.best_f1 = -1.0

    def on_epoch_end(
        self,
        epoch,
        logs=None
    ):

        probabilities = (
            self.model.predict(
                self.X_val,
                verbose=0
            ).reshape(-1)
        )

        predictions = (
            probabilities >= 0.35
        ).astype(int)

        f1 = f1_score(
            self.y_val,
            predictions,
            zero_division=0
        )

        precision = precision_score(
            self.y_val,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            self.y_val,
            predictions,
            zero_division=0
        )

        print(
            f" - val_f1: {f1:.4f}"
            f" - val_precision_custom: "
            f"{precision:.4f}"
            f" - val_recall_custom: "
            f"{recall:.4f}"
        )

        if f1 > self.best_f1:

            self.best_f1 = f1

            self.model.save(
                MODEL_PATH
            )

            print(
                f"  Saved improved F1 model "
                f"(val F1={f1:.4f})"
            )


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def evaluate_thresholds(
    model,
    X,
    y
):

    probabilities = (
        model.predict(
            X,
            verbose=0
        ).reshape(-1)
    )

    thresholds = np.arange(
        0.20,
        0.71,
        0.05
    )

    best_f1 = -1.0
    best_threshold = 0.35

    best_recall = -1.0
    best_recall_threshold = 0.35

    print()
    print("=" * 70)
    print("THRESHOLD EVALUATION")
    print("=" * 70)

    print(
        f"{'Threshold':>10}"
        f"{'Accuracy':>12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
    )

    print("-" * 60)

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y,
            predictions
        )

        precision = precision_score(
            y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0
        )

        print(
            f"{threshold:10.2f}"
            f"{accuracy:12.4f}"
            f"{precision:12.4f}"
            f"{recall:12.4f}"
            f"{f1:12.4f}"
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

        if recall > best_recall:

            best_recall = recall
            best_recall_threshold = threshold

    return (
        probabilities,
        best_threshold,
        best_f1,
        best_recall_threshold,
        best_recall
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("INDOOR VISION - FINAL GRU TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    X, y, groups = load_data()

    # --------------------------------------------------------
    # Split by recording
    # --------------------------------------------------------

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test
    ) = split_data(
        X,
        y,
        groups
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    print()
    print("=" * 60)
    print("MODEL ARCHITECTURE")
    print("=" * 60)

    model.summary()

    # --------------------------------------------------------
    # Class weights
    # --------------------------------------------------------
    #
    # We use balanced weighting, but do NOT artificially
    # multiply the fall class. The previous experiment showed
    # that excessive fall weighting caused massive false
    # positives.
    # --------------------------------------------------------

    negative = np.sum(
        y_train == 0
    )

    positive = np.sum(
        y_train == 1
    )

    total = (
        negative +
        positive
    )

    class_weight = {

        0: total / (
            2.0 * negative
        ),

        1: total / (
            2.0 * positive
        )
    }

    print()
    print(
        "Class weights:",
        class_weight
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    f1_callback = (
        ValidationF1Callback(
            X_val,
            y_val
        )
    )

    callbacks = [

        f1_callback,

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1
        ),

        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

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
        shuffle=True,
        verbose=1
    )

    # --------------------------------------------------------
    # Load model selected using validation F1
    # --------------------------------------------------------

    print()
    print(
        "Loading best validation-F1 model..."
    )

    best_model = (
        tf.keras.models.load_model(
            MODEL_PATH
        )
    )

    # --------------------------------------------------------
    # Find threshold on VALIDATION set
    # --------------------------------------------------------

    (
        _,
        validation_threshold,
        validation_f1,
        validation_recall_threshold,
        validation_recall
    ) = evaluate_thresholds(
        best_model,
        X_val,
        y_val
    )

    print()
    print("=" * 60)
    print("SELECTED VALIDATION THRESHOLD")
    print("=" * 60)

    print(
        f"Best F1 threshold : "
        f"{validation_threshold:.2f}"
    )

    print(
        f"Validation F1     : "
        f"{validation_f1:.4f}"
    )

    print(
        f"Best recall threshold : "
        f"{validation_recall_threshold:.2f}"
    )

    print(
        f"Validation recall    : "
        f"{validation_recall:.4f}"
    )

    # --------------------------------------------------------
    # Final TEST evaluation
    #
    # IMPORTANT:
    # Threshold was selected using validation data.
    # Test data is used only once for final evaluation.
    # --------------------------------------------------------

    test_probabilities = (
        best_model.predict(
            X_test,
            verbose=0
        ).reshape(-1)
    )

    test_predictions = (
        test_probabilities
        >= validation_threshold
    ).astype(int)

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

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("FINAL GRU TEST RESULTS")
    print("=" * 60)

    print(
        f"Threshold : "
        f"{validation_threshold:.2f}"
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

    print(cm)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_file = (
        RESULT_DIR
        / "gru_final_results.txt"
    )

    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "FINAL GRU TEST RESULTS\n"
        )

        f.write(
            "======================\n\n"
        )

        f.write(
            f"Threshold : "
            f"{validation_threshold:.4f}\n"
        )

        f.write(
            f"Accuracy  : "
            f"{accuracy:.4f}\n"
        )

        f.write(
            f"Precision : "
            f"{precision:.4f}\n"
        )

        f.write(
            f"Recall    : "
            f"{recall:.4f}\n"
        )

        f.write(
            f"F1-score  : "
            f"{f1:.4f}\n"
        )

        f.write(
            "\nConfusion Matrix:\n"
        )

        f.write(
            str(cm)
        )

        f.write(
            "\n\nValidation-selected threshold: "
            f"{validation_threshold:.4f}\n"
        )

    print()
    print(
        "Model saved:"
    )

    print(MODEL_PATH)

    print()
    print(
        "Results saved:"
    )

    print(result_file)

    print()
    print("=" * 60)
    print("GRU TRAINING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()