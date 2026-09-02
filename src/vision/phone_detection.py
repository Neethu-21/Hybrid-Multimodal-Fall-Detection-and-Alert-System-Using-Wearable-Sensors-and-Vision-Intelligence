from collections import deque
from pathlib import Path

import numpy as np
import tensorflow as tf


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "outdoor"
    / "phone_compatible_lstm.keras"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "models"
    / "outdoor"
    / "phone_compatible_lstm_scaler.npz"
)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 200
NUM_FEATURES = 9

# Selected using validation data
FALL_THRESHOLD = 0.40


# ============================================================
# PHONE DETECTOR
# ============================================================

class PhoneDetector:

    def __init__(
        self,
        sequence_length=SEQUENCE_LENGTH
    ):

        self.sequence_length = (
            sequence_length
        )

        self.buffer = deque(
            maxlen=sequence_length
        )

        self.prediction_count = 0

        # ----------------------------------------------------
        # Load trained model
        # ----------------------------------------------------

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Phone LSTM model not found:\n"
                f"{MODEL_PATH}"
            )

        print()
        print(
            "Loading phone-compatible LSTM..."
        )

        self.model = (
            tf.keras.models.load_model(
                MODEL_PATH
            )
        )

        # ----------------------------------------------------
        # Load training scaler
        # ----------------------------------------------------

        if not SCALER_PATH.exists():

            raise FileNotFoundError(
                f"Phone LSTM scaler not found:\n"
                f"{SCALER_PATH}"
            )

        scaler = np.load(
            SCALER_PATH
        )

        self.mean = (
            scaler["mean"]
        )

        self.std = (
            scaler["std"]
        )

        print(
            "Phone LSTM model loaded."
        )

        print(
            f"Threshold: "
            f"{FALL_THRESHOLD:.2f}"
        )

    # ========================================================
    # ADD SAMPLE
    # ========================================================

    def add_sample(
        self,
        sample
    ):

        if len(sample) != NUM_FEATURES:

            raise ValueError(
                f"Expected {NUM_FEATURES} "
                f"features, got "
                f"{len(sample)}"
            )

        self.buffer.append(
            np.asarray(
                sample,
                dtype=np.float32
            )
        )

    # ========================================================
    # READY?
    # ========================================================

    def is_ready(self):

        return (
            len(self.buffer)
            >= self.sequence_length
        )

    # ========================================================
    # GET WINDOW
    # ========================================================

    def get_window(self):

        if not self.is_ready():

            return None

        return np.asarray(
            self.buffer,
            dtype=np.float32
        )

    # ========================================================
    # PREDICT
    # ========================================================

    def predict(self):

        window = self.get_window()

        if window is None:

            return {
                "status": "WAITING",
                "samples": len(
                    self.buffer
                )
            }

        # ----------------------------------------------------
        # Apply EXACT training scaler
        # ----------------------------------------------------

        scaled_window = (
            (
                window
                - self.mean
            )
            / self.std
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Model expects:
        #
        # (batch, 200, 9)
        # ----------------------------------------------------

        model_input = (
            np.expand_dims(
                scaled_window,
                axis=0
            )
        )

        probability = float(
            self.model.predict(
                model_input,
                verbose=0
            )[0][0]
        )

        prediction = (
            "FALL"
            if probability
            >= FALL_THRESHOLD
            else "NO FALL"
        )

        self.prediction_count += 1

        return {

            "status": "PREDICTED",

            "samples":
                len(window),

            "fall_probability":
                probability,

            "threshold":
                FALL_THRESHOLD,

            "prediction":
                prediction,

            "prediction_number":
                self.prediction_count
        }