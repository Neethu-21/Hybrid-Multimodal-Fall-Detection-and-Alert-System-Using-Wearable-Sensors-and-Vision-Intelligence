from pathlib import Path
from collections import deque
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCALER_PATH = (
    PROJECT_ROOT
    / "models"
    / "outdoor"
    / "phone_compatible_lstm_scaler.npz"
)


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SIZE = 200

FEATURE_NAMES = [
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


buffer = deque(
    maxlen=WINDOW_SIZE
)


# ============================================================
# LOAD TRAINING SCALER
# ============================================================

if not SCALER_PATH.exists():

    raise FileNotFoundError(
        f"Scaler not found:\n{SCALER_PATH}"
    )

scaler = np.load(
    SCALER_PATH
)

train_mean = scaler["mean"]
train_std = scaler["std"]


# ============================================================
# LIVE SAMPLE
# ============================================================

def add_sample(sample):

    if len(sample) != 9:

        raise ValueError(
            f"Expected 9 values, "
            f"got {len(sample)}"
        )

    buffer.append(
        np.asarray(
            sample,
            dtype=np.float32
        )
    )


# ============================================================
# ANALYZE
# ============================================================

def analyze():

    if len(buffer) < WINDOW_SIZE:

        return

    X = np.asarray(
        buffer,
        dtype=np.float32
    )

    live_mean = np.mean(
        X,
        axis=0
    )

    live_std = np.std(
        X,
        axis=0
    )

    print()
    print("=" * 70)
    print("PHONE LIVE DATA DIAGNOSTIC")
    print("=" * 70)

    print()
    print(
        f"{'Feature':15s}"
        f"{'Live Mean':>14s}"
        f"{'Live Std':>14s}"
        f"{'Train Mean':>14s}"
        f"{'Train Std':>14s}"
    )

    print("-" * 70)

    for i, name in enumerate(
        FEATURE_NAMES
    ):

        print(
            f"{name:15s}"
            f"{live_mean[i]:14.4f}"
            f"{live_std[i]:14.4f}"
            f"{train_mean[i]:14.4f}"
            f"{train_std[i]:14.4f}"
        )

    # --------------------------------------------------------
    # Scaled live statistics
    # --------------------------------------------------------

    scaled = (
        X - train_mean
    ) / train_std

    scaled_mean = np.mean(
        scaled,
        axis=0
    )

    scaled_std = np.std(
        scaled,
        axis=0
    )

    print()
    print(
        "SCALED LIVE DATA"
    )

    print("-" * 70)

    print(
        f"{'Feature':15s}"
        f"{'Scaled Mean':>16s}"
        f"{'Scaled Std':>16s}"
    )

    print("-" * 70)

    for i, name in enumerate(
        FEATURE_NAMES
    ):

        print(
            f"{name:15s}"
            f"{scaled_mean[i]:16.4f}"
            f"{scaled_std[i]:16.4f}"
        )

    print()
    print("=" * 70)


# ============================================================
# HTTP RECEIVER
# ============================================================

from http.server import (
    BaseHTTPRequestHandler,
    HTTPServer
)

import json


class Handler(
    BaseHTTPRequestHandler
):

    def do_POST(self):

        try:

            length = int(
                self.headers.get(
                    "Content-Length",
                    0
                )
            )

            body = self.rfile.read(
                length
            )

            message = json.loads(
                body.decode(
                    "utf-8",
                    errors="replace"
                )
            )

            payload = message.get(
                "payload",
                []
            )

            sensors = {}

            for item in payload:

                name = item.get(
                    "name"
                )

                values = item.get(
                    "values"
                )

                if name not in {
                    "accelerometer",
                    "gravity",
                    "gyroscope"
                }:

                    continue

                if not isinstance(
                    values,
                    dict
                ):

                    continue

                try:

                    sensors[name] = [

                        float(
                            values["x"]
                        ),

                        float(
                            values["y"]
                        ),

                        float(
                            values["z"]
                        )
                    ]

                except (
                    KeyError,
                    TypeError,
                    ValueError
                ):

                    continue

            required = {
                "accelerometer",
                "gravity",
                "gyroscope"
            }

            if required.issubset(
                sensors.keys()
            ):

                sample = [

                    *sensors[
                        "accelerometer"
                    ],

                    *sensors[
                        "gravity"
                    ],

                    *sensors[
                        "gyroscope"
                    ]
                ]

                add_sample(
                    sample
                )

                if len(buffer) >= WINDOW_SIZE:

                    analyze()

                    # Stop after one diagnostic window
                    raise KeyboardInterrupt

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "application/json"
            )

            self.end_headers()

            self.wfile.write(
                b'{"status":"received"}'
            )

        except KeyboardInterrupt:

            raise

        except Exception as error:

            print(
                "Error:",
                error
            )

            try:

                self.send_response(400)
                self.end_headers()

            except Exception:

                pass

    def log_message(
        self,
        format,
        *args
    ):

        return


# ============================================================
# MAIN
# ============================================================

print()
print("=" * 70)
print("PHONE SENSOR DISTRIBUTION DIAGNOSTIC")
print("=" * 70)

print()
print(
    "Listening on http://0.0.0.0:8000"
)

print()
print(
    "Start Sensor Logger."
)

print(
    "Collecting one 200-sample window..."
)

print(
    "The diagnostic will stop automatically."
)

print()

server = HTTPServer(
    (
        "0.0.0.0",
        8000
    ),
    Handler
)

try:

    server.serve_forever()

except KeyboardInterrupt:

    print()
    print(
        "Diagnostic completed."
    )

finally:

    server.server_close()