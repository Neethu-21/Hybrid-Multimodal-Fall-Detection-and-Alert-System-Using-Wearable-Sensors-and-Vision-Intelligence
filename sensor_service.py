from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent

MODEL = ROOT / "models" / "outdoor" / "best_outdoor_lstm.keras"
SCALER = ROOT / "models" / "outdoor" / "outdoor_lstm_scaler.npz"

N = 200
THRESHOLD = 0.40

# ---------------------------------------------------------
# LOAD MODEL + SCALER
# ---------------------------------------------------------

model = tf.keras.models.load_model(MODEL)

s = np.load(SCALER)
mean = s["mean"]
std = s["std"]

safe_std = np.where(np.abs(std) < 1e-8, 1.0, std)

# Rolling sensor buffer
sensor_buffer = []

latest_probability = 0.0


# ---------------------------------------------------------
# EXTRACT 9 SENSOR VALUES FROM SENSOR LOGGER PAYLOAD
# ---------------------------------------------------------

def extract_sensor_values(payload):
    """
    Convert Sensor Logger payload into
    9-channel samples:

    acc_x, acc_y, acc_z,
    gravity_x, gravity_y, gravity_z,
    gyro_x, gyro_y, gyro_z
    """

    if not isinstance(payload, list):
        return []

    # Group sensor readings by timestamp
    grouped = {}

    for item in payload:

        if not isinstance(item, dict):
            continue

        name = item.get("name", "").lower()
        values = item.get("values", {})
        timestamp = item.get("time")

        if not isinstance(values, dict):
            continue

        if timestamp is None:
            continue

        try:
            x = float(values["x"])
            y = float(values["y"])
            z = float(values["z"])
        except (KeyError, TypeError, ValueError):
            continue

        if timestamp not in grouped:
            grouped[timestamp] = {}

        if name == "accelerometer":
            grouped[timestamp]["acc"] = [x, y, z]

        elif name == "gravity":
            grouped[timestamp]["gravity"] = [x, y, z]

        elif name == "gyroscope":
            grouped[timestamp]["gyro"] = [x, y, z]

    samples = []

    # Build the 9-channel vector
    for timestamp in sorted(grouped):

        item = grouped[timestamp]

        if (
            "acc" in item
            and "gravity" in item
            and "gyro" in item
        ):

            samples.append(
                item["acc"]
                + item["gravity"]
                + item["gyro"]
            )

    return samples
# ---------------------------------------------------------
# LSTM PREDICTION
# ---------------------------------------------------------

def predict_from_buffer():

    global sensor_buffer

    if len(sensor_buffer) < N:
        return None

    data = np.asarray(
        sensor_buffer[-N:],
        dtype=np.float32
    )

    if data.shape != (N, 9):
        return None

    X = (data - mean) / safe_std

    probability = model.predict(
        X[np.newaxis, :, :],
        verbose=0
    ).ravel()[0]

    return float(probability)


# ---------------------------------------------------------
# HTTP SERVER
# ---------------------------------------------------------

class SensorHandler(BaseHTTPRequestHandler):

    def do_POST(self):

        global sensor_buffer
        global latest_probability

        try:

            length = int(
                self.headers.get("Content-Length", 0)
            )

            body = self.rfile.read(length)

            data = json.loads(
                body.decode("utf-8")
            )

            print(
                "PHONE POST:",
                data.keys() if isinstance(data, dict)
                else type(data)
            )

            # -------------------------------------------------
            # Sensor Logger sends data inside "payload"
            # -------------------------------------------------

            payload = data.get(
                "payload",
                data
            )
            print("PAYLOAD SAMPLE:", json.dumps(payload)[:3000])

            new_samples = extract_sensor_values(
                payload
            )

            if not new_samples:

                raise ValueError(
                    "Could not extract the 9 required "
                    "sensor values from Sensor Logger payload."
                )

            sensor_buffer.extend(
                new_samples
            )

            # Keep only latest samples
            if len(sensor_buffer) > N:
                sensor_buffer = sensor_buffer[-N:]

            # Need 200 samples before prediction
            probability = predict_from_buffer()

            if probability is not None:

                latest_probability = probability

                fall = (
                    latest_probability >= THRESHOLD
                )

                print(
                    f"Samples: {len(sensor_buffer)}/{N} | "
                    f"Probability: "
                    f"{latest_probability:.4f} | "
                    f"Fall: {fall}"
                )

            else:

                fall = False

                print(
                    f"Samples: "
                    f"{len(sensor_buffer)}/{N} | "
                    f"Waiting for 200 samples..."
                )

            response = {
                "probability": latest_probability,
                "threshold": THRESHOLD,
                "fall": fall,
                "samples": len(sensor_buffer),
                "connected": True
            }

            out = json.dumps(
                response
            ).encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json"
            )
            self.send_header(
                "Access-Control-Allow-Origin",
                "*"
            )
            self.end_headers()

            self.wfile.write(out)

        except Exception as e:

            print(
                "Sensor error:",
                str(e)
            )

            out = json.dumps({
                "error": str(e),
                "connected": True
            }).encode("utf-8")

            self.send_response(400)

            self.send_header(
                "Content-Type",
                "application/json"
            )

            self.end_headers()

            self.wfile.write(out)


    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    def do_GET(self):

        if self.path == "/status":

            response = {
                "probability": latest_probability,
                "threshold": THRESHOLD,
                "fall": (
                    latest_probability >= THRESHOLD
                ),
                "samples": len(sensor_buffer),
                "connected": True
            }

            out = json.dumps(
                response
            ).encode("utf-8")

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "application/json"
            )

            self.end_headers()

            self.wfile.write(out)

        else:

            self.send_response(404)
            self.end_headers()


    def log_message(self, *args):
        return


# ---------------------------------------------------------
# START SERVER
# ---------------------------------------------------------

print("=" * 60)
print("LIVE PHONE SENSOR MONITOR")
print("=" * 60)
print(
    "Sensor Logger receiver: "
    "http://0.0.0.0:8000"
)
print(
    "Required sensors: "
    "Accelerometer + Gravity + Gyroscope"
)
print("Channels: 9")
print("Rolling window: 200 samples")
print("Fall threshold:", THRESHOLD)
print("Waiting for Sensor Logger...")
print("=" * 60)

server = HTTPServer(
    ("0.0.0.0", 8000),
    SensorHandler
)

server.serve_forever()