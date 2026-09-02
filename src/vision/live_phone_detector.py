from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import threading

from phone_detection import PhoneDetector


# ============================================================
# SETTINGS
# ============================================================

HOST = "0.0.0.0"
PORT = 8000

detector = PhoneDetector()

detector_lock = threading.Lock()


# ============================================================
# SENSOR EXTRACTION
# ============================================================

def extract_sensor_values(payload):

    sensors = {}

    for item in payload:

        name = item.get("name")
        values = item.get("values")

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

            x = float(values["x"])
            y = float(values["y"])
            z = float(values["z"])

        except (
            KeyError,
            TypeError,
            ValueError
        ):

            continue

        sensors[name] = [
            x,
            y,
            z
        ]

    required = {
        "accelerometer",
        "gravity",
        "gyroscope"
    }

    if not required.issubset(
        sensors.keys()
    ):

        return None

    return [
        *sensors["accelerometer"],
        *sensors["gravity"],
        *sensors["gyroscope"]
    ]


# ============================================================
# PROCESS SENSOR PAYLOAD
# ============================================================

def process_payload(payload):

    sample = extract_sensor_values(
        payload
    )

    if sample is None:

        return

    with detector_lock:

        detector.add_sample(
            sample
        )

        current_size = len(
            detector.buffer
        )

        if not detector.is_ready():

            print(
                f"\rBuffer: "
                f"{current_size}/"
                f"{detector.sequence_length}",
                end=""
            )

            return

        # ----------------------------------------------------
        # ACTUAL PHONE LSTM PREDICTION
        # ----------------------------------------------------

        result = detector.predict()

    # --------------------------------------------------------
    # DISPLAY PREDICTION
    # --------------------------------------------------------

    print()

    print(
        "-" * 60
    )

    print(
        "PHONE LSTM PREDICTION"
    )

    print(
        "-" * 60
    )

    print(
        f"Samples          : "
        f"{result['samples']}"
    )

    print(
        f"Fall probability : "
        f"{result['fall_probability']:.4f}"
    )

    print(
        f"Threshold        : "
        f"{result['threshold']:.2f}"
    )

    print(
        f"Prediction       : "
        f"{result['prediction']}"
    )

    if result["prediction"] == "FALL":

        print()

        print(
            "🚨 FALL DETECTED 🚨"
        )

    else:

        print()

        print(
            "✅ NO FALL"
        )

    print(
        "-" * 60
    )


# ============================================================
# HTTP HANDLER
# ============================================================

class SensorHandler(
    BaseHTTPRequestHandler
):

    def do_POST(self):

        try:

            content_length = int(
                self.headers.get(
                    "Content-Length",
                    0
                )
            )

            body = self.rfile.read(
                content_length
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

            process_payload(
                payload
            )

            self.send_response(200)

            self.send_header(
                "Content-Type",
                "application/json"
            )

            self.end_headers()

            response = {
                "status": "received"
            }

            self.wfile.write(
                json.dumps(
                    response
                ).encode("utf-8")
            )

        except (
            ConnectionResetError,
            BrokenPipeError
        ):

            return

        except Exception as error:

            print()

            print(
                "Request error:",
                error
            )

            try:

                self.send_response(400)

                self.end_headers()

            except Exception:

                pass

    def do_GET(self):

        with detector_lock:

            buffer_size = len(
                detector.buffer
            )

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "application/json"
        )

        self.end_headers()

        response = {

            "status": "running",

            "buffer_size":
                buffer_size,

            "required_samples":
                detector.sequence_length
        }

        self.wfile.write(
            json.dumps(
                response
            ).encode("utf-8")
        )

    def log_message(
        self,
        format,
        *args
    ):

        return


# ============================================================
# MAIN
# ============================================================

def main():

    print()

    print(
        "=" * 60
    )

    print(
        "LIVE PHONE SENSOR DETECTION PIPELINE"
    )

    print(
        "=" * 60
    )

    print()

    print(
        f"Listening on "
        f"http://{HOST}:{PORT}"
    )

    print()

    print(
        "Sensors:"
    )

    print(
        "  Accelerometer XYZ"
    )

    print(
        "  Gravity XYZ"
    )

    print(
        "  Gyroscope XYZ"
    )

    print()

    print(
        "Window:"
    )

    print(
        "  200 samples"
    )

    print()

    print(
        "Waiting for Sensor Logger..."
    )

    print(
        "Press CTRL+C to stop."
    )

    server = HTTPServer(
        (
            HOST,
            PORT
        ),
        SensorHandler
    )

    try:

        server.serve_forever()

    except KeyboardInterrupt:

        print()
        print()

        print(
            "Stopping live detector..."
        )

    finally:

        server.server_close()

        print()

        print(
            "Live detector stopped."
        )


if __name__ == "__main__":

    main()