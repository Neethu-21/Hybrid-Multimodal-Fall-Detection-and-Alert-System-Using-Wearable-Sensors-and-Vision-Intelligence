from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from datetime import datetime
import json


HOST = "0.0.0.0"
PORT = 8000

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RECORDINGS_DIR = (
    PROJECT_ROOT
    / "data"
    / "outdoor"
    / "phone_sensor"
)

RECORDINGS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RECORDING STATE
# ============================================================

current_file = None
current_label = None


# ============================================================
# SENSOR EXTRACTION
# ============================================================

def extract_sensor_rows(payload):
    """
    Extract:

        Accelerometer XYZ
        Gravity XYZ
        Gyroscope XYZ

    from Sensor Logger JSON.

    Returns rows with 9 values.
    """

    # Sensor Logger payload structure can vary.
    # Save the raw payload first so we never lose data.
    return payload


# ============================================================
# HTTP HANDLER
# ============================================================

class SensorHandler(BaseHTTPRequestHandler):

    def do_POST(self):

        global current_file

        content_length = int(
            self.headers.get(
                "Content-Length",
                0
            )
        )

        body = self.rfile.read(
            content_length
        )

        try:

            payload = json.loads(
                body.decode(
                    "utf-8",
                    errors="replace"
                )
            )

        except Exception as e:

            print(
                "Could not parse JSON:",
                e
            )

            self.send_response(400)
            self.end_headers()

            return

        # ----------------------------------------------------
        # Create recording file if needed
        # ----------------------------------------------------

        if current_file is None:

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            current_file = (
                RECORDINGS_DIR
                / f"recording_{timestamp}.jsonl"
            )

            print()
            print(
                "Recording started:"
            )

            print(
                current_file
            )

        # ----------------------------------------------------
        # Save raw Sensor Logger payload
        # ----------------------------------------------------

        with open(
            current_file,
            "a",
            encoding="utf-8"
        ) as file:

            file.write(
                json.dumps(
                    payload
                )
                + "\n"
            )

        print(
            f"\rPayloads received: "
            f"{current_file.stat().st_size:,} bytes",
            end=""
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

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
            ).encode(
                "utf-8"
            )
        )

    def do_GET(self):

        self.send_response(200)

        self.send_header(
            "Content-Type",
            "text/plain"
        )

        self.end_headers()

        self.wfile.write(
            b"Sensor receiver is running."
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
    print("=" * 60)
    print("PHONE SENSOR DATA LOGGER")
    print("=" * 60)

    print()
    print(
        f"Listening on "
        f"http://{HOST}:{PORT}"
    )

    print()
    print(
        "Saving recordings to:"
    )

    print(
        RECORDINGS_DIR
    )

    print()
    print(
        "Start Sensor Logger on the phone."
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
            "Stopping logger..."
        )

    finally:

        server.server_close()

        print()
        print(
            "Logger stopped."
        )


if __name__ == "__main__":

    main()