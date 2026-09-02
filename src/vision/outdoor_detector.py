import streamlit as st
import numpy as np
import cv2
import tempfile
import time
import json
import threading
from pathlib import Path
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer

# ============================================================
# OPTIONAL IMPORTS
# ============================================================

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        VideoProcessorBase,
    )
    WEBRTC_OK = True
except ImportError:
    WEBRTC_OK = False


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Smart Fall Detection System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def find_first(*paths):
    for p in paths:
        if p.exists():
            return p
    return None


YOLO_MODEL_PATH = find_first(
    PROJECT_ROOT / "yolo11n-pose.pt",
    PROJECT_ROOT / "models" / "vision" / "yolo11n-pose.pt",
)

VISION_MODEL_PATH = find_first(
    PROJECT_ROOT / "models" / "vision" / "best_gru.keras",
    PROJECT_ROOT / "models" / "vision" / "vision_fall_model.keras",
    PROJECT_ROOT / "best_gru.keras",
)

OUTDOOR_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "outdoor"
    / "best_outdoor_lstm.keras"
)

OUTDOOR_SCALER_PATH = (
    PROJECT_ROOT
    / "models"
    / "outdoor"
    / "outdoor_lstm_scaler.npz"
)


# ============================================================
# SETTINGS
# ============================================================

PHONE_PORT = 8000

VISION_SEQUENCE_LENGTH = 20
VISION_THRESHOLD = 0.35
VISION_FALL_CONFIRMATIONS = 3
VISION_NORMAL_CONFIRMATIONS = 5

OUTDOOR_SEQUENCE_LENGTH = 200
OUTDOOR_THRESHOLD = 0.40

PHONE_ONLINE_SECONDS = 5
PHONE_IDLE_SECONDS = 15


# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.3rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }

    .hero {
        padding: 30px;
        border-radius: 20px;
        background: linear-gradient(135deg,#eff6ff,#f8fafc);
        border: 1px solid #dbeafe;
        margin-bottom: 24px;
    }

    .hero-kicker {
        color: #2563eb;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1.3px;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .hero-title {
        color: #111827;
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .hero-text {
        color: #4b5563;
        font-size: 16px;
        line-height: 1.6;
        max-width: 900px;
    }

    .status-fall {
        padding: 16px;
        border-radius: 12px;
        background: #fee2e2;
        border: 2px solid #ef4444;
        color: #991b1b;
        text-align: center;
        font-size: 24px;
        font-weight: 800;
    }

    .status-safe {
        padding: 16px;
        border-radius: 12px;
        background: #dcfce7;
        border: 2px solid #22c55e;
        color: #166534;
        text-align: center;
        font-size: 23px;
        font-weight: 750;
    }

    .status-wait {
        padding: 16px;
        border-radius: 12px;
        background: #f3f4f6;
        border: 1px solid #d1d5db;
        color: #374151;
        text-align: center;
        font-size: 20px;
        font-weight: 650;
    }

    .online {
        padding: 10px 16px;
        border-radius: 10px;
        background: #dcfce7;
        color: #166534;
        font-weight: 700;
    }

    .idle {
        padding: 10px 16px;
        border-radius: 10px;
        background: #fef3c7;
        color: #92400e;
        font-weight: 700;
    }

    .offline {
        padding: 10px 16px;
        border-radius: 10px;
        background: #fee2e2;
        color: #991b1b;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def show_result(status, probability=None):

    if status == "FALL DETECTED":
        st.markdown(
            '<div class="status-fall">🚨 FALL DETECTED</div>',
            unsafe_allow_html=True,
        )

    elif status == "NORMAL":
        st.markdown(
            '<div class="status-safe">✓ NO FALL DETECTED</div>',
            unsafe_allow_html=True,
        )

    else:
        st.markdown(
            f'<div class="status-wait">{status}</div>',
            unsafe_allow_html=True,
        )

    if probability is not None:
        st.write(
            f"Fall probability: **{probability:.1%}**"
        )


# ============================================================
# TENSORFLOW
# ============================================================

@st.cache_resource
def get_tensorflow():

    try:
        import tensorflow as tf
        return tf

    except ModuleNotFoundError:
        return None


# ============================================================
# LOAD INDOOR MODELS
# ============================================================

@st.cache_resource
def load_vision_models():

    if YOLO is None:
        raise RuntimeError(
            "Ultralytics is not installed.\n\n"
            "Run:\n"
            "pip install ultralytics"
        )

    if YOLO_MODEL_PATH is None:
        raise FileNotFoundError(
            "YOLO11n-Pose model was not found.\n\n"
            "Expected:\n"
            f"{PROJECT_ROOT / 'yolo11n-pose.pt'}"
        )

    tf = get_tensorflow()

    if tf is None:
        raise RuntimeError(
            "TensorFlow is not installed in this Python environment.\n\n"
            "Install it using:\n"
            "pip install tensorflow"
        )

    if VISION_MODEL_PATH is None:
        raise FileNotFoundError(
            "Indoor GRU model was not found."
        )

    pose = YOLO(
        str(YOLO_MODEL_PATH)
    )

    gru = tf.keras.models.load_model(
        VISION_MODEL_PATH
    )

    return pose, gru


# ============================================================
# LOAD OUTDOOR MODEL
# ============================================================

@st.cache_resource
def load_outdoor_model():

    tf = get_tensorflow()

    if tf is None:
        raise RuntimeError(
            "TensorFlow is not installed in this environment."
        )

    if not OUTDOOR_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Outdoor model not found:\n"
            f"{OUTDOOR_MODEL_PATH}"
        )

    if not OUTDOOR_SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Outdoor scaler not found:\n"
            f"{OUTDOOR_SCALER_PATH}"
        )

    model = tf.keras.models.load_model(
        OUTDOOR_MODEL_PATH
    )

    scaler = np.load(
        OUTDOOR_SCALER_PATH
    )

    mean = scaler["mean"]
    std = scaler["std"]

    return model, mean, std


# ============================================================
# YOLO11 POSE PROCESSING
# ============================================================

def normalize_keypoints(keypoints):

    keypoints = np.asarray(
        keypoints,
        dtype=np.float32,
    )

    if keypoints.shape != (17, 2):
        return None

    # COCO keypoints:
    # 5 = left shoulder
    # 6 = right shoulder
    # 11 = left hip
    # 12 = right hip

    hip_center = (
        keypoints[11]
        + keypoints[12]
    ) / 2.0

    normalized = (
        keypoints - hip_center
    )

    shoulder_distance = np.linalg.norm(
        normalized[5]
        - normalized[6]
    )

    if shoulder_distance < 1e-6:
        return None

    normalized = (
        normalized / shoulder_distance
    )

    return normalized.astype(
        np.float32
    )


def select_person(result):

    if (
        result.keypoints is None
        or len(result.keypoints) == 0
    ):
        return None

    points = (
        result.keypoints.xy
        .cpu()
        .numpy()
    )

    if len(points) == 0:
        return None

    if result.boxes is not None:
        box_conf = (
            result.boxes.conf
            .cpu()
            .numpy()
        )
    else:
        box_conf = np.ones(
            len(points)
        )

    if result.keypoints.conf is not None:
        kp_conf = (
            result.keypoints.conf
            .cpu()
            .numpy()
        )
    else:
        kp_conf = np.ones(
            (len(points), 17)
        )

    candidates = []

    for i in range(len(points)):

        if box_conf[i] < 0.50:
            continue

        valid = np.sum(
            kp_conf[i] >= 0.40
        )

        if valid < 5:
            continue

        average = float(
            np.mean(kp_conf[i])
        )

        if average < 0.40:
            continue

        score = (
            0.6 * float(box_conf[i])
            + 0.4 * average
        )

        candidates.append(
            (score, i)
        )

    if not candidates:
        return None

    candidates.sort(
        reverse=True
    )

    return points[
        candidates[0][1]
    ]


# ============================================================
# INDOOR DETECTOR
# ============================================================

class VisionDetector:

    def __init__(
        self,
        pose,
        gru,
    ):

        self.pose = pose
        self.gru = gru

        self.sequence = deque(
            maxlen=VISION_SEQUENCE_LENGTH
        )

        self.fall_count = 0
        self.normal_count = 0

        self.probability = 0.0

        self.status = "WAITING"

    def reset(self):

        self.sequence.clear()

        self.fall_count = 0
        self.normal_count = 0

        self.probability = 0.0

        self.status = "WAITING"

    def process_frame(
        self,
        frame,
    ):

        results = self.pose(
            frame,
            conf=0.50,
            classes=[0],
            verbose=False,
        )

        result = results[0]

        person = select_person(
            result
        )

        output = result.plot()

        # ----------------------------------------------------
        # NO PERSON
        # ----------------------------------------------------

        if person is None:

            self.sequence.clear()

            self.fall_count = 0
            self.normal_count = 0

            self.status = "NO PERSON"

            return (
                output,
                self.status,
                self.probability,
            )

        # ----------------------------------------------------
        # NORMALIZE YOLO KEYPOINTS
        # ----------------------------------------------------

        normalized = normalize_keypoints(
            person
        )

        if normalized is None:

            self.status = "ANALYZING"

            return (
                output,
                self.status,
                self.probability,
            )

        self.sequence.append(
            normalized
        )

        # ----------------------------------------------------
        # WAIT FOR 20 FRAMES
        # ----------------------------------------------------

        if (
            len(self.sequence)
            < VISION_SEQUENCE_LENGTH
        ):

            self.status = (
                f"ANALYZING "
                f"{len(self.sequence)}/"
                f"{VISION_SEQUENCE_LENGTH}"
            )

            return (
                output,
                self.status,
                self.probability,
            )

        # ----------------------------------------------------
        # GRU INPUT
        # ----------------------------------------------------

        X = np.asarray(
            self.sequence,
            dtype=np.float32,
        ).reshape(
            1,
            VISION_SEQUENCE_LENGTH,
            34,
        )

        prediction = self.gru.predict(
            X,
            verbose=0,
        )

        probability = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        probability = float(
            np.clip(
                probability,
                0.0,
                1.0,
            )
        )

        self.probability = probability

        # ----------------------------------------------------
        # TEMPORAL CONFIRMATION
        # ----------------------------------------------------
        #
        # IMPORTANT:
        # One high-probability frame does NOT immediately
        # change the final result.
        #
        # 3 consecutive positive predictions -> FALL
        # 5 consecutive normal predictions -> NORMAL
        #
        # This prevents the UI from rapidly changing between
        # FALL and NORMAL.
        # ----------------------------------------------------

        if probability >= VISION_THRESHOLD:

            self.fall_count += 1
            self.normal_count = 0

        else:

            self.normal_count += 1
            self.fall_count = 0

        if (
            self.fall_count
            >= VISION_FALL_CONFIRMATIONS
        ):

            self.status = "FALL DETECTED"

        elif (
            self.normal_count
            >= VISION_NORMAL_CONFIRMATIONS
        ):

            self.status = "NORMAL"

        # Otherwise KEEP the previous confirmed status.
        # Do NOT flip the UI on every frame.

        return (
            output,
            self.status,
            self.probability,
        )


# ============================================================
# DRAW INDOOR RESULT
# ============================================================

def draw_vision_status(
    frame,
    status,
    probability,
):

    output = frame.copy()

    h, w = output.shape[:2]

    if status == "FALL DETECTED":

        cv2.rectangle(
            output,
            (10, 10),
            (w - 10, h - 10),
            (0, 0, 255),
            5,
        )

        cv2.rectangle(
            output,
            (20, 20),
            (390, 78),
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            output,
            "FALL DETECTED",
            (35, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            3,
        )

    elif status == "NORMAL":

        cv2.rectangle(
            output,
            (20, 20),
            (270, 72),
            (22, 163, 74),
            -1,
        )

        cv2.putText(
            output,
            "NO FALL DETECTED",
            (35, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
        )

    elif status.startswith("ANALYZING"):

        cv2.putText(
            output,
            status,
            (25, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

    cv2.putText(
        output,
        f"Fall probability: {probability:.1%}",
        (25, h - 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )

    return output


# ============================================================
# VIDEO ANALYSIS
# ============================================================

def process_video_file(
    uploaded_file,
):

    pose, gru = load_vision_models()

    detector = VisionDetector(
        pose,
        gru,
    )

    suffix = Path(
        uploaded_file.name
    ).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp:

        temp.write(
            uploaded_file.getbuffer()
        )

        video_path = temp.name

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        Path(video_path).unlink(
            missing_ok=True
        )

        raise RuntimeError(
            "Unable to open the uploaded video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 25.0

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    output_path = (
        Path(tempfile.gettempdir())
        / (
            f"fall_detection_"
            f"{int(time.time())}.mp4"
        )
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(
            *"mp4v"
        ),
        fps,
        (width, height),
    )

    preview = st.empty()
    progress = st.progress(0.0)
    result_placeholder = st.empty()

    final_fall = False
    max_probability = 0.0

    processed = 0

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            (
                output,
                status,
                probability,
            ) = detector.process_frame(
                frame
            )

            output = draw_vision_status(
                output,
                status,
                probability,
            )

            writer.write(
                output
            )

            max_probability = max(
                max_probability,
                probability,
            )

            # ONLY confirmed fall counts.
            if status == "FALL DETECTED":
                final_fall = True

            if processed % 3 == 0:

                preview.image(
                    cv2.cvtColor(
                        output,
                        cv2.COLOR_BGR2RGB,
                    ),
                    channels="RGB",
                    use_container_width=True,
                )

            processed += 1

            if total_frames > 0:

                progress.progress(
                    min(
                        processed
                        / total_frames,
                        1.0,
                    )
                )

        progress.progress(1.0)

    finally:

        cap.release()
        writer.release()

        Path(video_path).unlink(
            missing_ok=True
        )

    if final_fall:

        result_placeholder.error(
            f"🚨 FALL DETECTED\n\n"
            f"Maximum probability: "
            f"{max_probability:.1%}"
        )

    else:

        result_placeholder.success(
            "✓ NO FALL DETECTED"
        )

    st.video(
        str(output_path)
    )

    with open(
        output_path,
        "rb",
    ) as f:

        st.download_button(
            "Download analysed video",
            data=f,
            file_name="fall_detection_result.mp4",
            mime="video/mp4",
        )

    return (
        final_fall,
        max_probability,
    )


# ============================================================
# LIVE CAMERA
# ============================================================

def get_live_detector():

    pose, gru = load_vision_models()

    if (
        "vision_detector"
        not in st.session_state
    ):

        st.session_state[
            "vision_detector"
        ] = VisionDetector(
            pose,
            gru,
        )

    return st.session_state[
        "vision_detector"
    ]


# ============================================================
# PHONE SENSOR STATE
# ============================================================

phone_lock = threading.Lock()

phone_buffer = deque(
    maxlen=OUTDOOR_SEQUENCE_LENGTH
)

phone_probability = 0.0

phone_status = (
    "WAITING FOR PHONE"
)

phone_last_received = 0.0


# ============================================================
# SENSOR LOGGER EXTRACTION
# ============================================================

def extract_phone_samples(
    payload,
):

    if not isinstance(
        payload,
        list,
    ):
        return []

    grouped = {}

    for item in payload:

        if not isinstance(
            item,
            dict,
        ):
            continue

        name = str(
            item.get(
                "name",
                "",
            )
        ).lower()

        values = item.get(
            "values",
            {},
        )

        timestamp = item.get(
            "time"
        )

        if not isinstance(
            values,
            dict,
        ):
            continue

        if timestamp is None:
            continue

        if name not in {
            "accelerometer",
            "gravity",
            "gyroscope",
        }:
            continue

        try:

            x = float(
                values["x"]
            )

            y = float(
                values["y"]
            )

            z = float(
                values["z"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if timestamp not in grouped:

            grouped[timestamp] = {}

        if name == "accelerometer":

            grouped[timestamp][
                "accelerometer"
            ] = [
                x,
                y,
                z,
            ]

        elif name == "gravity":

            grouped[timestamp][
                "gravity"
            ] = [
                x,
                y,
                z,
            ]

        elif name == "gyroscope":

            grouped[timestamp][
                "gyroscope"
            ] = [
                x,
                y,
                z,
            ]

    samples = []

    required = {
        "accelerometer",
        "gravity",
        "gyroscope",
    }

    for timestamp in sorted(
        grouped
    ):

        item = grouped[
            timestamp
        ]

        if not required.issubset(
            item
        ):
            continue

        samples.append(
            item["accelerometer"]
            + item["gravity"]
            + item["gyroscope"]
        )

    return samples


# ============================================================
# OUTDOOR PREDICTION
# ============================================================

def predict_outdoor_window(
    data,
):

    model, mean, std = (
        load_outdoor_model()
    )

    data = np.asarray(
        data,
        dtype=np.float32,
    )

    if data.shape != (
        OUTDOOR_SEQUENCE_LENGTH,
        9,
    ):
        raise ValueError(
            "Outdoor input must have "
            f"shape "
            f"({OUTDOOR_SEQUENCE_LENGTH}, 9)."
        )

    safe_std = np.where(
        np.abs(std) < 1e-8,
        1.0,
        std,
    )

    X = (
        data - mean
    ) / safe_std

    prediction = model.predict(
        X[np.newaxis, ...],
        verbose=0,
    )

    probability = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )

    return float(
        np.clip(
            probability,
            0.0,
            1.0,
        )
    )


# ============================================================
# PROCESS PHONE DATA
# ============================================================

def process_phone_payload(
    payload,
):

    global phone_probability
    global phone_status
    global phone_last_received

    samples = extract_phone_samples(
        payload
    )

    if not samples:
        return

    with phone_lock:

        for sample in samples:

            phone_buffer.append(
                np.asarray(
                    sample,
                    dtype=np.float32,
                )
            )

        phone_last_received = (
            time.time()
        )

        count = len(
            phone_buffer
        )

        if count < OUTDOOR_SEQUENCE_LENGTH:

            phone_status = (
                f"COLLECTING "
                f"{count}/"
                f"{OUTDOOR_SEQUENCE_LENGTH}"
            )

            return

        X = np.asarray(
            phone_buffer,
            dtype=np.float32,
        )

    try:

        probability = (
            predict_outdoor_window(
                X
            )
        )

        with phone_lock:

            phone_probability = (
                probability
            )

            if (
                probability
                >= OUTDOOR_THRESHOLD
            ):

                phone_status = (
                    "FALL DETECTED"
                )

            else:

                phone_status = (
                    "NORMAL"
                )

    except Exception as error:

        with phone_lock:

            phone_status = (
                f"ERROR: {error}"
            )


# ============================================================
# PHONE HTTP SERVER
# ============================================================

class PhoneHandler(
    BaseHTTPRequestHandler
):

    def do_POST(self):

        try:

            length = int(
                self.headers.get(
                    "Content-Length",
                    0,
                )
            )

            body = self.rfile.read(
                length
            )

            message = json.loads(
                body.decode(
                    "utf-8",
                    errors="replace",
                )
            )

            payload = message.get(
                "payload",
                message
            )

            process_phone_payload(
                payload
            )

            response = json.dumps(
                {
                    "status": "received",
                    "samples": len(
                        phone_buffer
                    ),
                    "connected": True,
                }
            ).encode(
                "utf-8"
            )

            self.send_response(
                200
            )

            self.send_header(
                "Content-Type",
                "application/json",
            )

            self.send_header(
                "Access-Control-Allow-Origin",
                "*",
            )

            self.end_headers()

            self.wfile.write(
                response
            )

        except (
            ConnectionResetError,
            BrokenPipeError,
        ):

            return

        except Exception as error:

            response = json.dumps(
                {
                    "error": str(error)
                }
            ).encode(
                "utf-8"
            )

            try:

                self.send_response(
                    400
                )

                self.send_header(
                    "Content-Type",
                    "application/json",
                )

                self.end_headers()

                self.wfile.write(
                    response
                )

            except Exception:
                pass

    def do_GET(self):

        response = json.dumps(
            {
                "status": "running"
            }
        ).encode(
            "utf-8"
        )

        self.send_response(
            200
        )

        self.send_header(
            "Content-Type",
            "application/json",
        )

        self.end_headers()

        self.wfile.write(
            response
        )

    def log_message(
        self,
        format,
        *args,
    ):
        return


@st.cache_resource
def start_phone_server():

    server = HTTPServer(
        (
            "0.0.0.0",
            PHONE_PORT,
        ),
        PhoneHandler,
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    thread.start()

    return server


# ============================================================
# PHONE CONNECTION STATUS
# ============================================================

def get_phone_connection_status():

    with phone_lock:

        last = (
            phone_last_received
        )

    if last == 0:

        return "OFFLINE"

    elapsed = (
        time.time()
        - last
    )

    if elapsed <= PHONE_ONLINE_SECONDS:

        return "ONLINE"

    if elapsed <= PHONE_IDLE_SECONDS:

        return "IDLE"

    return "OFFLINE"


def show_phone_connection():

    state = (
        get_phone_connection_status()
    )

    if state == "ONLINE":

        st.markdown(
            '<div class="online">'
            "🟢 ONLINE — Phone sensor stream active"
            "</div>",
            unsafe_allow_html=True,
        )

    elif state == "IDLE":

        st.markdown(
            '<div class="idle">'
            "🟡 IDLE — No recent sensor data"
            "</div>",
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div class="offline">'
            "🔴 OFFLINE — Phone sensor stream not connected"
            "</div>",
            unsafe_allow_html=True,
        )

    return state


# ============================================================
# SENSOR FILE
# ============================================================

def parse_sensor_file(
    uploaded_file,
):

    raw = uploaded_file.getvalue()

    text = raw.decode(
        "utf-8",
        errors="ignore",
    )

    rows = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        line = line.rstrip(";")

        parts = line.split(",")

        values = []

        for part in parts:

            try:

                values.append(
                    float(
                        part.strip()
                    )
                )

            except ValueError:

                continue

        if len(values) >= 9:

            rows.append(
                values[:9]
            )

    if not rows:

        raise ValueError(
            "No valid 9-channel sensor "
            "samples were found."
        )

    return np.asarray(
        rows,
        dtype=np.float32,
    )


def predict_sensor_file(
    uploaded_file,
):

    data = parse_sensor_file(
        uploaded_file
    )

    if (
        len(data)
        < OUTDOOR_SEQUENCE_LENGTH
    ):

        raise ValueError(
            f"Only {len(data)} samples found. "
            f"{OUTDOOR_SEQUENCE_LENGTH} samples "
            "are required."
        )

    model, mean, std = (
        load_outdoor_model()
    )

    safe_std = np.where(
        np.abs(std) < 1e-8,
        1.0,
        std,
    )

    windows = []

    for start in range(
        0,
        len(data)
        - OUTDOOR_SEQUENCE_LENGTH
        + 1,
        100,
    ):

        windows.append(
            data[
                start:
                start
                + OUTDOOR_SEQUENCE_LENGTH
            ]
        )

    # Always include the final 200 samples.
    final_window = data[
        -OUTDOOR_SEQUENCE_LENGTH:
    ]

    if not windows or not np.array_equal(
        windows[-1],
        final_window,
    ):

        windows.append(
            final_window
        )

    X = np.asarray(
        windows,
        dtype=np.float32,
    )

    X = (
        X - mean
    ) / safe_std

    probabilities = (
        model.predict(
            X,
            verbose=0,
        )
        .ravel()
    )

    probabilities = np.clip(
        probabilities,
        0.0,
        1.0,
    )

    return (
        data,
        probabilities,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "Smart Fall Detection"
)

page = st.sidebar.radio(
    "Detection",
    [
        "Home",
        "Indoor",
        "Outdoor",
        "Automatic",
    ],
)


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">
                Smart Fall Detection System
            </div>

            <div class="hero-title">
                Detect falls. Respond faster.
            </div>

            <div class="hero-text">
                A multimodal fall-detection system designed
                for different environments. Indoor monitoring
                uses YOLO11n-Pose with a GRU classifier,
                while outdoor monitoring uses phone-based
                9-channel motion sensing with an LSTM.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader(
            "🏠 Indoor Vision"
        )

        st.caption(
            "YOLO11n-Pose extracts 17 body "
            "keypoints from live camera or "
            "recorded video."
        )

    with c2:

        st.subheader(
            "📱 Outdoor Sensing"
        )

        st.caption(
            "Accelerometer, gravity and "
            "gyroscope data are analyzed "
            "using the outdoor LSTM."
        )

    with c3:

        st.subheader(
            "🚨 Fall Alert"
        )

        st.caption(
            "Confirmed fall predictions are "
            "clearly displayed for immediate attention."
        )


# ============================================================
# INDOOR
# ============================================================

elif page == "Indoor":

    st.title(
        "Indoor Fall Detection"
    )

    indoor_mode = st.radio(
        "Input",
        [
            "Live Camera",
            "Upload Video / Dataset",
        ],
        horizontal=True,
    )

    # --------------------------------------------------------
    # LIVE CAMERA
    # --------------------------------------------------------

    if indoor_mode == "Live Camera":

        if not WEBRTC_OK:

            st.error(
                "streamlit-webrtc is not installed."
            )

            st.code(
                "pip install streamlit-webrtc"
            )

            st.stop()

        try:

            detector = (
                get_live_detector()
            )

        except Exception as error:

            st.error(
                str(error)
            )

            st.stop()

        class LiveVideoProcessor(
            VideoProcessorBase
        ):

            def __init__(self):

                self.detector = (
                    detector
                )

            def recv(
                self,
                frame,
            ):

                image = frame.to_ndarray(
                    format="bgr24"
                )

                (
                    output,
                    status,
                    probability,
                ) = (
                    self.detector
                    .process_frame(
                        image
                    )
                )

                output = (
                    draw_vision_status(
                        output,
                        status,
                        probability,
                    )
                )

                return (
                    frame.from_ndarray(
                        output,
                        format="bgr24",
                    )
                )

        st.info(
            "Start the camera once. "
            "Detection continues automatically."
        )

        webrtc_streamer(
            key="indoor-yolo11-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=(
                LiveVideoProcessor
            ),
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            async_processing=True,
        )

        if detector.status == "FALL DETECTED":

            st.error(
                "🚨 FALL DETECTED"
            )

        elif detector.status == "NORMAL":

            st.success(
                "✓ NO FALL DETECTED"
            )

        else:

            st.info(
                detector.status
            )

        st.write(
            f"Fall probability: "
            f"**{detector.probability:.1%}**"
        )

    # --------------------------------------------------------
    # UPLOAD VIDEO
    # --------------------------------------------------------

    else:

        uploaded_video = (
            st.file_uploader(
                "Upload a video",
                type=[
                    "mp4",
                    "avi",
                    "mov",
                    "mkv",
                ],
            )
        )

        if uploaded_video is not None:

            if st.button(
                "Analyze Video",
                type="primary",
                use_container_width=True,
            ):

                with st.spinner(
                    "Analyzing video with YOLO11n-Pose + GRU..."
                ):

                    try:

                        process_video_file(
                            uploaded_video
                        )

                    except Exception as error:

                        st.error(
                            f"Video analysis failed: "
                            f"{error}"
                        )


# ============================================================
# OUTDOOR
# ============================================================

elif page == "Outdoor":

    st.title(
        "Outdoor Fall Detection"
    )

    outdoor_mode = st.radio(
        "Input",
        [
            "Live Phone",
            "Upload Sensor Data",
        ],
        horizontal=True,
    )

    # --------------------------------------------------------
    # LIVE PHONE
    # --------------------------------------------------------

    if outdoor_mode == "Live Phone":

        start_phone_server()

        show_phone_connection()

        with phone_lock:

            status = phone_status

            probability = (
                phone_probability
            )

            samples = len(
                phone_buffer
            )

        show_result(
            status,
            probability,
        )

        st.caption(
            f"Live sensor samples: "
            f"{samples}/"
            f"{OUTDOOR_SEQUENCE_LENGTH}"
        )

        st.caption(
            "Sensor Logger must send POST data "
            "to this computer on port 8000."
        )

        # Refresh only while live phone mode is open.
        time.sleep(1)

        st.rerun()

    # --------------------------------------------------------
    # UPLOAD SENSOR DATA
    # --------------------------------------------------------

    else:

        st.write(
            "Upload recorded 9-channel phone "
            "sensor data."
        )

        sensor_files = (
            st.file_uploader(
                "Sensor dataset",
                type=[
                    "txt",
                    "csv",
                ],
                accept_multiple_files=True,
            )
        )

        if sensor_files:

            if st.button(
                "Analyze Sensor Data",
                type="primary",
                use_container_width=True,
            ):

                for file in sensor_files:

                    try:

                        with st.spinner(
                            f"Analyzing {file.name}..."
                        ):

                            (
                                data,
                                probabilities,
                            ) = (
                                predict_sensor_file(
                                    file
                                )
                            )

                        maximum = float(
                            np.max(
                                probabilities
                            )
                        )

                        fall = bool(
                            np.any(
                                probabilities
                                >= OUTDOOR_THRESHOLD
                            )
                        )

                        st.subheader(
                            file.name
                        )

                        st.caption(
                            f"{len(data)} samples"
                        )

                        show_result(
                            "FALL DETECTED"
                            if fall
                            else "NORMAL",
                            maximum,
                        )

                    except Exception as error:

                        st.error(
                            f"{file.name}: "
                            f"{error}"
                        )


# ============================================================
# AUTOMATIC
# ============================================================

elif page == "Automatic":

    st.title(
        "Automatic Fall Detection"
    )

    st.write(
        "Upload video or sensor data. "
        "The application automatically selects "
        "the appropriate detection branch."
    )

    uploaded_files = (
        st.file_uploader(
            "Upload video or sensor data",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv",
                "txt",
                "csv",
            ],
            accept_multiple_files=True,
        )
    )

    if uploaded_files:

        videos = []
        sensors = []

        for file in uploaded_files:

            extension = (
                Path(file.name)
                .suffix
                .lower()
            )

            if extension in {
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
            }:

                videos.append(file)

            elif extension in {
                ".txt",
                ".csv",
            }:

                sensors.append(file)

        # ----------------------------------------------------
        # VIDEOS
        # ----------------------------------------------------

        for file in videos:

            st.subheader(
                f"Indoor: {file.name}"
            )

            if st.button(
                f"Analyze {file.name}",
                key=f"video_{file.name}",
                type="primary",
            ):

                try:

                    with st.spinner(
                        "YOLO11n-Pose + GRU analysis..."
                    ):

                        process_video_file(
                            file
                        )

                except Exception as error:

                    st.error(
                        f"{file.name}: "
                        f"{error}"
                    )

        # ----------------------------------------------------
        # SENSOR DATA
        # ----------------------------------------------------

        for file in sensors:

            st.subheader(
                f"Outdoor: {file.name}"
            )

            if st.button(
                f"Analyze {file.name}",
                key=f"sensor_{file.name}",
                type="primary",
            ):

                try:

                    with st.spinner(
                        "Outdoor LSTM analysis..."
                    ):

                        (
                            data,
                            probabilities,
                        ) = (
                            predict_sensor_file(
                                file
                            )
                        )

                    maximum = float(
                        np.max(
                            probabilities
                        )
                    )

                    fall = bool(
                        np.any(
                            probabilities
                            >= OUTDOOR_THRESHOLD
                        )
                    )

                    show_result(
                        "FALL DETECTED"
                        if fall
                        else "NORMAL",
                        maximum,
                    )

                    st.caption(
                        f"{len(data)} samples"
                    )

                except Exception as error:

                    st.error(
                        f"{file.name}: "
                        f"{error}"
                    )