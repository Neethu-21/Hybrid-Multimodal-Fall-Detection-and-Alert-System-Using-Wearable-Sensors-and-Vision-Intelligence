import streamlit as st
import numpy as np
import cv2
import tempfile
import time
import json
import threading
import csv
from pathlib import Path
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import tensorflow as tf
from ultralytics import YOLO

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

GRU_MODEL = PROJECT_ROOT / "models" / "vision" / "best_gru.keras"
YOLO_MODEL = PROJECT_ROOT / "models" / "vision" / "yolo11n-pose.pt"

# If YOLO file is stored in project root, use it.
if not YOLO_MODEL.exists():
    root_yolo = PROJECT_ROOT / "yolo11n-pose.pt"
    if root_yolo.exists():
        YOLO_MODEL = root_yolo

OUTDOOR_MODEL = PROJECT_ROOT / "models" / "outdoor" / "best_outdoor_lstm.keras"
OUTDOOR_SCALER = PROJECT_ROOT / "models" / "outdoor" / "outdoor_lstm_scaler.npz"

PHONE_PORT = 8000

VISION_SEQUENCE_LENGTH = 20
VISION_FALL_THRESHOLD = 0.35
VISION_CONFIRM_FRAMES = 3
YOLO_CONFIDENCE = 0.50
KEYPOINT_DRAW_CONFIDENCE = 0.20
MAX_MISSED_FRAMES = 8
PROBABILITY_SMOOTHING_WINDOW = 3
FALL_ALERT_HOLD_FRAMES = 90

OUTDOOR_SEQUENCE_LENGTH = 200
OUTDOOR_THRESHOLD = 0.40

# ============================================================
# UI
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.0rem;
        max-width: 1250px;
    }

    .hero {
        padding: 28px 30px;
        border-radius: 14px;
        background: #ffffff;
        border: 1px solid #e3e6eb;
        margin-bottom: 20px;
    }

    .hero-kicker {
        color: #2850e8;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .hero-title {
        color: #111827;
        font-size: 32px;
        font-weight: 800;
        line-height: 1.2;
        margin-bottom: 8px;
    }

    .hero-text {
        color: #5b6472;
        font-size: 14px;
        line-height: 1.6;
        max-width: 850px;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e3e6eb;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 18px;
    }

    .result {
        padding: 16px 20px;
        border-radius: 9px;
        border: 1px solid #e3e6eb;
        background: #ffffff;
        font-weight: 700;
        font-size: 16px;
    }

    .result-fall {
        border-left: 5px solid #b42318;
        background: #fef3f2;
        color: #b42318;
    }

    .result-safe {
        border-left: 5px solid #067647;
        background: #ecfdf3;
        color: #067647;
    }

    .result-wait {
        border-left: 5px solid #98a2b3;
        background: #f2f4f7;
        color: #475467;
    }

    .status-online {
        color: #067647;
        font-weight: 700;
    }

    .status-offline {
        color: #b42318;
        font-weight: 700;
    }

    .status-idle {
        color: #b54708;
        font-weight: 700;
    }

    .sidebar-status {
        font-size: 12px;
        padding: 5px 0;
    }

    .dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 7px;
    }

    .green { background: #12b76a; }
    .red { background: #f04438; }
    .amber { background: #f79009; }
    .gray { background: #98a2b3; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# RESULT UI
# ============================================================

def show_result(status, probability=None):
    if status == "FALL DETECTED":
        css = "result result-fall"
        text = "🚨 FALL DETECTED"
    elif status == "NORMAL":
        css = "result result-safe"
        text = "✓ NO FALL DETECTED"
    else:
        css = "result result-wait"
        text = status

    st.markdown(
        f'<div class="{css}">{text}</div>',
        unsafe_allow_html=True,
    )

    if probability is not None:
        st.write(f"Fall probability: **{probability:.1%}**")


# ============================================================
# MODEL LOADING
# IMPORTANT: INDOOR = YOLO11n-POSE + GRU
# ============================================================

@st.cache_resource
def load_vision_models():
    if not YOLO_MODEL.exists():
        raise FileNotFoundError(
            f"YOLO11 pose model not found:\n{YOLO_MODEL}"
        )

    if not GRU_MODEL.exists():
        raise FileNotFoundError(
            f"GRU model not found:\n{GRU_MODEL}"
        )

    pose_model = YOLO(str(YOLO_MODEL))
    gru_model = tf.keras.models.load_model(str(GRU_MODEL))

    return pose_model, gru_model


@st.cache_resource
def load_outdoor_model():
    if not OUTDOOR_MODEL.exists():
        raise FileNotFoundError(
            f"Outdoor LSTM model not found:\n{OUTDOOR_MODEL}"
        )

    if not OUTDOOR_SCALER.exists():
        raise FileNotFoundError(
            f"Outdoor scaler not found:\n{OUTDOOR_SCALER}"
        )

    model = tf.keras.models.load_model(str(OUTDOOR_MODEL))

    scaler = np.load(str(OUTDOOR_SCALER))
    mean = scaler["mean"].astype(np.float32)
    std = scaler["std"].astype(np.float32)
    safe_std = np.where(np.abs(std) < 1e-8, 1.0, std)

    return model, mean, safe_std


# ============================================================
# YOLO11 POSE
# ============================================================

SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]


def normalize_keypoints(keypoints):
    keypoints = np.asarray(keypoints, dtype=np.float32)

    left_hip = keypoints[11]
    right_hip = keypoints[12]

    center = (left_hip + right_hip) / 2.0
    normalized = keypoints - center

    left_shoulder = normalized[5]
    right_shoulder = normalized[6]

    shoulder_distance = np.linalg.norm(
        left_shoulder - right_shoulder
    )

    if not np.isfinite(shoulder_distance) or shoulder_distance < 1e-6:
        shoulder_distance = 1.0

    normalized = normalized / shoulder_distance

    return np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def select_person(result):
    if result.keypoints is None or len(result.keypoints) == 0:
        return None

    keypoints = (
        result.keypoints.xy
        .detach()
        .cpu()
        .numpy()
    )

    if len(keypoints) == 0:
        return None

    if result.boxes is None or len(result.boxes) == 0:
        return None

    box_conf = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    if len(box_conf) == 0:
        return None

    best_index = int(np.argmax(box_conf))
    detection_confidence = float(box_conf[best_index])

    if detection_confidence < YOLO_CONFIDENCE:
        return None

    person = keypoints[best_index].astype(np.float32)

    if person.shape != (17, 2):
        return None

    if result.keypoints.conf is not None:
        kp_conf = (
            result.keypoints.conf
            .detach()
            .cpu()
            .numpy()
        )
        person_kp_conf = kp_conf[best_index].astype(np.float32)
    else:
        person_kp_conf = np.ones(17, dtype=np.float32)

    bbox = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()[best_index]
        .astype(np.float32)
    )

    return person, person_kp_conf, bbox, detection_confidence


def predict_fall(sequence, gru_model):
    sequence = np.asarray(sequence, dtype=np.float32)
    sequence = sequence.reshape(
        1,
        VISION_SEQUENCE_LENGTH,
        34,
    )

    probability = float(
        gru_model.predict(sequence, verbose=0)[0][0]
    )

    return float(np.clip(probability, 0.0, 1.0))


def draw_person(
    frame,
    keypoints,
    keypoint_confidence,
    bbox,
    detection_confidence,
):
    output = frame.copy()

    h, w = output.shape[:2]

    x1, y1, x2, y2 = bbox.astype(int)

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2,
    )

    cv2.putText(
        output,
        f"Person {detection_confidence:.2f}",
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for start, end in SKELETON_CONNECTIONS:
        if (
            keypoint_confidence[start] < KEYPOINT_DRAW_CONFIDENCE
            or keypoint_confidence[end] < KEYPOINT_DRAW_CONFIDENCE
        ):
            continue

        p1 = tuple(keypoints[start].astype(int))
        p2 = tuple(keypoints[end].astype(int))

        cv2.line(
            output,
            p1,
            p2,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

    for i in range(17):
        if keypoint_confidence[i] < KEYPOINT_DRAW_CONFIDENCE:
            continue

        x, y = keypoints[i].astype(int)

        if 0 <= x < w and 0 <= y < h:
            cv2.circle(
                output,
                (x, y),
                4,
                (0, 255, 0),
                -1,
                cv2.LINE_AA,
            )

    return output


class VisionFallDetector:

    def __init__(self, pose_model, gru_model):
        self.pose_model = pose_model
        self.gru_model = gru_model

        self.sequence = deque(maxlen=VISION_SEQUENCE_LENGTH)
        self.probabilities = deque(
            maxlen=PROBABILITY_SMOOTHING_WINDOW
        )

        self.positive_count = 0
        self.fall_detected = False
        self.fall_hold = 0

        self.missed_frames = 0
        self.last_probability = None

    def reset(self):
        self.sequence.clear()
        self.probabilities.clear()
        self.positive_count = 0
        self.fall_detected = False
        self.fall_hold = 0
        self.missed_frames = 0
        self.last_probability = None

    def predict_frame(self, frame):

        results = self.pose_model.predict(
            frame,
            conf=YOLO_CONFIDENCE,
            classes=[0],
            verbose=False,
        )

        result = results[0]
        detection = select_person(result)

        person_found = detection is not None

        if person_found:

            (
                person,
                keypoint_confidence,
                bbox,
                detection_confidence,
            ) = detection

            self.missed_frames = 0

            normalized = normalize_keypoints(person)

            self.sequence.append(
                normalized.reshape(-1)
            )

            if len(self.sequence) == VISION_SEQUENCE_LENGTH:

                probability = predict_fall(
                    self.sequence,
                    self.gru_model,
                )

                self.probabilities.append(probability)

                smoothed = float(
                    np.median(self.probabilities)
                )

                self.last_probability = smoothed

                if smoothed >= VISION_FALL_THRESHOLD:
                    self.positive_count += 1
                else:
                    self.positive_count = 0

                if self.positive_count >= VISION_CONFIRM_FRAMES:
                    self.fall_detected = True
                    self.fall_hold = FALL_ALERT_HOLD_FRAMES

        else:

            self.missed_frames += 1

            if self.missed_frames > MAX_MISSED_FRAMES:
                self.sequence.clear()
                self.probabilities.clear()
                self.positive_count = 0
                self.last_probability = None

        if self.fall_detected:

            if self.fall_hold > 0:
                self.fall_hold -= 1
            else:
                self.fall_detected = False
                self.positive_count = 0
                self.probabilities.clear()

        output = frame.copy()

        if person_found:
            output = draw_person(
                output,
                person,
                keypoint_confidence,
                bbox,
                detection_confidence,
            )

        if self.fall_detected:
            status = "FALL DETECTED"
            status_color = (0, 0, 255)

            cv2.rectangle(
                output,
                (10, 10),
                (output.shape[1] - 10, output.shape[0] - 10),
                status_color,
                5,
            )

        elif person_found:
            status = "NORMAL"
            status_color = (0, 255, 0)

        elif self.missed_frames <= MAX_MISSED_FRAMES:
            status = "TRACKING..."
            status_color = (0, 255, 255)

        else:
            status = "NO PERSON"
            status_color = (0, 165, 255)

        cv2.putText(
            output,
            status,
            (30, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            status_color,
            3,
            cv2.LINE_AA,
        )

        probability_text = (
            "Fall probability: --"
            if self.last_probability is None
            else f"Fall probability: {self.last_probability:.2f}"
        )

        cv2.putText(
            output,
            probability_text,
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            output,
            f"Pose buffer: {len(self.sequence)}/{VISION_SEQUENCE_LENGTH}",
            (30, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return output, status, self.last_probability


# ============================================================
# UPLOADED VIDEO
# ============================================================

def analyse_uploaded_video(uploaded_file):

    pose_model, gru_model = load_vision_models()

    detector = VisionFallDetector(
        pose_model,
        gru_model,
    )

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp:
        temp.write(uploaded_file.getbuffer())
        input_path = Path(temp.name)

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open uploaded video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 25.0

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    output_path = (
        Path(tempfile.gettempdir())
        / f"analysed_fall_{int(time.time())}.mp4"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    preview = st.empty()
    progress = st.progress(0.0)

    final_status = "NORMAL"
    max_probability = 0.0
    frame_no = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        output, status, probability = detector.predict_frame(
            frame
        )

        writer.write(output)

        if status == "FALL DETECTED":
            final_status = "FALL DETECTED"

        if probability is not None:
            max_probability = max(
                max_probability,
                probability,
            )

        if frame_no % 3 == 0:
            preview.image(
                cv2.cvtColor(
                    output,
                    cv2.COLOR_BGR2RGB,
                ),
                channels="RGB",
                use_container_width=True,
            )

        frame_no += 1

        if total > 0:
            progress.progress(
                min(frame_no / total, 1.0)
            )

    cap.release()
    writer.release()

    try:
        input_path.unlink()
    except Exception:
        pass

    progress.progress(1.0)

    return (
        output_path,
        final_status,
        max_probability,
    )


# ============================================================
# LIVE CAMERA
# ============================================================

try:
    from streamlit_webrtc import (
        webrtc_streamer,
        WebRtcMode,
        VideoProcessorBase,
    )
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


def create_live_camera_processor():

    pose_model, gru_model = load_vision_models()

    class LiveProcessor(VideoProcessorBase):

        def __init__(self):
            self.detector = VisionFallDetector(
                pose_model,
                gru_model,
            )

        def recv(self, frame):

            image = frame.to_ndarray(
                format="bgr24"
            )

            output, _, _ = (
                self.detector.predict_frame(
                    image
                )
            )

            return frame.from_ndarray(
                output,
                format="bgr24",
            )

    return LiveProcessor


# ============================================================
# OUTDOOR LSTM
# ============================================================

def predict_sensor_array(data):

    model, mean, safe_std = load_outdoor_model()

    data = np.asarray(
        data,
        dtype=np.float32,
    )

    # Input MUST be (samples, 9)
    if data.ndim != 2 or data.shape[1] != 9:
        raise ValueError(
            f"Expected sensor data shape (samples, 9), "
            f"got {data.shape}"
        )

    if len(data) < OUTDOOR_SEQUENCE_LENGTH:
        raise ValueError(
            f"Only {len(data)} samples found. "
            f"{OUTDOOR_SEQUENCE_LENGTH} required."
        )

    probabilities = []

    # Sliding windows of exactly 200 × 9
    for end in range(
        OUTDOOR_SEQUENCE_LENGTH,
        len(data) + 1,
    ):

        window = data[
            end - OUTDOOR_SEQUENCE_LENGTH:end,
            :
        ]

        # Explicitly verify window
        if window.shape != (
            OUTDOOR_SEQUENCE_LENGTH,
            9,
        ):
            raise ValueError(
                f"Invalid window shape: {window.shape}"
            )

        X = (
            window - mean
        ) / safe_std

        # EXACTLY (1, 200, 9)
        X = X.reshape(
            1,
            OUTDOOR_SEQUENCE_LENGTH,
            9,
        )

        probability = float(
            model.predict(
                X,
                verbose=0,
            ).reshape(-1)[0]
        )

        probabilities.append(
            np.clip(
                probability,
                0.0,
                1.0,
            )
        )

    return np.asarray(
        probabilities,
        dtype=np.float32,
    )


def parse_sensor_file(uploaded_file):

    text = uploaded_file.getvalue().decode(
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

        numeric = []

        for part in parts:

            try:
                numeric.append(
                    float(part.strip())
                )
            except ValueError:
                pass

        if len(numeric) >= 9:
            rows.append(
                numeric[-9:]
            )

    if not rows:
        raise ValueError(
            "No valid 9-channel sensor samples found."
        )

    return np.asarray(
        rows,
        dtype=np.float32,
    )


def predict_sensor_file(uploaded_file):

    data = parse_sensor_file(
        uploaded_file
    )

    probabilities = predict_sensor_array(
        data
    )

    return data, probabilities


# ============================================================
# SENSOR LOGGER PAYLOAD PARSING
# IMPORTANT:
# Sensor Logger sends accelerometer, gravity and gyroscope
# readings as separate objects. They may NOT be one object.
# This code groups readings by timestamp.
# ============================================================

def _time_number(value):

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_sensor_samples(payload):

    if isinstance(payload, dict):
        payload = [payload]

    if not isinstance(payload, list):
        return []

    events = {
        "accelerometer": [],
        "gravity": [],
        "gyroscope": [],
    }

    for item in payload:

        if not isinstance(item, dict):
            continue

        name = str(
            item.get("name", "")
        ).strip().lower()

        if name not in events:
            continue

        values = item.get(
            "values",
            {},
        )

        if not isinstance(values, dict):
            continue

        try:
            vector = [
                float(values["x"]),
                float(values["y"]),
                float(values["z"]),
            ]
        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        timestamp = item.get("time")

        events[name].append(
            (timestamp, vector)
        )

    if not all(events.values()):
        return []

    # First try exact timestamp matching.
    grouped = {}

    for sensor_name, readings in events.items():

        for timestamp, vector in readings:

            key = str(timestamp)

            if key not in grouped:
                grouped[key] = {}

            grouped[key][sensor_name] = vector

    samples = []

    for key in sorted(grouped):

        group = grouped[key]

        if all(
            sensor in group
            for sensor in (
                "accelerometer",
                "gravity",
                "gyroscope",
            )
        ):
            samples.append(
                group["accelerometer"]
                + group["gravity"]
                + group["gyroscope"]
            )

    if samples:
        return samples

    # Fallback: pair readings by order.
    count = min(
        len(events["accelerometer"]),
        len(events["gravity"]),
        len(events["gyroscope"]),
    )

    for i in range(count):

        samples.append(
            events["accelerometer"][i][1]
            + events["gravity"][i][1]
            + events["gyroscope"][i][1]
        )

    return samples


# ============================================================
# LIVE PHONE SERVER
# ============================================================

@st.cache_resource
def get_phone_state():

    return {
        "lock": threading.Lock(),
        "buffer": deque(
            maxlen=OUTDOOR_SEQUENCE_LENGTH
        ),
        "probability": 0.0,
        "status": "IDLE",
        "last_received": 0.0,
        "request_count": 0,
        "server_error": None,
    }


phone_state = get_phone_state()


def process_phone_payload(payload):

    samples = extract_sensor_samples(
        payload
    )

    if not samples:
        return 0

    with phone_state["lock"]:

        for sample in samples:

            phone_state["buffer"].append(
                np.asarray(
                    sample,
                    dtype=np.float32,
                )
            )

        phone_state["last_received"] = time.time()
        phone_state["request_count"] += 1

        count = len(phone_state["buffer"])

    # Do not run inference until 200 samples exist.
    if count < OUTDOOR_SEQUENCE_LENGTH:

        with phone_state["lock"]:
            phone_state["status"] = (
                f"COLLECTING {count}/{OUTDOOR_SEQUENCE_LENGTH}"
            )

        return len(samples)

    try:

        with phone_state["lock"]:
            X = np.asarray(
                phone_state["buffer"],
                dtype=np.float32,
            )

        probability = float(
            predict_sensor_array(X)[-1]
        )

        with phone_state["lock"]:

            phone_state["probability"] = probability

            phone_state["status"] = (
                "FALL DETECTED"
                if probability >= OUTDOOR_THRESHOLD
                else "NORMAL"
            )

    except Exception as error:

        with phone_state["lock"]:
            phone_state["status"] = (
                f"ERROR: {error}"
            )

    return len(samples)


class PhoneHandler(BaseHTTPRequestHandler):

    def do_POST(self):

        try:

            length = int(
                self.headers.get(
                    "Content-Length",
                    0,
                )
            )

            body = self.rfile.read(length)

            message = json.loads(
                body.decode(
                    "utf-8",
                    errors="replace",
                )
            )

            payload = message.get(
                "payload",
                message,
            )

            received = process_phone_payload(
                payload
            )

            print(
                f"PHONE POST received | "
                f"samples={received}"
            )

            response = {
                "status": "received",
                "samples_received": received,
                "buffer_size": len(
                    phone_state["buffer"]
                ),
            }

            out = json.dumps(
                response
            ).encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json",
            )
            self.send_header(
                "Access-Control-Allow-Origin",
                "*",
            )
            self.end_headers()
            self.wfile.write(out)

        except Exception as error:

            print(
                "PHONE SERVER ERROR:",
                error,
            )

            try:

                out = json.dumps({
                    "error": str(error)
                }).encode("utf-8")

                self.send_response(400)
                self.send_header(
                    "Content-Type",
                    "application/json",
                )
                self.send_header(
                    "Access-Control-Allow-Origin",
                    "*",
                )
                self.end_headers()
                self.wfile.write(out)

            except Exception:
                pass

    def do_GET(self):

        with phone_state["lock"]:
            last = phone_state["last_received"]
            samples = len(phone_state["buffer"])

        out = json.dumps({
            "status": "running",
            "phone_stream": (
                last > 0
                and time.time() - last < 5
            ),
            "samples": samples,
        }).encode("utf-8")

        self.send_response(200)
        self.send_header(
            "Content-Type",
            "application/json",
        )
        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )
        self.end_headers()
        self.wfile.write(out)

    def do_OPTIONS(self):

        self.send_response(200)
        self.send_header(
            "Access-Control-Allow-Origin",
            "*",
        )
        self.send_header(
            "Access-Control-Allow-Methods",
            "GET, POST, OPTIONS",
        )
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type",
        )
        self.end_headers()

    def log_message(self, *args):
        return


@st.cache_resource
def start_phone_server():

    server = ThreadingHTTPServer(
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

    print(
        f"Sensor Logger receiver running on port {PHONE_PORT}"
    )

    return server


# ============================================================
# STATUS HELPERS
# ============================================================

def phone_connection_state():

    with phone_state["lock"]:
        last = phone_state["last_received"]

    if last <= 0:
        return "IDLE"

    age = time.time() - last

    if age < 5:
        return "ONLINE"

    return "OFFLINE"


def model_status(path):

    return "ONLINE" if path.exists() else "OFFLINE"


# ============================================================
# NAVIGATION
# ============================================================

st.sidebar.markdown(
    """
    <h3 style="margin-bottom:0;">🚨 Smart Fall Detection</h3>
    <div style="font-size:11px;color:#94A3B8;">
    SAFETY & MONITORING PLATFORM
    </div>
    """,
    unsafe_allow_html=True,
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
# SIDEBAR STATUS
# KEEP ONLINE / OFFLINE / IDLE
# ============================================================

vision_state = model_status(
    YOLO_MODEL
) if GRU_MODEL.exists() else "OFFLINE"

outdoor_state = (
    model_status(OUTDOOR_MODEL)
    if OUTDOOR_SCALER.exists()
    else "OFFLINE"
)

phone_state_text = phone_connection_state()

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**SYSTEM STATUS**"
)

vision_class = (
    "green" if vision_state == "ONLINE"
    else "red"
)

outdoor_class = (
    "green" if outdoor_state == "ONLINE"
    else "red"
)

phone_class = {
    "ONLINE": "green",
    "OFFLINE": "red",
    "IDLE": "gray",
}[phone_state_text]

st.sidebar.markdown(
    f"""
    <div class="sidebar-status">
        <span class="dot {vision_class}"></span>
        Vision model
        <span style="float:right;" class="status-{vision_state.lower()}">
        {vision_state}
        </span>
    </div>

    <div class="sidebar-status">
        <span class="dot {outdoor_class}"></span>
        Outdoor LSTM
        <span style="float:right;" class="status-{outdoor_state.lower()}">
        {outdoor_state}
        </span>
    </div>

    <div class="sidebar-status">
        <span class="dot {phone_class}"></span>
        Phone stream
        <span style="float:right;" class="status-{phone_state_text.lower()}">
        {phone_state_text}
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Indoor — YOLO11n-Pose + GRU\n\n"
    "Outdoor — 9-channel phone sensors + LSTM\n\n"
    "Automatic — routes files by type"
)



# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.markdown(
    """
    <div class="hero-title">
        Detect falls. Respond faster.
    </div>

    <div class="hero-text">
        A multimodal fall-detection system for indoor and
        outdoor environments. Indoor monitoring uses
        YOLO11n-Pose with a GRU temporal classifier, while
        outdoor monitoring uses nine-channel phone motion
        sensing with an LSTM classifier.
    </div>
    """,
    unsafe_allow_html=True,
)
    # --------------------------------------------------------
    # IMAGE PATHS
    # --------------------------------------------------------

    APP_DIR = Path(__file__).resolve().parent
    ASSETS_DIR = APP_DIR / "assets"

    indoor_path = ASSETS_DIR / "indoor.png"
    outdoor_path = ASSETS_DIR / "outdoor.png"
    alert_path = ASSETS_DIR / "alert.png"

    # --------------------------------------------------------
    # THREE HOME CARDS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:

        if indoor_path.is_file():
            with open(indoor_path, "rb") as f:
                st.image(
                    f.read(),
                    width=330,
                )
        else:
            st.error(
                f"indoor.png NOT FOUND\n\n{indoor_path}"
            )

        st.subheader("Indoor Vision")

        st.write(
            "YOLO11n-Pose extracts 17 body keypoints "
            "and the GRU analyzes the temporal pose sequence."
        )

    with c2:

        if outdoor_path.is_file():
            with open(outdoor_path, "rb") as f:
                st.image(
                    f.read(),
                    width=330,
                )
        else:
            st.error(
                f"outdoor.png NOT FOUND\n\n{outdoor_path}"
            )

        st.subheader("Outdoor Sensing")

        st.write(
            "Accelerometer, gravity and gyroscope provide "
            "nine-channel temporal sensor data."
        )

    with c3:

        if alert_path.is_file():
            with open(alert_path, "rb") as f:
                st.image(
                    f.read(),
                    width=330,
                )
        else:
            st.error(
                f"alert.png NOT FOUND\n\n{alert_path}"
            )

        st.subheader("Fall Alert")

        st.write(
            "The final fall probability is compared "
            "with the configured decision threshold."
        )
    
# ============================================================
# INDOOR
# ============================================================

elif page == "Indoor":

    st.title("Indoor Fall Detection")

    mode = st.radio(
        "Select input",
        [
            "Live Camera",
            "Upload Video / Dataset",
        ],
        horizontal=True,
    )

    if mode == "Live Camera":

        st.markdown(
            "### Camera Status"
        )

        if not WEBRTC_AVAILABLE:

            st.error(
                "streamlit-webrtc is not installed."
            )

            st.code(
                r".\.venv\Scripts\python.exe -m pip install streamlit-webrtc"
            )

        else:

            try:

                # Load the actual YOLO11 + GRU models.
                load_vision_models()

                ctx = webrtc_streamer(
                    key="indoor-yolo11-camera",
                    mode=WebRtcMode.SENDRECV,
                    video_processor_factory=(
                        create_live_camera_processor()
                    ),
                    media_stream_constraints={
                        "video": True,
                        "audio": False,
                    },
                    async_processing=True,
                )

                if ctx.state.playing:
                    st.success(
                        "🟢 Camera: ONLINE"
                    )
                else:
                    st.warning(
                        "🟡 Camera: IDLE — press START"
                    )

                st.caption(
                    "YOLO11n-Pose → 17 keypoints → GRU → fall probability"
                )

            except Exception as error:

                st.error(
                    f"Indoor camera error: {error}"
                )

    else:

        st.write(
            "Upload a recorded indoor video. "
            "Analysis starts automatically."
        )

        uploaded_video = st.file_uploader(
            "Video / dataset",
            type=[
                "mp4",
                "avi",
                "mov",
                "mkv",
            ],
            accept_multiple_files=False,
        )

        if uploaded_video is not None:

            file_key = (
                uploaded_video.name
                + str(uploaded_video.size)
            )

            if (
                st.session_state.get(
                    "processed_video_key"
                )
                != file_key
            ):

                with st.spinner(
                    "Analyzing with YOLO11n-Pose + GRU..."
                ):

                    try:

                        (
                            output_path,
                            status,
                            probability,
                        ) = analyse_uploaded_video(
                            uploaded_video
                        )

                        st.session_state[
                            "processed_video_key"
                        ] = file_key

                        st.session_state[
                            "processed_video_path"
                        ] = str(output_path)

                        st.session_state[
                            "processed_video_status"
                        ] = status

                        st.session_state[
                            "processed_video_probability"
                        ] = probability

                    except Exception as error:

                        st.error(
                            f"Video analysis failed: {error}"
                        )

            if (
                st.session_state.get(
                    "processed_video_key"
                )
                == file_key
            ):

                output_path = Path(
                    st.session_state[
                        "processed_video_path"
                    ]
                )

                if output_path.exists():

                    st.subheader(
                        "Analysed Video"
                    )

                    st.video(
                        str(output_path)
                    )

                    show_result(
                        st.session_state[
                            "processed_video_status"
                        ],
                        st.session_state[
                            "processed_video_probability"
                        ],
                    )

                    with open(
                        output_path,
                        "rb",
                    ) as file:

                        st.download_button(
                            "Download analysed video",
                            data=file,
                            file_name=(
                                "fall_detection_result.mp4"
                            ),
                            mime="video/mp4",
                        )

# ============================================================
# OUTDOOR
# ============================================================

elif page == "Outdoor":

    st.title("Outdoor Fall Detection")

    mode = st.radio(
        "Select input",
        [
            "Live Phone",
            "Upload Sensor Data",
        ],
        horizontal=True,
    )

    if mode == "Live Phone":

        start_phone_server()

        @st.fragment(run_every="1s")
        def live_phone_status():

            connection = phone_connection_state()

            with phone_state["lock"]:

                status = phone_state["status"]
                probability = phone_state["probability"]
                samples = len(
                    phone_state["buffer"]
                )

            if connection == "ONLINE":

                st.success(
                    "🟢 Phone stream: ONLINE"
                )

            elif connection == "OFFLINE":

                st.error(
                    "🔴 Phone stream: OFFLINE"
                )

            else:

                st.warning(
                    "🟡 Phone stream: IDLE — waiting for Sensor Logger"
                )

            if samples < OUTDOOR_SEQUENCE_LENGTH:

                st.info(
                    f"Collecting sensor data: "
                    f"{samples}/{OUTDOOR_SEQUENCE_LENGTH}"
                )

            else:

                show_result(
                    status,
                    probability,
                )

            st.caption(
                f"Live samples: "
                f"{samples}/{OUTDOOR_SEQUENCE_LENGTH}"
            )

            st.code(
                f"http://<YOUR-PC-IP>:{PHONE_PORT}",
                language="text",
            )

        live_phone_status()

    else:

        st.write(
            "Upload recorded nine-channel sensor data. "
            "The outdoor LSTM analyses 200-sample temporal windows."
        )

        sensor_files = st.file_uploader(
            "Sensor dataset",
            type=[
                "txt",
                "csv",
            ],
            accept_multiple_files=True,
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

                            data, probabilities = (
                                predict_sensor_file(file)
                            )

                        maximum = float(
                            np.max(probabilities)
                        )

                        file_fall = bool(
                            np.any(
                                probabilities
                                >= OUTDOOR_THRESHOLD
                            )
                        )

                        st.markdown(
                            f"### {file.name}"
                        )

                        st.caption(
                            f"{len(data)} samples | "
                            f"{len(probabilities)} windows"
                        )

                        show_result(
                            "FALL DETECTED"
                            if file_fall
                            else "NORMAL",
                            maximum,
                        )

                    except Exception as error:

                        st.error(
                            f"{file.name}: {error}"
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
        "The application automatically selects the matching branch."
    )

    files = st.file_uploader(
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

    if files:

        videos = [
            f for f in files
            if Path(f.name).suffix.lower()
            in {
                ".mp4",
                ".avi",
                ".mov",
                ".mkv",
            }
        ]

        sensors = [
            f for f in files
            if Path(f.name).suffix.lower()
            in {
                ".txt",
                ".csv",
            }
        ]

        for file in videos:

            with st.spinner(
                f"Analyzing {file.name} with YOLO11 + GRU..."
            ):

                try:

                    (
                        output_path,
                        status,
                        probability,
                    ) = analyse_uploaded_video(
                        file
                    )

                    st.video(
                        str(output_path)
                    )

                    show_result(
                        status,
                        probability,
                    )

                except Exception as error:

                    st.error(
                        f"{file.name}: {error}"
                    )

        for file in sensors:

            with st.spinner(
                f"Analyzing {file.name} with outdoor LSTM..."
            ):

                try:

                    data, probabilities = (
                        predict_sensor_file(file)
                    )

                    maximum = float(
                        np.max(probabilities)
                    )

                    file_fall = bool(
                        np.any(
                            probabilities
                            >= OUTDOOR_THRESHOLD
                        )
                    )

                    st.markdown(
                        f"### {file.name}"
                    )

                    st.caption(
                        f"{len(data)} samples | "
                        f"{len(probabilities)} windows"
                    )

                    show_result(
                        "FALL DETECTED"
                        if file_fall
                        else "NORMAL",
                        maximum,
                    )

                except Exception as error:

                    st.error(
                        f"{file.name}: {error}"
                    )
