from pathlib import Path
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GRU_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "vision"
    / "best_gru.keras"
)

YOLO_MODEL_NAME = "yolo11n-pose.pt"


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 20

# Selected from validation F1 evaluation
FALL_THRESHOLD = 0.50

# Consecutive positive GRU predictions
FALL_CONFIRMATION_FRAMES = 3

# YOLO confidence
YOLO_CONFIDENCE = 0.50

# Short detection dropout tolerance
MAX_MISSED_FRAMES = 8

# Probability smoothing
PROBABILITY_SMOOTHING_WINDOW = 3

# Skeleton drawing confidence
KEYPOINT_DRAW_CONFIDENCE = 0.20

# Hold fall alert on screen
FALL_ALERT_HOLD_FRAMES = 90


# ============================================================
# COCO 17-KEYPOINT SKELETON
# ============================================================

SKELETON_CONNECTIONS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),

    (5, 6),

    (5, 7),
    (7, 9),

    (6, 8),
    (8, 10),

    (5, 11),
    (6, 12),

    (11, 12),

    (11, 13),
    (13, 15),

    (12, 14),
    (14, 16),
]


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 65)
print("LOADING FINAL INDOOR FALL DETECTOR")
print("=" * 65)

print("Loading YOLO Pose model...")

pose_model = YOLO(
    YOLO_MODEL_NAME
)

print("Loading trained GRU model...")

gru_model = tf.keras.models.load_model(
    GRU_MODEL_PATH
)

print("Models loaded successfully.")


# ============================================================
# NORMALIZE KEYPOINTS
# ============================================================

def normalize_keypoints(keypoints):

    keypoints = np.asarray(
        keypoints,
        dtype=np.float32
    )

    left_hip = keypoints[11]
    right_hip = keypoints[12]

    center = (
        left_hip +
        right_hip
    ) / 2.0

    normalized = (
        keypoints - center
    )

    left_shoulder = normalized[5]
    right_shoulder = normalized[6]

    shoulder_distance = np.linalg.norm(
        left_shoulder -
        right_shoulder
    )

    if (
        not np.isfinite(
            shoulder_distance
        )
        or shoulder_distance < 1e-6
    ):
        shoulder_distance = 1.0

    normalized = (
        normalized /
        shoulder_distance
    )

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return normalized.astype(
        np.float32
    )


# ============================================================
# SELECT PERSON
# ============================================================

def select_person(result):

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):
        return None

    box_conf = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    if len(box_conf) == 0:
        return None

    best_index = int(
        np.argmax(box_conf)
    )

    detection_confidence = float(
        box_conf[best_index]
    )

    if (
        detection_confidence
        < YOLO_CONFIDENCE
    ):
        return None

    if (
        result.keypoints is None
        or len(result.keypoints) == 0
    ):
        return None

    keypoints = (
        result.keypoints.xy
        .detach()
        .cpu()
        .numpy()
    )

    if best_index >= len(keypoints):
        return None

    person = keypoints[
        best_index
    ].astype(
        np.float32
    )

    if person.shape != (17, 2):
        return None

    # --------------------------------------------------------
    # Keypoint confidence
    # --------------------------------------------------------

    if result.keypoints.conf is not None:

        kp_conf = (
            result.keypoints.conf
            .detach()
            .cpu()
            .numpy()
        )

        person_kp_conf = (
            kp_conf[best_index]
            .astype(np.float32)
        )

    else:

        person_kp_conf = np.ones(
            17,
            dtype=np.float32
        )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    bbox = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .numpy()[best_index]
        .astype(np.float32)
    )

    return (
        person,
        person_kp_conf,
        bbox,
        detection_confidence
    )


# ============================================================
# GRU PREDICTION
# ============================================================

def predict_fall(sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    sequence = sequence.reshape(
        1,
        SEQUENCE_LENGTH,
        34
    )

    probability = float(
        gru_model.predict(
            sequence,
            verbose=0
        )[0][0]
    )

    return float(
        np.clip(
            probability,
            0.0,
            1.0
        )
    )


# ============================================================
# DRAW SELECTED PERSON
# ============================================================

def draw_selected_person(
    frame,
    keypoints,
    keypoint_confidence,
    bbox,
    detection_confidence
):

    output = frame.copy()

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x1, y1, x2, y2 = (
        bbox.astype(int)
    )

    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        output.shape[1] - 1,
        x2
    )

    y2 = min(
        output.shape[0] - 1,
        y2
    )

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        output,
        f"person {detection_confidence:.2f}",
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Skeleton
    # --------------------------------------------------------

    for start, end in SKELETON_CONNECTIONS:

        if (
            keypoint_confidence[start]
            < KEYPOINT_DRAW_CONFIDENCE
            or
            keypoint_confidence[end]
            < KEYPOINT_DRAW_CONFIDENCE
        ):
            continue

        p1 = tuple(
            keypoints[start].astype(int)
        )

        p2 = tuple(
            keypoints[end].astype(int)
        )

        cv2.line(
            output,
            p1,
            p2,
            (255, 0, 255),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # Points
    # --------------------------------------------------------

    for i in range(17):

        if (
            keypoint_confidence[i]
            < KEYPOINT_DRAW_CONFIDENCE
        ):
            continue

        x, y = (
            keypoints[i].astype(int)
        )

        cv2.circle(
            output,
            (x, y),
            4,
            (0, 255, 0),
            -1,
            cv2.LINE_AA
        )

    return output


# ============================================================
# MAIN DETECTOR
# ============================================================

def run_detector():

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    # --------------------------------------------------------
    # Webcam resolution
    # --------------------------------------------------------

    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1280
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        720
    )

    # --------------------------------------------------------
    # Window
    # --------------------------------------------------------

    WINDOW_NAME = (
        "Indoor Fall Detection - YOLO + GRU"
    )

    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )

    cv2.setWindowProperty(
        WINDOW_NAME,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    # --------------------------------------------------------
    # Temporal sequence
    # --------------------------------------------------------

    sequence_buffer = deque(
        maxlen=SEQUENCE_LENGTH
    )

    # --------------------------------------------------------
    # Probability history
    # --------------------------------------------------------

    probability_history = deque(
        maxlen=PROBABILITY_SMOOTHING_WINDOW
    )

    # --------------------------------------------------------
    # Person tracking state
    # --------------------------------------------------------

    last_valid_pose = None
    last_keypoint_confidence = None
    last_bbox = None
    last_detection_confidence = 0.0

    missed_frames = 0

    # --------------------------------------------------------
    # Fall state
    # --------------------------------------------------------

    positive_count = 0

    fall_detected = False

    fall_alert_counter = 0

    last_probability = None

    # ========================================================
    # START
    # ========================================================

    print()
    print("=" * 65)
    print("INDOOR FALL DETECTOR STARTED")
    print("=" * 65)
    print()
    print("YOLO Pose + Improved GRU")
    print(
        f"Sequence length       : "
        f"{SEQUENCE_LENGTH}"
    )
    print(
        f"YOLO confidence       : "
        f"{YOLO_CONFIDENCE}"
    )
    print(
        f"Fall threshold        : "
        f"{FALL_THRESHOLD}"
    )
    print(
        f"Fall confirmation     : "
        f"{FALL_CONFIRMATION_FRAMES}"
        f" frames"
    )
    print(
        f"Probability smoothing : "
        f"{PROBABILITY_SMOOTHING_WINDOW}"
    )
    print()
    print("Q / ESC = Quit")
    print("R       = Reset fall alert")
    print()

    # ========================================================
    # LOOP
    # ========================================================

    while True:

        success, frame = camera.read()

        if not success:

            print(
                "Could not read webcam frame."
            )

            break

        # ----------------------------------------------------
        # YOLO TRACK
        # ----------------------------------------------------

        results = pose_model.track(
            frame,
            persist=True,
            conf=YOLO_CONFIDENCE,
            classes=[0],
            verbose=False
        )

        result = results[0]

        detection = select_person(
            result
        )

        person_found = (
            detection is not None
        )

        # ====================================================
        # PERSON FOUND
        # ====================================================

        if person_found:

            (
                person,
                keypoint_confidence,
                bbox,
                detection_confidence
            ) = detection

            missed_frames = 0

            # ------------------------------------------------
            # Save current pose
            # ------------------------------------------------

            last_valid_pose = (
                person.copy()
            )

            last_keypoint_confidence = (
                keypoint_confidence.copy()
            )

            last_bbox = (
                bbox.copy()
            )

            last_detection_confidence = (
                detection_confidence
            )

            # ------------------------------------------------
            # Normalize and add to sequence
            # ------------------------------------------------

            normalized = (
                normalize_keypoints(
                    person
                )
            )

            sequence_buffer.append(
                normalized
            )

            # ------------------------------------------------
            # GRU prediction
            # ------------------------------------------------

            if (
                len(sequence_buffer)
                == SEQUENCE_LENGTH
            ):

                sequence = np.array(
                    sequence_buffer,
                    dtype=np.float32
                )

                probability = (
                    predict_fall(
                        sequence
                    )
                )

                probability_history.append(
                    probability
                )

                # ------------------------------------------------
                # Median smoothing
                # ------------------------------------------------

                smoothed_probability = float(
                    np.median(
                        probability_history
                    )
                )

                last_probability = (
                    smoothed_probability
                )

                # ------------------------------------------------
                # FALL THRESHOLD
                # ------------------------------------------------

                if (
                    smoothed_probability
                    >= FALL_THRESHOLD
                ):

                    positive_count += 1

                else:

                    positive_count = 0

                # ------------------------------------------------
                # CONFIRM FALL
                # ------------------------------------------------

                if (
                    positive_count
                    >= FALL_CONFIRMATION_FRAMES
                ):

                    fall_detected = True

                    fall_alert_counter = (
                        FALL_ALERT_HOLD_FRAMES
                    )

        # ====================================================
        # PERSON TEMPORARILY LOST
        # ====================================================

        else:

            missed_frames += 1

            # ------------------------------------------------
            # IMPORTANT:
            #
            # Do NOT immediately clear the sequence.
            # A fast movement/fall can cause YOLO to miss
            # a few frames.
            # ------------------------------------------------

            if (
                missed_frames
                <= MAX_MISSED_FRAMES
            ):

                pass

            else:

                # ------------------------------------------------
                # Person has genuinely been absent for a while.
                # ------------------------------------------------

                sequence_buffer.clear()

                probability_history.clear()

                positive_count = 0

                last_probability = None

                last_valid_pose = None

                last_keypoint_confidence = None

                last_bbox = None

                last_detection_confidence = 0.0

        # ====================================================
        # FALL ALERT HOLD
        # ====================================================

        if fall_detected:

            if fall_alert_counter > 0:

                fall_alert_counter -= 1

            else:

                fall_detected = False

                positive_count = 0

                probability_history.clear()

        # ====================================================
        # DISPLAY
        # ====================================================

        annotated_frame = frame.copy()

        # ----------------------------------------------------
        # Draw ONLY currently detected person.
        #
        # This prevents ghost skeletons.
        # ----------------------------------------------------

        if person_found:

            annotated_frame = (
                draw_selected_person(
                    annotated_frame,
                    person,
                    keypoint_confidence,
                    bbox,
                    detection_confidence
                )
            )

        # ----------------------------------------------------
        # Status
        # ----------------------------------------------------

        if fall_detected:

            status = "FALL DETECTED"

            status_color = (
                0,
                0,
                255
            )

            status_scale = 1.3

            status_thickness = 4

        elif person_found:

            status = "NORMAL"

            status_color = (
                0,
                255,
                0
            )

            status_scale = 1.1

            status_thickness = 3

        elif (
            missed_frames
            <= MAX_MISSED_FRAMES
        ):

            status = "TRACKING..."

            status_color = (
                255,
                255,
                0
            )

            status_scale = 1.1

            status_thickness = 3

        else:

            status = "NO PERSON"

            status_color = (
                0,
                165,
                255
            )

            status_scale = 1.1

            status_thickness = 3

        # ----------------------------------------------------
        # Status text
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            status,
            (35, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            status_scale,
            status_color,
            status_thickness,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Probability
        # ----------------------------------------------------

        if last_probability is None:

            probability_text = (
                "Fall probability: --"
            )

        else:

            probability_text = (
                f"Fall probability: "
                f"{last_probability:.2f}"
            )

        cv2.putText(
            annotated_frame,
            probability_text,
            (35, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Buffer
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"Buffer: "
            f"{len(sequence_buffer)}/"
            f"{SEQUENCE_LENGTH}",
            (35, 155),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.80,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Confirmation
        # ----------------------------------------------------

        if (
            len(sequence_buffer)
            == SEQUENCE_LENGTH
            and not fall_detected
        ):

            cv2.putText(
                annotated_frame,
                f"Fall confirmation: "
                f"{positive_count}/"
                f"{FALL_CONFIRMATION_FRAMES}",
                (35, 195),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        # ----------------------------------------------------
        # Controls
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            "Q / ESC: Quit    R: Reset Alert",
            (
                35,
                annotated_frame.shape[0] - 30
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            annotated_frame
        )

        # ----------------------------------------------------
        # Keyboard
        # ----------------------------------------------------

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if (
            key == ord("q")
            or key == ord("Q")
            or key == 27
        ):

            print()
            print(
                "Stopping detector..."
            )

            break

        if (
            key == ord("r")
            or key == ord("R")
        ):

            fall_detected = False

            fall_alert_counter = 0

            positive_count = 0

            probability_history.clear()

            print(
                "Fall alert reset."
            )

    # ========================================================
    # CLEANUP
    # ========================================================

    camera.release()

    cv2.destroyAllWindows()

    cv2.waitKey(1)

    print(
        "Indoor fall detector stopped."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_detector()