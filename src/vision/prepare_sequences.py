from pathlib import Path
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

POSE_DIR = PROJECT_ROOT / "data" / "indoor" / "pose_data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "indoor" / "sequences"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

SEQUENCE_LENGTH = 20
STRIDE = 5

NUM_KEYPOINTS = 17
COORDINATES = 2

# Minimum confidence required for a frame to be considered
# a reasonably valid pose.
MIN_FRAME_CONFIDENCE = 0.35

# Minimum number of valid frames in a 20-frame window.
MIN_VALID_FRAMES = 15

# Maximum gap of missing frames that we will interpolate.
MAX_INTERPOLATION_GAP = 5

# Fall posture thresholds.
# These are intentionally not extremely aggressive.
TORSO_ANGLE_THRESHOLD = 50.0
BODY_RATIO_THRESHOLD = 1.25

# Motion threshold for detecting a rapid transition.
MOTION_THRESHOLD = 0.08

# Number of frames at the end of the window in which
# we look for a fall transition.
FALL_TRANSITION_FRAMES = 8


# ============================================================
# LOAD ONE POSE FILE
# ============================================================

def load_pose_file(file_path):

    data = np.load(
        file_path,
        allow_pickle=True
    )

    keypoints = data["keypoints"]
    confidence = data["confidence"]

    return keypoints, confidence


# ============================================================
# SELECT MAIN PERSON
# ============================================================

def select_main_person(
    frame_keypoints,
    frame_confidence
):

    if (
        frame_keypoints is None
        or len(frame_keypoints) == 0
    ):
        return None, 0.0

    if (
        frame_confidence is None
        or len(frame_confidence) == 0
    ):
        return (
            frame_keypoints[0].astype(np.float32),
            0.0
        )

    person_scores = np.mean(
        frame_confidence,
        axis=1
    )

    best_person = int(
        np.argmax(person_scores)
    )

    selected = (
        frame_keypoints[best_person]
        .astype(np.float32)
    )

    score = float(
        person_scores[best_person]
    )

    return selected, score


# ============================================================
# BUILD CONTINUOUS POSE SEQUENCE
# ============================================================

def build_pose_sequence(
    keypoints,
    confidence
):

    frame_count = len(keypoints)

    selected = np.zeros(
        (
            frame_count,
            NUM_KEYPOINTS,
            COORDINATES
        ),
        dtype=np.float32
    )

    valid = np.zeros(
        frame_count,
        dtype=bool
    )

    frame_scores = np.zeros(
        frame_count,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Select the main person in every frame
    # --------------------------------------------------------

    for i in range(frame_count):

        person, score = select_main_person(
            keypoints[i],
            confidence[i]
        )

        if person is None:
            continue

        selected[i] = person
        frame_scores[i] = score

        if score >= MIN_FRAME_CONFIDENCE:

            # Check that the important body points exist.
            important = person[
                [5, 6, 11, 12]
            ]

            if np.all(
                np.isfinite(important)
            ):
                valid[i] = True

    # --------------------------------------------------------
    # Interpolate SHORT missing gaps
    # --------------------------------------------------------

    for kp in range(NUM_KEYPOINTS):

        for coord in range(COORDINATES):

            values = selected[:, kp, coord]

            valid_indices = np.where(valid)[0]

            if len(valid_indices) < 2:
                continue

            for j in range(
                len(valid_indices) - 1
            ):

                left = valid_indices[j]
                right = valid_indices[j + 1]

                gap = right - left - 1

                if (
                    gap > 0
                    and gap <= MAX_INTERPOLATION_GAP
                ):

                    selected[
                        left:right + 1,
                        kp,
                        coord
                    ] = np.linspace(
                        values[left],
                        values[right],
                        right - left + 1
                    )

                    valid[
                        left:right + 1
                    ] = True

    return selected, valid


# ============================================================
# NORMALIZE ONE WINDOW
# ============================================================

def normalize_window(window):

    window = window.astype(
        np.float32
    ).copy()

    output = np.zeros_like(
        window,
        dtype=np.float32
    )

    for t in range(len(window)):

        frame = window[t]

        left_hip = frame[11]
        right_hip = frame[12]

        left_shoulder = frame[5]
        right_shoulder = frame[6]

        center = (
            left_hip +
            right_hip
        ) / 2.0

        centered = (
            frame - center
        )

        shoulder_distance = np.linalg.norm(
            left_shoulder -
            right_shoulder
        )

        if shoulder_distance < 1e-6:

            # Fall back to body height
            body_height = np.max(
                frame[:, 1]
            ) - np.min(
                frame[:, 1]
            )

            if body_height < 1e-6:
                body_height = 1.0

            scale = body_height

        else:

            scale = shoulder_distance

        output[t] = (
            centered / scale
        )

    return output


# ============================================================
# TORSO ANGLE
# ============================================================

def torso_angle_degrees(frame):

    left_shoulder = frame[5]
    right_shoulder = frame[6]

    left_hip = frame[11]
    right_hip = frame[12]

    shoulder_center = (
        left_shoulder +
        right_shoulder
    ) / 2.0

    hip_center = (
        left_hip +
        right_hip
    ) / 2.0

    torso = (
        shoulder_center -
        hip_center
    )

    # Image Y axis points downward.
    vertical = np.array(
        [0.0, -1.0],
        dtype=np.float32
    )

    norm = np.linalg.norm(
        torso
    )

    if norm < 1e-6:
        return 0.0

    torso = torso / norm

    cosine = np.clip(
        np.dot(
            torso,
            vertical
        ),
        -1.0,
        1.0
    )

    angle = np.degrees(
        np.arccos(cosine)
    )

    return float(angle)


# ============================================================
# BODY ASPECT RATIO
# ============================================================

def body_aspect_ratio(frame):

    valid_points = frame[
        np.all(
            np.isfinite(frame),
            axis=1
        )
    ]

    if len(valid_points) < 5:
        return 0.0

    width = (
        np.max(
            valid_points[:, 0]
        )
        -
        np.min(
            valid_points[:, 0]
        )
    )

    height = (
        np.max(
            valid_points[:, 1]
        )
        -
        np.min(
            valid_points[:, 1]
        )
    )

    if height < 1e-6:
        return 0.0

    return float(
        width / height
    )


# ============================================================
# FALL-LIKE WINDOW CHECK
# ============================================================

def is_fall_window(window):

    frame_count = len(window)

    if frame_count < SEQUENCE_LENGTH:
        return False

    angles = []
    ratios = []
    centers = []

    for frame in window:

        angles.append(
            torso_angle_degrees(
                frame
            )
        )

        ratios.append(
            body_aspect_ratio(
                frame
            )
        )

        hip_center = (
            frame[11] +
            frame[12]
        ) / 2.0

        centers.append(
            hip_center
        )

    angles = np.array(
        angles,
        dtype=np.float32
    )

    ratios = np.array(
        ratios,
        dtype=np.float32
    )

    centers = np.array(
        centers,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Final portion of the sequence
    # --------------------------------------------------------

    start = max(
        0,
        frame_count -
        FALL_TRANSITION_FRAMES
    )

    final_angles = angles[start:]
    final_ratios = ratios[start:]

    # --------------------------------------------------------
    # Condition 1:
    # Body becomes substantially horizontal.
    # --------------------------------------------------------

    horizontal_frames = np.sum(
        final_angles >=
        TORSO_ANGLE_THRESHOLD
    )

    horizontal_posture = (
        horizontal_frames >= 3
    )

    # --------------------------------------------------------
    # Condition 2:
    # Body becomes wide relative to height.
    # --------------------------------------------------------

    lying_frames = np.sum(
        final_ratios >=
        BODY_RATIO_THRESHOLD
    )

    horizontal_shape = (
        lying_frames >= 3
    )

    # --------------------------------------------------------
    # Condition 3:
    # There is a meaningful movement of the hip
    # through the sequence.
    # --------------------------------------------------------

    if len(centers) >= 2:

        movement = np.linalg.norm(
            np.diff(
                centers,
                axis=0
            ),
            axis=1
        )

        peak_motion = float(
            np.max(movement)
        )

    else:

        peak_motion = 0.0

    rapid_motion = (
        peak_motion >=
        MOTION_THRESHOLD
    )

    # --------------------------------------------------------
    # Condition 4:
    # Posture transition.
    #
    # Early frames are relatively upright and
    # later frames become horizontal.
    # --------------------------------------------------------

    early_count = max(
        1,
        frame_count // 3
    )

    early_angles = angles[
        :early_count
    ]

    early_upright = (
        np.mean(
            early_angles
        ) < 45.0
    )

    late_horizontal = (
        np.mean(
            final_angles
        ) >=
        TORSO_ANGLE_THRESHOLD
    )

    transition = (
        early_upright
        and late_horizontal
    )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------
    #
    # A fall window should NOT be positive merely because
    # the subject is standing or moving.
    #
    # We require strong horizontal posture, preferably with
    # a transition or movement.
    # --------------------------------------------------------

    if horizontal_posture:

        if (
            transition
            or horizontal_shape
            or rapid_motion
        ):
            return True

    if (
        horizontal_shape
        and rapid_motion
    ):
        return True

    return False


# ============================================================
# CREATE WINDOWS
# ============================================================

def create_windows(
    sequence,
    valid,
    sequence_length,
    stride,
    sequence_name
):

    windows = []
    labels = []

    total_frames = len(sequence)

    if total_frames < sequence_length:
        return windows, labels

    is_fall_video = (
        sequence_name.lower().startswith(
            "fall"
        )
    )

    is_adl_video = (
        sequence_name.lower().startswith(
            "adl"
        )
    )

    if not (
        is_fall_video
        or is_adl_video
    ):
        return windows, labels

    for start in range(
        0,
        total_frames -
        sequence_length + 1,
        stride
    ):

        end = (
            start +
            sequence_length
        )

        window_valid = valid[
            start:end
        ]

        # ----------------------------------------------------
        # Reject windows with too many missing frames
        # ----------------------------------------------------

        valid_count = int(
            np.sum(window_valid)
        )

        if valid_count < MIN_VALID_FRAMES:
            continue

        window = sequence[
            start:end
        ]

        # ----------------------------------------------------
        # Normalize AFTER selecting the complete window
        # ----------------------------------------------------

        normalized = normalize_window(
            window
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        if is_adl_video:

            label = 0

        else:

            # IMPORTANT:
            # Do NOT label every window from a fall video
            # as a fall.
            #
            # Only windows containing fall-like temporal
            # behaviour are labelled positive.
            if is_fall_window(
                window
            ):

                label = 1

            else:

                # Pre-fall / normal activity in a fall
                # recording is treated as ADL.
                label = 0

        windows.append(
            normalized
        )

        labels.append(
            label
        )

    return windows, labels


# ============================================================
# PROCESS ONE SEQUENCE
# ============================================================

def process_sequence(file_path):

    sequence_name = file_path.stem

    print("\n" + "=" * 60)
    print(
        f"Processing: {sequence_name}"
    )

    keypoints, confidence = (
        load_pose_file(
            file_path
        )
    )

    print(
        f"Frames: {len(keypoints)}"
    )

    # --------------------------------------------------------
    # Build clean continuous pose sequence
    # --------------------------------------------------------

    selected, valid = (
        build_pose_sequence(
            keypoints,
            confidence
        )
    )

    print(
        "Valid frames before interpolation:",
        int(np.sum(valid)),
        "/",
        len(valid)
    )

    # --------------------------------------------------------
    # Create windows
    # --------------------------------------------------------

    windows, labels = (
        create_windows(
            selected,
            valid,
            SEQUENCE_LENGTH,
            STRIDE,
            sequence_name
        )
    )

    if len(windows) == 0:

        print(
            "No usable windows created."
        )

        return

    windows = np.array(
        windows,
        dtype=np.float32
    )

    labels = np.array(
        labels,
        dtype=np.int64
    )

    fall_count = int(
        np.sum(labels == 1)
    )

    adl_count = int(
        np.sum(labels == 0)
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        f"{sequence_name}.npz"
    )

    np.savez_compressed(
        output_file,
        X=windows,
        y=labels
    )

    print(
        f"Windows created : {len(windows)}"
    )

    print(
        f"ADL windows     : {adl_count}"
    )

    print(
        f"Fall windows    : {fall_count}"
    )

    print(
        f"Saved           : {output_file}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    if not POSE_DIR.exists():

        raise FileNotFoundError(
            f"Pose directory not found:\n"
            f"{POSE_DIR}"
        )

    pose_files = sorted(
        POSE_DIR.glob("*.npz")
    )

    print("\n")
    print("=" * 60)
    print("INDOOR SEQUENCE PREPARATION")
    print("=" * 60)

    print(
        f"Pose files found: {len(pose_files)}"
    )

    if len(pose_files) == 0:

        print(
            "No pose files found."
        )

        return

    total_windows = 0
    total_falls = 0
    total_adl = 0

    for file_path in pose_files:

        process_sequence(
            file_path
        )

        # Read back the generated file so we can
        # print a final dataset summary.
        output_file = (
            OUTPUT_DIR /
            f"{file_path.stem}.npz"
        )

        if output_file.exists():

            data = np.load(
                output_file
            )

            y = data["y"]

            total_windows += len(y)

            total_falls += int(
                np.sum(y == 1)
            )

            total_adl += int(
                np.sum(y == 0)
            )

    print("\n")
    print("=" * 60)
    print("FINAL SEQUENCE DATASET")
    print("=" * 60)

    print(
        f"Total windows : {total_windows}"
    )

    print(
        f"ADL windows   : {total_adl}"
    )

    print(
        f"Fall windows  : {total_falls}"
    )

    print("=" * 60)
    print(
        "SEQUENCE PREPARATION COMPLETED"
    )
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()