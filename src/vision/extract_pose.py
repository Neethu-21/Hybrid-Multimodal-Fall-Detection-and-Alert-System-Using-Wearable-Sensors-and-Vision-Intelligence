from pathlib import Path
import numpy as np
from ultralytics import YOLO
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = PROJECT_ROOT / "data" / "indoor" / "URFD"
OUTPUT_DIR = PROJECT_ROOT / "data" / "indoor" / "pose_data"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# YOLO POSE MODEL
# ============================================================

MODEL_NAME = "yolo11n-pose.pt"

print("Loading YOLO pose model...")
model = YOLO(MODEL_NAME)

print("Model loaded successfully.")


# ============================================================
# FIND IMAGE FILES
# ============================================================

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def get_images(sequence_folder):

    images = []

    for file in sequence_folder.rglob("*"):
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(file)

    return sorted(images)


# ============================================================
# PROCESS ONE SEQUENCE
# ============================================================

def process_sequence(sequence_folder):

    sequence_name = sequence_folder.name

    print("\n" + "=" * 60)
    print(f"Processing: {sequence_name}")

    images = get_images(sequence_folder)

    if len(images) == 0:
        print("No images found. Skipping.")
        return

    print(f"Images found: {len(images)}")

    keypoints_sequence = []
    confidence_sequence = []
    frame_names = []

    for index, image_path in enumerate(images):

        try:
            image = Image.open(image_path).convert("RGB")

            results = model(
                image,
                verbose=False
            )

            result = results[0]

            # ------------------------------------------------
            # No person detected
            # ------------------------------------------------

            if result.keypoints is None or len(result.keypoints) == 0:

                keypoints_sequence.append(
                    np.zeros((1, 17, 2), dtype=np.float32)
                )

                confidence_sequence.append(
                    np.zeros((1, 17), dtype=np.float32)
                )

            else:

                # All detected persons
                keypoints = result.keypoints.xy.cpu().numpy()

                if result.keypoints.conf is not None:
                    confidence = result.keypoints.conf.cpu().numpy()
                else:
                    confidence = np.ones(
                        (keypoints.shape[0], keypoints.shape[1]),
                        dtype=np.float32
                    )

                keypoints_sequence.append(
                    keypoints.astype(np.float32)
                )

                confidence_sequence.append(
                    confidence.astype(np.float32)
                )

            frame_names.append(image_path.name)

            if (index + 1) % 100 == 0:
                print(
                    f"Processed {index + 1}/{len(images)} frames"
                )

        except Exception as error:

            print(
                f"Error processing {image_path.name}: {error}"
            )

    # --------------------------------------------------------
    # Save sequence
    # --------------------------------------------------------

    output_file = OUTPUT_DIR / f"{sequence_name}.npz"

    np.savez_compressed(
        output_file,
        keypoints=np.array(keypoints_sequence, dtype=object),
        confidence=np.array(confidence_sequence, dtype=object),
        frame_names=np.array(frame_names)
    )

    print(f"Saved: {output_file}")


# ============================================================
# MAIN
# ============================================================

def main():

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found:\n{DATASET_DIR}"
        )

    sequence_folders = [
        folder
        for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
        and (
            folder.name.lower().startswith("fall")
            or folder.name.lower().startswith("adl")
        )
    ]

    sequence_folders = sorted(sequence_folders)

    print("\nURFD DATASET")
    print("-" * 60)
    print(f"Dataset location: {DATASET_DIR}")
    print(f"Sequences found: {len(sequence_folders)}")

    if len(sequence_folders) == 0:

        print(
            "\nNo extracted sequence folders were found."
        )

        print(
            "Make sure the ZIP files have been extracted "
            "inside data/indoor/URFD/"
        )

        return

    for sequence_folder in sequence_folders:
        process_sequence(sequence_folder)

    print("\n" + "=" * 60)
    print("POSE EXTRACTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()