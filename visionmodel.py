import os
import glob
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- Configuration ---
# Update this to the exact path printed by your kagglehub script
URFD_PATH = r"URFD/UR_fall_detection_dataset_cam0_rgb" # MAKE SURE THIS IS CORRECT

# --- Initialize MediaPipe Tasks API (Modern Way) ---
# Ensure 'pose_landmarker_lite.task' is in the same folder as this script!
try:
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False)
    detector = vision.PoseLandmarker.create_from_options(options)
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load the MediaPipe task file. Did you download 'pose_landmarker_lite.task'?\nError details: {e}")
    exit()

def extract_vision_features(image_path):
    """Extracts keypoints and returns geometric features using the new Tasks API."""
    # 1. Load image using MediaPipe's new Image object
    mp_image = mp.Image.create_from_file(image_path)
    
    # 2. Detect poses
    detection_result = detector.detect(mp_image)
    
    # Check if a person was detected
    if not detection_result.pose_landmarks:
        return None
        
    # The new API returns a list of lists. We grab the first person detected [0]
    landmarks = detection_result.pose_landmarks[0]
    
    # 3. Get Key Coordinates
    # In the new API, index 0 is Nose, 11/12 are Shoulders, 23/24 are Hips, 27/28 are Ankles
    nose = landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    left_ankle = landmarks[27]
    right_ankle = landmarks[28]
    
    # 4. Calculate Features
    # Feature 1: Bounding Box Aspect Ratio (Width / Height)
    min_x = min(nose.x, left_ankle.x, right_ankle.x, left_shoulder.x, right_shoulder.x)
    max_x = max(nose.x, left_ankle.x, right_ankle.x, left_shoulder.x, right_shoulder.x)
    min_y = min(nose.y, left_ankle.y, right_ankle.y, left_shoulder.y, right_shoulder.y)
    max_y = max(nose.y, left_ankle.y, right_ankle.y, left_shoulder.y, right_shoulder.y)
    
    width = max_x - min_x
    height = max_y - min_y
    aspect_ratio = width / height if height > 0 else 0
    
    # Feature 2: Hip Center Y (to track dropping)
    hip_center_y = (left_hip.y + right_hip.y) / 2
    
    # Feature 3: Shoulder Center Y
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
    
    return [aspect_ratio, hip_center_y, shoulder_center_y]

def process_urfd():
    print("Processing URFD Data using Tasks API... (This will take a while)")
    X = []
    y = []
    
    # Safely check path
    if not os.path.exists(URFD_PATH):
        raise FileNotFoundError(f"Cannot find URFD_PATH: {URFD_PATH}")
    
    # Find all sequence folders (e.g., adl-04-cam0-rgb, fall-01-cam0-rgb)
    folders = [f.path for f in os.scandir(URFD_PATH) if f.is_dir()]
    
    for folder in folders:
        folder_name = os.path.basename(folder)
        # Determine label based on folder name
        label = 1 if "fall" in folder_name.lower() else 0
        
        # Get all PNGs and sort them
        images = glob.glob(os.path.join(folder, "*.png"))
        images.sort()
        
        for img_path in images:
            features = extract_vision_features(img_path)
            if features is not None:
                X.append(features)
                y.append(label)
                
    return np.array(X), np.array(y)

# --- Execution ---
if __name__ == "__main__":
    X, y = process_urfd()

    print(f"Total Frames Extracted: {len(X)}")
    print(f"Fall Frames: {sum(y)}, Normal Frames: {len(y) - sum(y)}")

    if len(X) == 0:
        print("ERROR: No data extracted. Check your URFD_PATH!")
    else:
        # Split and Train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        print("Training SVM Classifier...")
        clf = SVC(kernel='rbf', probability=True)
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        print("\n--- Vision Model Evaluation ---")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Fall", "Fall"]))

        # Save Model
        joblib.dump(clf, "vision_fall_model.pkl")
        print("Model saved to vision_fall_model.pkl")