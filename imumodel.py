import os
import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- Configuration ---
# Update this to the exact path printed by your kagglehub script
SISFALL_PATH = r"E:\FD\sisfall\SisFall_dataset"
WINDOW_SIZE = 100 # Example: 50Hz data * 2 seconds = 100 rows

def extract_features(window_df):
    """Calculates statistical features for a given window of IMU data."""
    # Assuming columns: Acc_X, Acc_Y, Acc_Z
    acc_x = window_df.iloc[:, 0].values
    acc_y = window_df.iloc[:, 1].values
    acc_z = window_df.iloc[:, 2].values
    
    # Calculate Signal Vector Magnitude (SVM)
    svm = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    
    # Extract features
    features = [
        np.mean(svm),
        np.std(svm),
        np.max(svm),
        np.min(svm),
        np.mean(acc_x),
        np.std(acc_x),
        np.mean(acc_y),
        np.std(acc_y),
        np.mean(acc_z),
        np.std(acc_z)
    ]
    return features

def process_sisfall():
    print("Processing SisFall Data...")
    X = []
    y = []
    
    # SisFall files are stored in subfolders, so recurse through all txt files
    files = glob.glob(os.path.join(SISFALL_PATH, "**", "*.txt"), recursive=True)
    print(f"Found {len(files)} SisFall files")
    
    for file in files:
        filename = os.path.basename(file)
        if filename.lower() == "desktop.ini":
            continue

        # Determine label from filename. (SisFall uses F for Fall, D for ADL)
        label = 1 if filename.startswith("F") else 0
        
        try:
            # Read the TXT files that use comma-separated values and end rows with semicolons
            df = pd.read_csv(file, sep=r"[\s,;]+", header=None, engine="python")
            df = df.dropna(axis=1, how="all")
            
            if len(df) < WINDOW_SIZE:
                continue
            
            # Create overlapping windows
            for i in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE // 2):
                window = df.iloc[i : i + WINDOW_SIZE]
                features = extract_features(window)
                X.append(features)
                y.append(label)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            
    return np.array(X), np.array(y)

# --- Execution ---
X, y = process_sisfall()

print(f"Total Windows Extracted: {len(X)}")
print(f"Falls: {sum(y)}, Normal: {len(y) - sum(y)}")

# Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("\n--- IMU Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["No Fall", "Fall"]))

# Save Model
joblib.dump(clf, "imu_fall_model.pkl")
print("Model saved to imu_fall_model.pkl")