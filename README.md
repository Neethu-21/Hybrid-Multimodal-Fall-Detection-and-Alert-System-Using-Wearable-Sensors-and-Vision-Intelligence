# Multimodal Fall Detection System

A real-time multimodal fall detection system designed to detect falls
using vision-based and smartphone sensor-based approaches.

## Project Overview

The system uses two independent detection pipelines:

1. Indoor Vision-Based Fall Detection
2. Outdoor / Smartphone Sensor-Based Fall Detection

Both pipelines produce a fall prediction that can be used by the
real-time application for fall monitoring and alert generation.

---

## 1. Indoor Vision-Based Fall Detection

The indoor pipeline uses RGB video for fall detection.

### Pipeline

RGB Video
→ Frame Extraction
→ YOLO11n-Pose
→ 17 Body Keypoints
→ Temporal Pose Sequence
→ GRU
→ Fall Probability
→ Decision Threshold
→ Fall / No Fall

### YOLO11n-Pose

YOLO11n-Pose is used to detect the person and extract 17 body
keypoints from each video frame.

For each frame, the pose representation contains:

- X coordinate
- Y coordinate
- Keypoint confidence

The pose of a frame can be represented as:

P_t = {(x_i, y_i, c_i)}, i = 1, ..., 17

where:

- x_i = X coordinate of the keypoint
- y_i = Y coordinate of the keypoint
- c_i = confidence of the keypoint

### Temporal Pose Sequence

Keypoints extracted from consecutive frames are arranged in
chronological order to form a temporal sequence.

The system uses a 20-frame sequence for the vision model.

The sequence allows the model to learn changes in body posture over
time rather than relying on a single frame.

### GRU

A Gated Recurrent Unit (GRU) is used to learn temporal patterns in
the sequence of body keypoints.

It learns changes such as:

- Posture changes
- Sudden downward movement
- Change from upright to horizontal posture
- Motion followed by a relatively stationary posture

### Fall Prediction

The GRU produces a prediction score which is converted into a
probability using the sigmoid function:

p = 1 / (1 + e^(-x))

The current real-time vision application uses a decision threshold
of 0.35.

- Probability >= 0.35 → Fall Detected
- Probability < 0.35 → Normal

The system also uses multiple confirmation frames to reduce
unstable single-frame predictions.

---

## 2. Outdoor / Smartphone Sensor-Based Fall Detection

The outdoor pipeline uses smartphone sensor data.

Sensor data is collected using Sensor Logger and transmitted to the
PC through HTTP.

### Sensors Used

Three sensor sources are used:

#### Accelerometer

- X axis
- Y axis
- Z axis

#### Gravity

- X axis
- Y axis
- Z axis

#### Gyroscope

- X axis
- Y axis
- Z axis

Therefore, the sensor input contains:

3 + 3 + 3 = 9 channels

The sensor vector at time t can be represented as:

S_t = [s_t,1, s_t,2, ..., s_t,9]

### HTTP Data Transfer

Sensor Logger collects the smartphone sensor readings and sends
them to the PC using HTTP POST requests.

The sensor readings are carried in the HTTP request payload.

The application receives the incoming sensor data and processes it
for real-time fall detection.

### Sensor Preprocessing

The continuous sensor channels are standardized using training-data
statistics.

Normalization is performed using:

z = (x - μ) / σ

where:

- x = sensor value
- μ = mean calculated from the training data
- σ = standard deviation calculated from the training data
- z = normalized sensor value

### Temporal Window

Sensor readings are arranged into fixed-length temporal windows so
that the model can learn how sensor values change over time.

The current application uses a 200-sample sequence for the outdoor
sensor pipeline.

Each sample contains 9 sensor channels.

Therefore, the model input is conceptually:

200 × 9

### LSTM

A Long Short-Term Memory (LSTM) network is used to learn temporal
patterns in the sensor sequence.

The LSTM can learn patterns involving:

- Sudden acceleration changes
- Changes in device orientation
- Rotational movement
- Motion patterns associated with a fall

### Fall Prediction

The LSTM output is converted into a probability using the sigmoid
function:

p = 1 / (1 + e^(-x))

The current real-time outdoor pipeline uses a decision threshold
of 0.35.

- Probability >= 0.35 → Fall Detected
- Probability < 0.35 → Normal

---

## 3. Datasets

### URFD

The University of Rzeszow Fall Detection Dataset (URFD) is used for
the indoor vision-based fall detection pipeline.

The project processes:

- 40 ADL recordings
- 30 Fall recordings

Video frames are processed using YOLO11n-Pose to obtain 17 body
keypoints.

### SisFall

SisFall is used for sensor-based fall detection.

The project uses its 9-channel sensor representation consisting of
accelerometer, gyroscope and gravity data.

The dataset contains:

- 19 Activities of Daily Living (ADL)
- 15 fall activity types

### Smartphone Sensor Data

Additional real-world smartphone sensor recordings are collected
using Sensor Logger.

These recordings are used with a phone-compatible LSTM pipeline
for real-time sensor-based fall detection.

---

## 4. Models Used

### YOLO11n-Pose

Purpose:
- Person detection
- Human pose estimation
- Extraction of 17 body keypoints

### GRU

Purpose:
- Learn temporal changes in body posture
- Classify indoor pose sequences as fall or normal

### LSTM

Purpose:
- Learn temporal patterns in sensor data
- Classify sensor sequences as fall or normal

---

## 5. Fall Decision

The model output is converted into a probability using the sigmoid
function:

p = 1 / (1 + e^(-x))

The current application uses:

### Vision Threshold

0.35

### Outdoor Sensor Threshold

0.35

A prediction at or above the corresponding threshold is classified
as a fall.

---

## 6. Real-Time Application

The project includes a Streamlit-based application for real-time
fall detection.

The application supports:

- Indoor video-based detection
- Outdoor sensor-based detection
- Smartphone sensor streaming
- Fall probability display
- Fall / Normal status
- Real-time monitoring
- Alert generation

The smartphone sensor stream is received through an HTTP server
running on the PC.

---

## 7. Technology Stack

- Python
- Streamlit
- YOLO11n-Pose
- GRU
- LSTM
- TensorFlow
- PyTorch
- OpenCV
- NumPy
- Streamlit-WebRTC
- Sensor Logger
- HTTP

---

## 8. Project Structure

```text
Major Project/
│
├── app/
│   └── app.py
│
├── models/
│   ├── vision/
│   └── outdoor/
│
├── requirements.txt
├── README.md
└── .gitignore