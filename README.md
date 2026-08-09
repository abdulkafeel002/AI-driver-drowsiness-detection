# AI Driver Drowsiness Detection System

A real-time driver drowsiness detection system built with **Python, OpenCV, MediaPipe, and YOLOv8**.

The system uses a webcam to monitor the driver's facial features and identify signs associated with drowsiness, such as prolonged eye closure and yawning. A scoring mechanism combines these indicators and triggers warnings when drowsiness reaches a certain level.

## Features

* Real-time webcam-based monitoring
* Facial landmark detection
* Eye closure and blink detection
* Yawning detection
* Drowsiness scoring
* Visual warnings and alerts
* Audible alarm
* Mobile phone detection using YOLOv8
* Real-time processing using computer vision

## Technologies Used

* **Python**
* **OpenCV** – video capture and image processing
* **MediaPipe Face Landmarker** – facial landmark detection
* **YOLOv8** – object detection
* **NumPy** – numerical calculations
* **Ultralytics** – YOLO implementation
* **Threading** – non-blocking alarm handling
* **Winsound** – audio alerts on Windows

## How It Works

The system processes the webcam feed continuously:

```text
Webcam
   ↓
Frame Capture
   ↓
YOLO Object Detection
   ↓
MediaPipe Face Landmarks
   ↓
Eye & Mouth Analysis
   ↓
Drowsiness Score
   ↓
Warning / Alarm
```

### 1. Face Landmark Detection

MediaPipe Face Landmarker is used to detect facial landmarks in real time.

These landmarks provide the coordinates required to analyze the driver's eyes and mouth.

### 2. Eye Closure Detection

The system uses eye landmarks to calculate the **Eye Aspect Ratio (EAR)**.

When the eyes remain closed for a sufficient period, the system treats this as a potential sign of drowsiness and increases the drowsiness score.

### 3. Yawning Detection

Mouth landmarks are used to calculate the **Mouth Aspect Ratio (MAR)**.

Yawning events are monitored over time rather than treating a single frame as proof of drowsiness.

### 4. Drowsiness Scoring

Multiple indicators are combined into a drowsiness score.

This allows the system to consider patterns such as:

* Prolonged eye closure
* Repeated yawning
* Other detected driver conditions

When the score reaches the defined alert level, the system provides a warning and activates the alarm.

### 5. Object Detection

A pretrained **YOLOv8** model is used to detect objects in the camera feed, including mobile phones.

This adds an additional driver-monitoring capability to the system.

## Models

This project uses pretrained models rather than training a drowsiness model from scratch.

### MediaPipe Face Landmarker

Used for real-time facial landmark detection and analysis of:

* Eyes
* Mouth
* Facial geometry

### YOLOv8

A pretrained YOLOv8 model is used for real-time object detection.

## Installation

Clone the repository:

```bash
git clone https://github.com/abdulkafeel002/AI-driver-drowsiness-detection.git
cd AI-driver-drowsiness-detection
```

Install the required Python packages:

```bash
pip install opencv-python mediapipe numpy ultralytics
```

If a `requirements.txt` file is provided in the repository, installing from it is recommended:

```bash
pip install -r requirements.txt
```

## Required Files

The project requires the MediaPipe Face Landmarker model:

```text
face_landmarker.task
```

Place the model file in the expected project directory.

The YOLO model used by the project should also be available at the path configured in the Python code.

## Running the Project

Run the main Python script:

```bash
python main.py
```

Make sure your webcam is connected and accessible.

## Project Structure

```text
AI-driver-drowsiness-detection/
│
├── main.py
├── face_landmarker.task
├── yolov8n.pt
├── requirements.txt
├── README.md
└── ...
```

The exact file structure may vary depending on the current version of the project.

## Limitations

The system is a computer-vision prototype and its performance can be affected by environmental conditions, including:

* Poor lighting
* Face obstruction
* Significant camera angles
* Partial face visibility
* Glasses and reflections
* Low camera frame rates
* Camera quality

These conditions can affect facial landmark detection and object detection.

## Future Improvements

* Improve performance in low-light environments
* Improve robustness with glasses and partial face occlusion
* Add adaptive user-specific calibration
* Improve false-positive handling
* Optimize performance for low-end hardware
* Deploy the system on an edge device
* Integrate with vehicle hardware for additional alerts

## Disclaimer

This project is an educational computer-vision prototype and is **not intended to replace certified automotive driver-monitoring or safety systems**.

## Author

**Abdul Kafeel**

Built to explore **Computer Vision, AI, and real-time driver monitoring using Python**.
