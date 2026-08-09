# AI-driver-drowsiness-detection

# Driver Drowsiness Detection System

A real-time driver drowsiness detection system built using **Python, OpenCV, MediaPipe, and YOLO**.

The goal of this project is to monitor a driver's face and detect signs of drowsiness such as prolonged eye closure and yawning. When drowsiness is detected, the system provides visual warnings and an audible alert.

## Features

* Real-time face and driver monitoring
* Eye closure detection using facial landmarks
* Yawning detection
* Drowsiness scoring system
* Visual warning and alert indicators
* Audible alarm when drowsiness is detected
* Driver/mobile phone detection using YOLO
* Real-time webcam processing

## Technologies Used

* **Python**
* **OpenCV** – camera capture and image processing
* **MediaPipe Face Landmarker** – facial landmark detection
* **YOLOv8** – object detection
* **NumPy** – numerical calculations
* **Threading** – handling the alarm without blocking video processing
* **Winsound** – audible alerts on Windows

## How It Works

The system processes the webcam feed frame by frame.

```text
Webcam
   ↓
Frame Capture
   ↓
YOLO Object Detection
   ↓
Face Landmark Detection
   ↓
Eye & Mouth Analysis
   ↓
Drowsiness Score
   ↓
Warning / Alarm
```

### Eye Closure Detection

Facial landmarks around the eyes are used to calculate the **Eye Aspect Ratio (EAR)**.

If the EAR remains below a defined threshold for a certain duration, the system considers the driver's eyes to be closed and increases the drowsiness score.

### Yawning Detection

The system analyzes the distance between mouth landmarks to calculate the **Mouth Aspect Ratio (MAR)**.

Repeated or prolonged yawning contributes to the drowsiness score.

### Drowsiness Score

Instead of relying on a single condition, the system combines multiple indicators such as:

* Prolonged eye closure
* Yawning
* Other detected driver-related conditions

This helps make the system more reliable than simply checking whether the eyes are closed in a single frame.

## Models Used

### MediaPipe Face Landmarker

A pretrained MediaPipe face landmark model is used to locate facial landmarks in real time.

The landmarks are used for:

* Eye analysis
* Mouth analysis
* EAR calculation
* MAR calculation

### YOLOv8

A pretrained YOLOv8 model is used for object detection.

It is used to identify objects relevant to driver monitoring, such as a **mobile phone**.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

Install the required dependencies:

```bash
pip install opencv-python mediapipe numpy ultralytics
```

If required, install the remaining dependencies according to your Python environment.

## Required Model Files

The project requires the MediaPipe Face Landmarker model:

```text
face_landmarker.task
```

Place the model file in the project directory or update the model path in the Python script.

The YOLO model will be loaded through Ultralytics.

## Running the Project

Run the main Python file:

```bash
python main.py
```

Make sure your webcam is connected and accessible.

## Project Structure

```text
Driver-Drowsiness-Detection/
│
├── main.py
├── face_landmarker.task
├── yolov8n.pt
├── requirements.txt
├── README.md
└── assets/
```

> File names may differ depending on the final project structure.

## Limitations

The system is designed as a computer-vision prototype and can be affected by environmental conditions such as:

* Poor lighting
* Significant face obstruction
* Extreme camera angles
* Partial visibility of the driver's face
* Camera quality and frame rate

These conditions can affect facial landmark detection and therefore the accuracy of drowsiness detection.

## Future Improvements

Possible improvements include:

* Better performance in low-light environments
* Improved handling of glasses and face occlusion
* More robust multi-face handling
* Driver-specific calibration
* Additional drowsiness indicators
* Deployment on an embedded/edge device
* Integration with a vehicle safety system

## Disclaimer

This project is an educational/prototype implementation of a computer-vision-based driver monitoring system. It is **not intended to replace certified automotive safety systems**.

## Author

**Abdul Kafeel**

Built as a project to explore **Computer Vision, AI, and real-time driver monitoring** using Python.
