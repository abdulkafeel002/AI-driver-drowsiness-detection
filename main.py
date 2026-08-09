import cv2
import mediapipe as mp
import numpy as np
import time
import threading
import winsound
from ultralytics import YOLO


yolo = YOLO("yolov8n.pt")

MODEL_PATH = "face_landmarker.task"

options = mp.tasks.vision.FaceLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_faces=1
)

landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)


LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

EAR_THRESHOLD = 0.20
MAR_THRESHOLD = 0.65
PHONE_CONF = 0.30

blink_count = 0
yawn_count = 0

closed_start = None
mouth_start = None
last_yawn = 0

scored_yawns = 1

score = 0
alarm_running = False

GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)


def dist(a, b):
    return np.sqrt(
        (a.x - b.x) ** 2 +
        (a.y - b.y) ** 2
    )


def calculate_ear(points, lm):
    p = [lm[i] for i in points]

    h = dist(p[0], p[3])

    if h == 0:
        return 0

    return (
        dist(p[1], p[5]) +
        dist(p[2], p[4])
    ) / (2 * h)


def calculate_mar(lm):
    horizontal = dist(
        lm[78],
        lm[308]
    )

    if horizontal == 0:
        return 0

    return dist(
        lm[13],
        lm[14]
    ) / horizontal


def beep():
    winsound.Beep(
        2500,
        1500
    )


def text(
    frame,
    value,
    pos,
    size=0.7,
    color=GREEN,
    thickness=2
):
    cv2.putText(
        frame,
        value,
        pos,
        cv2.FONT_HERSHEY_DUPLEX,
        size,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        value,
        pos,
        cv2.FONT_HERSHEY_DUPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_eye(frame, lm, points):
    h, w = frame.shape[:2]

    p = [
        (
            int(lm[i].x * w),
            int(lm[i].y * h)
        )
        for i in points
    ]

    for i in range(len(p)):
        cv2.line(
            frame,
            p[i],
            p[(i + 1) % len(p)],
            (255, 255, 255),
            2
        )


def draw_mouth(frame, lm):
    h, w = frame.shape[:2]

    left = (
        int(lm[78].x * w),
        int(lm[78].y * h)
    )

    top = (
        int(lm[13].x * w),
        int(lm[13].y * h)
    )

    right = (
        int(lm[308].x * w),
        int(lm[308].y * h)
    )

    cv2.line(
        frame,
        left,
        top,
        GREEN,
        2
    )

    cv2.line(
        frame,
        top,
        right,
        GREEN,
        2
    )

    cv2.line(
        frame,
        right,
        left,
        GREEN,
        2
    )


cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

window_name = "AI Driver Drowsiness Detection"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    window_name,
    1280,
    720
)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(
        frame,
        1
    )

    h, w = frame.shape[:2]

    results = yolo.predict(
        frame,
        imgsz=320,
        conf=PHONE_CONF,
        classes=[67],
        verbose=False
    )

    phones = 0

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            if confidence < PHONE_CONF:
                continue

            class_id = int(
                box.cls[0]
            )

            if class_id != 67:
                continue

            phones += 1

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            label = f"PHONE {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                YELLOW,
                2
            )

            text(
                frame,
                label,
                (
                    x1,
                    max(y1 - 10, 25)
                ),
                0.55,
                GREEN,
                2
            )

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = landmarker.detect_for_video(
        image,
        int(time.time() * 1000)
    )

    if result.face_landmarks:

        lm = result.face_landmarks[0]

        draw_eye(
            frame,
            lm,
            LEFT_EYE
        )

        draw_eye(
            frame,
            lm,
            RIGHT_EYE
        )

        draw_mouth(
            frame,
            lm
        )

        left_ear = calculate_ear(
            LEFT_EYE,
            lm
        )

        right_ear = calculate_ear(
            RIGHT_EYE,
            lm
        )

        eye_ratio = (
            left_ear +
            right_ear
        ) / 2

        if eye_ratio < EAR_THRESHOLD:

            eye_status = "CLOSED"

            if closed_start is None:
                closed_start = time.time()

            close_time = (
                time.time() -
                closed_start
            )

        else:

            eye_status = "OPEN"
            close_time = 0

            if closed_start is not None:
                blink_count += 1

            closed_start = None

        mouth_ratio = calculate_mar(lm)

        current_time = time.time()

        yawn_status = "NO YAWN"

        if mouth_ratio > MAR_THRESHOLD:

            if mouth_start is None:
                mouth_start = current_time

            open_time = (
                current_time -
                mouth_start
            )

            if (
                open_time >= 1
                and
                current_time - last_yawn >= 2
            ):

                yawn_count += 1
                last_yawn = current_time
                yawn_status = "YAWN"

            elif open_time > 0.3:

                yawn_status = "MOUTH OPEN"

        else:

            mouth_start = None

        change = 0

        if eye_status == "CLOSED":

            change += 1.5

            if close_time > 3:
                change += 2

        if yawn_count > scored_yawns:

            new_yawns = yawn_count - scored_yawns

            change += new_yawns * 5

            scored_yawns = yawn_count

        if change > 0:
            score += change
        else:
            score -= 0.8

        score = max(
            0,
            min(
                100,
                score
            )
        )

        if score >= 70:

            status = "SLEEP DETECTED"

            if not alarm_running:

                alarm_running = True

                threading.Thread(
                    target=beep,
                    daemon=True
                ).start()

        elif score >= 40:

            status = "DROWSY"

        else:

            status = "AWAKE"
            alarm_running = False

        text(
            frame,
            "EYE MONITOR",
            (25, 45),
            0.8,
            (255, 255, 255),
            2
        )

        text(
            frame,
            f"EAR: {eye_ratio:.3f}",
            (25, 78),
            0.65,
            (255, 255, 255),
            2
        )

        if eye_status == "CLOSED":

            text(
                frame,
                f"EYES: {eye_status}",
                (25, 110),
                0.65,
                RED,
                2
            )

        else:

            text(
                frame,
                f"EYES: {eye_status}",
                (25, 110),
                0.65,
                GREEN,
                2
            )

        text(
            frame,
            "FACE ANALYSIS",
            (25, 155),
            0.8,
            (255, 255, 255),
            2
        )

        text(
            frame,
            f"MAR: {mouth_ratio:.3f}",
            (25, 188),
            0.65,
            (255, 255, 255),
            2
        )

        if yawn_status == "YAWN":

            text(
                frame,
                f"YAWN: {yawn_status}",
                (25, 220),
                0.65,
                YELLOW,
                2
            )

        elif yawn_status == "MOUTH OPEN":

            text(
                frame,
                f"YAWN: {yawn_status}",
                (25, 220),
                0.65,
                YELLOW,
                2
            )

        else:

            text(
                frame,
                f"YAWN: {yawn_status}",
                (25, 220),
                0.65,
                GREEN,
                2
            )

        if phones > 0:

            text(
                frame,
                f"PHONES: {phones}",
                (25, 252),
                0.65,
                YELLOW,
                2
            )

        else:

            text(
                frame,
                f"PHONES: {phones}",
                (25, 252),
                0.65,
                GREEN,
                2
            )

        text(
            frame,
            "STATISTICS",
            (25, 303),
            0.8,
            (255, 255, 255),
            2
        )

        text(
            frame,
            f"BLINKS: {blink_count}",
            (25, 336),
            0.65,
            (255, 255, 255),
            2
        )

        text(
            frame,
            f"YAWNS: {yawn_count}",
            (25, 368),
            0.65,
            (255, 255, 255),
            2
        )

        if status == "SLEEP DETECTED":

            status_color = RED

        elif status == "DROWSY":

            status_color = YELLOW

        else:

            status_color = GREEN

        text(
            frame,
            f"STATUS: {status}",
            (400, 70),
            1.0,
            status_color,
            3
        )

        text(
            frame,
            f"SCORE: {int(score)}",
            (400, 115),
            0.85,
            status_color,
            2
        )

    else:

        text(
            frame,
            "FACE NOT DETECTED",
            (400, 70),
            0.9,
            RED,
            2
        )

    if phones > 0:

        text(
            frame,
            "WARNING: DO NOT USE MOBILE PHONE",
            (
                max(
                    20,
                    w // 2 - 300
                ),
                h - 40
            ),
            0.8,
            YELLOW,
            2
        )

    cv2.imshow(
        window_name,
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
