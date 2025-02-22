import cv2
import face_recognition
import numpy as np
import dlib
import time
import random
from ultralytics import YOLO
import winsound

# Load YOLO Model for Phone Detection
model = YOLO("yolov8n.pt")  # YOLOv8 Nano model

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Load Face Detector and Landmark Predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Optimized Phone Detection (Runs Every 10 Frames)
def detect_phone(frame, frame_count):
    if frame_count % 2 != 0:  # Run YOLO every 10 frames for efficiency
        return False

    results = model(frame)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])  
            conf = float(box.conf[0])  
            print(cls)
            if cls == 67 or cls == 28 or cls == 15 and conf > 0.2 :  # Class ID 67 = "cell phone"
                print("❌ Phone Detected! Process Rejected!")
                cv2.putText(frame, "Phone Detected! Process Rejected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Verification", frame)
                cv2.waitKey(2000)  
                return True  
    return False  

# Highly Optimized Eye Blink Detection
def eye_blink_verification():
    required_blinks = random.randint(3,4)
    completed_blinks = 0
    print(f"🔹 Blink your eyes {required_blinks} times!")

    start_time = time.time()
    consecutive_closed_frames = 0
    EAR_BUFFER = []  # Stores EAR values dynamically
    frame_count = 0  # YOLO Optimization
    baseline_ear = None

    while completed_blinks < required_blinks and time.time() - start_time < 30:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # **Phone Detection (Runs every 10 frames)**
        if detect_phone(frame, frame_count):
            return False  # Stop if phone detected

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        if len(faces) == 1:
            shape = predictor(gray, faces[0])

            left_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # **Dynamically Adjust EAR Baseline (Ignore First 1.5s)**
            if time.time() - start_time < 1.5:
                EAR_BUFFER.append(avg_ear)
                continue  # Skip early frames to establish a stable baseline

            if baseline_ear is None:
                baseline_ear = np.median(EAR_BUFFER)  # Set initial baseline

            # **Adaptive Blink Threshold**
            blink_threshold = max(baseline_ear * 0.75, 0.18)  # Ensures threshold doesn't drop too low

            # **Smooth Blink Detection**
            if avg_ear < blink_threshold:
                consecutive_closed_frames += 1
            else:
                print(random.randint(5,7))
                if consecutive_closed_frames >= random.randint(5,7):  # Only count if closed for 3+ frames
                    completed_blinks += 1
                    winsound.Beep(1000,500)
                    print(f"✔ Blink {completed_blinks}/{required_blinks} detected!")
                consecutive_closed_frames = 0  

            # **Draw face rectangle**
            x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # **Display blink count**
            elapsed_time = int(15 - (time.time() - start_time))
            cv2.putText(frame, f"Blinks: {completed_blinks}/{required_blinks} | Time Left: {elapsed_time}s",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        else:
            cv2.putText(frame, "⚠ Face not detected!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Eye Blink Verification", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    if completed_blinks >= required_blinks:
        print("✅ Eye Blink Verification Successful!")
        return True

    print("❌ Verification Rejected!")
    return False

# Main Execution
if eye_blink_verification():
    print("✅ Eye Blink Verified! Moving to Next Step.")

print("🔹 Process Completed!")
cap.release()
cv2.destroyAllWindows()
