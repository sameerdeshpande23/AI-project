import cv2
import face_recognition
import numpy as np
import dlib
import time
import random
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import winsound
import os

# Load YOLO Model for Phone Detection
model = YOLO("yolov8n.pt")

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Load Face Detector and Landmark Predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Excel File for Attendance Storage
attendance_file = "attendance.xlsx"

# Student Database
student_data = {}
known_face_encodings = []
known_face_names = {}
student_credentials = {}

# Load Student Faces
def load_known_faces():
    global known_face_encodings, known_face_names, student_data
    student_data.clear()
    known_face_encodings.clear()
    known_face_names.clear()
    
    if not os.path.exists("students"):
        os.makedirs("students")

    for file in os.listdir("students"):
        if file.endswith(".jpg") or file.endswith(".png"):
            name = os.path.splitext(file)[0]
            image_path = os.path.join("students", file)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names[encoding[0].tobytes()] = name
                student_data[name] = {"image": image_path}
                print(f"✅ Loaded {name}")
            else:
                print(f"⚠ Warning: No face detected in {image_path}")

load_known_faces()

# Function to Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Function to Calculate Face Distance
def calculate_distance(face_width_pixels, reference_width=14, focal_length=600):
    return (reference_width * focal_length) / face_width_pixels

# Function to Detect Phones (YOLO every 2 frames)
def detect_phone(frame, frame_count):
    if frame_count % 2 != 0:
        return False
    results = model(frame)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls == 67 and conf > 0.3:
                print("❌ Phone Detected! Process Rejected!")
                cv2.putText(frame, "Phone Detected! Process Rejected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Verification", frame)
                cv2.waitKey(2000)
                return True
    return False

# Function to Register a New Face
def register_new_face(frame):
    name = input("Enter Name: ")
    image_path = f"students/{name}.jpg"
    cv2.imwrite(image_path, frame)
    print(f"✅ New face registered for {name}. Restarting face recognition...")
    load_known_faces()

# Verification and Face Recognition
def verification_and_recognition():
    required_blinks = random.randint(2, 3)
    completed_blinks = 0
    print(f"🔹 Blink your eyes {required_blinks} times!")

    start_time = time.time()
    consecutive_closed_frames = 0
    EAR_BUFFER = []
    frame_count = 0
    baseline_ear = None
    recognized_name = None

    while completed_blinks < required_blinks and time.time() - start_time < 60:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Phone Detection
        if detect_phone(frame, frame_count):
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        # Face Recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                recognized_name = known_face_names[known_face_encodings[matches.index(True)].tobytes()]
                print(f"✅ Face Recognized: {recognized_name}")

        if len(faces) == 1:
            shape = predictor(gray, faces[0])
            left_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(shape.part(i).x, shape.part(i).y) for i in range(42, 48)])

            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if time.time() - start_time < 1.5:
                EAR_BUFFER.append(avg_ear)
                continue

            if baseline_ear is None:
                baseline_ear = np.median(EAR_BUFFER)

            blink_threshold = max(baseline_ear * 0.75, 0.19)

            if avg_ear < blink_threshold:
                consecutive_closed_frames += 1
            else:
                if consecutive_closed_frames >= random.randint(4, 5):  # Randomized frame count
                    completed_blinks += 1
                    winsound.Beep(1000, 500)
                    print(f"✔ Blink {completed_blinks}/{required_blinks} detected!")
                consecutive_closed_frames = 0

            # Face Distance Calculation
            face_width = faces[0].width()
            face_distance = calculate_distance(face_width)

            if face_distance < 70:
                print("❌ Too Close! Move Back!")
                cv2.putText(frame, "Move Back! Too Close!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Verification", frame)
                cv2.waitKey(500)
                continue

            elapsed_time = int(30 - (time.time() - start_time))
            cv2.putText(frame, f"Blinks: {completed_blinks}/{required_blinks} | Time Left: {elapsed_time}s",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Verification", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    if completed_blinks >= required_blinks:
        if recognized_name:
            print(f"✅ Marking Attendance for {recognized_name}...")
            save_attendance_to_excel(recognized_name)
        else:
            print("⚠ Face Not Recognized! Please Register.")
            register_new_face(frame)
    else:
        print("❌ Verification Failed!")

    cap.release()
    cv2.destroyAllWindows()

# Save Attendance
def save_attendance_to_excel(name):
    now = datetime.now()
    new_entry = pd.DataFrame([[name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")]],
                             columns=["Name", "Date", "Time"])
    try:
        existing_data = pd.read_excel(attendance_file)
        updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
    except FileNotFoundError:
        updated_data = new_entry
    updated_data.to_excel(attendance_file, index=False)

verification_and_recognition()