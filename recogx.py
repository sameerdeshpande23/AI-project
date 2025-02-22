import cv2
import face_recognition
import numpy as np
import dlib
import time
import random
import winsound
import pandas as pd
from ultralytics import YOLO
from datetime import datetime

# Load YOLO Model for Phone Detection
model = YOLO("yolov8n.pt")

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Load Face Detector and Landmark Predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Store Face Encodings & Names
known_face_encodings = []
known_face_names = []

# Function to Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])  
    B = np.linalg.norm(eye[2] - eye[4])  
    C = np.linalg.norm(eye[0] - eye[3])  
    return (A + B) / (2.0 * C)  

# Load Student Images for Face Recognition
def load_known_faces():
    global known_face_encodings, known_face_names
    
    student_images = {
        "sameer": "students/Sameer.jpg",
        "kiran": "students/Kiran.jpg"
    }
    
    for name, img_path in student_images.items():
        try:
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            
            if encoding:  
                known_face_encodings.append(encoding[0])
                known_face_names.append(name)
                print(f"✅ {name} added successfully!")
            else:
                print(f"⚠ Warning: No face detected in {img_path}. Please check the image.")

        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")

load_known_faces()  # Load faces at startup

# Attendance Table (Pandas DataFrame)
attendance_data = pd.DataFrame(columns=["Name", "Date", "Time"])

# Phone Detection (Runs Every 2 Frames)
def detect_phone(frame, frame_count):
    if frame_count % 2 != 0:  
        return False  

    results = model(frame)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])  
            conf = float(box.conf[0])  
            if cls in [67, 28, 15] and conf > 0.2:  
                print("❌ Phone Detected! Process Rejected!")
                cv2.putText(frame, "Phone Detected! Process Rejected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Verification", frame)
                cv2.waitKey(500)  
                return True  
    return False  

# Highly Optimized Eye Blink Detection
def eye_blink_verification():
    required_blinks = random.randint(4, 5)
    completed_blinks = 0
    print(f"🔹 Blink your eyes {required_blinks} times!")

    start_time = time.time()
    consecutive_closed_frames = 0
    EAR_BUFFER = []  
    frame_count = 0  
    baseline_ear = None

    while completed_blinks < required_blinks and time.time() - start_time < 30:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        if detect_phone(frame, frame_count):  
            return False  

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

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

            blink_threshold = max(baseline_ear * 0.75, 0.18)  

            if avg_ear < blink_threshold:
                consecutive_closed_frames += 1
            else:
                if consecutive_closed_frames >= random.randint(6, 8):  
                    completed_blinks += 1
                    winsound.Beep(1000, 500)
                    print(f"✔ Blink {completed_blinks}/{required_blinks} detected!")
                consecutive_closed_frames = 0  

            x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            elapsed_time = int(30 - (time.time() - start_time))
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

# Face Recognition & Attendance Marking
def face_recognition_attendance():
    print("🔹 Switching to Face Recognition...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"

            if True in matches:
                matched_index = matches.index(True)
                name = known_face_names[matched_index]

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time_str = now.strftime("%H:%M:%S")

                global attendance_data
                new_entry = pd.DataFrame([{"Name": name, "Date": date, "Time": time_str}])
                attendance_data = pd.concat([attendance_data, new_entry], ignore_index=True)

                print(f"✅ Attendance Marked: {name} at {time_str}")

                cap.release()
                cv2.destroyAllWindows()
                return

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

# Run System
if eye_blink_verification():
    face_recognition_attendance()

cap.release()
cv2.destroyAllWindows()
