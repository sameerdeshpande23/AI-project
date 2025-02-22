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

# Load YOLO Model for Phone Detection
model = YOLO("yolov8n.pt")  # YOLOv8 Nano model

# Initialize Webcam
cap = cv2.VideoCapture(0)

# Load Face Detector and Landmark Predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Excel File for Attendance Storage
attendance_file = "attendance.xlsx"

# Student Database (Face Linked with Register Number & Course)
student_data = {
    "sameer": {"image": "students/Sameer.jpg", "register_no": "21CS101", "course": "Computer Science"},
    "kiran": {"image": "students/Kiran.jpg", "register_no": "21IT202", "course": "Information Technology"}
}

# Storage for Encodings & Credentials
known_face_encodings = []
known_face_names = []
student_credentials = {}

# Function to Load Student Images & Data
def load_known_faces():
    global known_face_encodings, known_face_names, student_credentials
    
    for name, details in student_data.items():
        try:
            image = face_recognition.load_image_file(details["image"])
            encoding = face_recognition.face_encodings(image)

            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(name)
                student_credentials[name] = {
                    "register_no": details["register_no"],
                    "course": details["course"]
                }
                print(f"✅ {name} added | Reg No: {details['register_no']} | Course: {details['course']}")
            else:
                print(f"⚠ Warning: No face detected in {details['image']}. Check the image.")

        except Exception as e:
            print(f"❌ Error loading {details['image']}: {e}")

load_known_faces()  # Load at startup

# Function to Calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Function to Detect Phones (YOLO)
def detect_phone(frame, frame_count):
    if frame_count % 5 != 0:  # Run YOLO every 5 frames for efficiency
        return False

    results = model(frame)
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])  
            conf = float(box.conf[0])  
            if cls == 67 and conf > 0.3:  # Class ID 67 = "cell phone"
                print("❌ Phone Detected! Process Rejected!")
                cv2.putText(frame, "Phone Detected! Process Rejected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow("Verification", frame)
                cv2.waitKey(2000)  
                return True  
    return False  

# Eye Blink Verification
def eye_blink_verification():
    required_blinks = random.randint(5,6)
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

        # Phone Detection (Every 5 Frames)
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
                if consecutive_closed_frames >= random.randint(3,5):
                    completed_blinks += 1
                    winsound.Beep(1000, 500)
                    print(f"✔ Blink {completed_blinks}/{required_blinks} detected!")
                consecutive_closed_frames = 0  

            x, y, w, h = (faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height())
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            elapsed_time = int(30 - (time.time() - start_time))
            cv2.putText(frame, f"Blinks: {completed_blinks}/{required_blinks} | Time Left: {elapsed_time}s",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Eye Blink Verification", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    if completed_blinks >= required_blinks:
        print("✅ Eye Blink Verification Successful!")
        return True

    print("❌ Verification Rejected!")
    return False

# Save Attendance Data to Excel
def save_attendance_to_excel(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    register_no = student_credentials.get(name, {}).get("register_no", "Unknown")
    course_name = student_credentials.get(name, {}).get("course", "Unknown")

    new_entry = pd.DataFrame([{
        "Name": name,
        "Register No": register_no,
        "Course": course_name,
        "Date": date,
        "Time": time_str
    }])

    try:
        existing_data = pd.read_excel(attendance_file)
        updated_data = pd.concat([existing_data, new_entry], ignore_index=True)
    except FileNotFoundError:
        updated_data = new_entry  

    updated_data.to_excel(attendance_file, index=False)  
    print(f"✅ Attendance Marked for {name}")

# Face Recognition Attendance (Print Only Once)
def face_recognition_attendance():
    print("🔹 Switching to Face Recognition...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                name = known_face_names[matches.index(True)]
                print(f"✅ Face Recognized: {name}")  # Print recognized face only once
                save_attendance_to_excel(name)  
                
                cap.release()  # Release camera after recognition
                cv2.destroyAllWindows()  # Close all windows
                return  # Exit the function immediately

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

if eye_blink_verification():
    face_recognition_attendance()

cap.release()
cv2.destroyAllWindows()