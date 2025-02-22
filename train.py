import cv2
import os

# Create folders for storing images
os.makedirs("dataset/real_faces", exist_ok=True)
os.makedirs("dataset/phone_faces", exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

category = input("Enter category (real_faces / phone_faces): ")
count = 0

while count < 100:  # Capture 100 images per category
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture", frame)

    # Save images in respective folders
    img_name = f"dataset/{category}/{count}.jpg"
    cv2.imwrite(img_name, frame)
    count += 1

    if cv2.waitKey(100) & 0xFF == 27:  # Press 'ESC' to stop
        break

cap.release()
cv2.destroyAllWindows()
