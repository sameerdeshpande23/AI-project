import cv2
import face_recognition

# Initialize webcam
cap = cv2.VideoCapture(0)

# Load and encode the reference image (simage.jpg)
img2 = cv2.imread("image.jpg")
if img2 is None:
    print("Error: Could not read 'simage.jpg'. Check file path.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
encodings2 = face_recognition.face_encodings(rgb_img2)
if not encodings2:
    print("Error: No face detected in 'simage.jpg'.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

img_encoding2 = encodings2[0]

while True:
    ret, img = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Convert frame to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces and encode
    encodings = face_recognition.face_encodings(rgb_img)
    if encodings:
        img_encoding = encodings[0]
        result = face_recognition.compare_faces([img_encoding2], img_encoding)
        print("Match Result: ", result)

    # Display the frame
    cv2.imshow('Live Video', img)

    # Exit on pressing 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
