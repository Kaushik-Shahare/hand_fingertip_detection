import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from modules.draw_hand_landmark import draw_landmarks_on_image
from modules.draw_fingertips import draw_fingertips

# Create a HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# Capture video from the default camera (device index 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Process video frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert the frame to the format expected by MediaPipe (RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert to MediaPipe Image object
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Detect hand landmarks from the video frame
    detection_result = detector.detect(image)

    # Process the classification result (draw landmarks)
    annotated_frame = draw_landmarks_on_image(image.numpy_view(), detection_result)

    # Process the classification result (draw fingertips)
    annotated_frame_with_fingertips = draw_fingertips(annotated_frame, detection_result)

    # Convert the annotated image to BGR format for OpenCV and display it

    # cv2.imshow("Hand Landmarks", cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
    cv2.imshow("Hand Landmarks", cv2.cvtColor(annotated_frame_with_fingertips, cv2.COLOR_RGB2BGR))

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()