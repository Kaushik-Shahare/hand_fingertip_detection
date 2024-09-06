# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from modules.draw_hand_landmark import draw_landmarks_on_image

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# STEP 3: Load the input image.
# image = mp.Image.create_from_file("image.jpg")
image = mp.Image.create_from_file("./images/hand_image2.jpeg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)

# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# print("detected hand landmarks:", detection_result)

# Convert the image to BGR format for OpenCV and display it
cv2.imshow("Hand Landmarks", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# Wait for a key press to close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()