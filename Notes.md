# Points to keep in mind

- **There should be ample light on the hand for the camera to capture the image properly.**
- **The hand should cover atleast 40-60% of the image for accurate detection of each finger.**

# Steps to Create a Model to Detect Touch on Plain Paper

1. Calibrating for Paper Surface: You’ll need to map the hand landmarks relative to a specific surface (e.g., a sheet of paper) placed in a known position in the camera feed. This calibration will establish a boundary for the paper, such that the hand positions can be tracked relative to this surface.
2. Detecting Touch:

- Use the z coordinate of the landmarks to approximate the distance of the hand from the camera. When the z value of certain key points (such as fingertips) reaches a specific threshold, you can assume the hand is “touching” the surface.
- If the fingertips are within the boundary of the paper (which you can establish using the x and y coordinates), and the z-coordinate indicates proximity, then you can register a “touch.”

3. Modeling Touch Events:

- Create a classification model to detect “touch” events by labeling and training it with different hand positions, where some frames represent touch events (fingertips touching the paper) and others represent no-touch (hand hovering above the paper).

4. Machine Learning (Optional):

- You could use a simple rule-based approach for detecting touch (as described above).
- Alternatively, you can collect more data (both touch and no-touch scenarios) and train a small machine learning model (like a Support Vector Machine or Neural Network) to classify the hand states as “touch” or “no-touch” based on the landmark coordinates.

## Sample data of a hand landmark

- The data consist of 21 landmarks, each represented by a 3D coordinate (x, y, z) in the camera frame.

```python
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
image = mp.Image.create_from_file("hand_image1.jpeg")
detection_result = detector.detect(image)
# data = detection_result.hand_landmarks

data = [
    [  # List 1 of normalized landmarks
        {"x": 0.6230142116546631, "y": 0.2349795550107956, "z": -0.020784111693501472, "visibility": 0.0, "presence": 0.0},
        {"x": 0.5446934103965759, "y": 0.3060069680213928, "z": -0.038099661469459534, "visibility": 0.0, "presence": 0.0},
        {"x": 0.48813343048095703, "y": 0.38670122623443604, "z": -0.04997442290186882, "visibility": 0.0, "presence": 0.0},
        {"x": 0.44667232036590576, "y": 0.45446327328681946, "z": -0.0636899545788765, "visibility": 0.0, "presence": 0.0},
        {"x": 0.5978812575340271, "y": 0.43914124369621277, "z": -0.06609123200178146, "visibility": 0.0, "presence": 0.0},
        {"x": 0.5676192045211792, "y": 0.5881829261779785, "z": -0.09225496649742126, "visibility": 0.0, "presence": 0.0},
        {"x": 0.5617557764053345, "y": 0.6846891641616821, "z": -0.10763510316610336, "visibility": 0.0, "presence": 0.0},
        {"x": 0.5644922852516174, "y": 0.759198784828186, "z": -0.11795509606599808, "visibility": 0.0, "presence": 0.0},
        {"x": 0.6629648208618164, "y": 0.4655023515224457, "z": -0.0691547691822052, "visibility": 0.0, "presence": 0.0},
        {"x": 0.6478598713874817, "y": 0.65712970495224, "z": -0.10050265491008759, "visibility": 0.0, "presence": 0.0},
        # More data here...
    ],
    [  # List 2 of normalized landmarks
        {"x": 0.30110034346580505, "y": 0.04882150888442993, "z": -3.7128896224203345e-07, "visibility": 0.0, "presence": 0.0},
        {"x": 0.31040942668914795, "y": 0.10618855804204941, "z": -0.03728482127189636, "visibility": 0.0, "presence": 0.0},
        {"x": 0.28395476937294006, "y": 0.16940900683403015, "z": -0.06051388010382652, "visibility": 0.0, "presence": 0.0},
        {"x": 0.24574613571166992, "y": 0.22655218839645386, "z": -0.07453552633523941, "visibility": 0.0, "presence": 0.0},
        {"x": 0.21224063634872437, "y": 0.2809610366821289, "z": -0.08947129547595978, "visibility": 0.0, "presence": 0.0},
        # More data here...
    ]
]
```
