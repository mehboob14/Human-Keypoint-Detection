import cv2
import mediapipe as mp
from typing import Tuple

#constants for drawing
LANDMARK_RADIUS = 5
LANDMARK_COLOR = (0, 0, 255)  
CONNECTION_COLOR = (0, 255, 255)  
CONNECTION_THICKNESS = 5
BOUNDING_BOX_COLOR = (255, 0, 0)  
BOUNDING_BOX_THICKNESS = 3

class PoseEstimator:
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=detection_confidence,
                                      min_tracking_confidence=tracking_confidence)
        self.mp_drawing = mp.solutions.drawing_utils

    def process_image(self, image_path: str) -> Tuple[bool, any, any]:
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)
        return results.pose_landmarks is not None, results, image

class LandmarkAnalyzer:
    def __init__(self, visibility_threshold=0.5):
        self.critical_landmarks = [11, 12, 23, 24, 25, 26]
        self.visibility_threshold = visibility_threshold

    def check_full_body(self, landmarks) -> bool:
        return all(landmarks.landmark[lm].visibility > self.visibility_threshold for lm in self.critical_landmarks)

    def get_bounding_box(self, landmarks, width, height):
        x_coords = [lm.x for lm in landmarks.landmark]
        y_coords = [lm.y for lm in landmarks.landmark]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return (int(x_min * width), int(y_min * height),
                int(x_max * width), int(y_max * height))

class PoseVisualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose

    def draw_landmarks(self, image, landmarks):
        self.mp_drawing.draw_landmarks(
            image,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=LANDMARK_COLOR, thickness=-1, circle_radius=LANDMARK_RADIUS),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=CONNECTION_COLOR, thickness=CONNECTION_THICKNESS)
        )

    def draw_bounding_box(self, image, box):
        x_min, y_min, x_max, y_max = box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), BOUNDING_BOX_COLOR, BOUNDING_BOX_THICKNESS)

class PoseDetectionApp:
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.estimator = PoseEstimator()
        self.analyzer = LandmarkAnalyzer()
        self.visualizer = PoseVisualizer()

    def run(self):
        detected, results, image = self.estimator.process_image(self.image_path)

        if not detected:
            print("No human detected.")
            return

        height, width, _ = image.shape
        landmarks = results.pose_landmarks
        box = self.analyzer.get_bounding_box(landmarks, width, height)

        body_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / (width * height)

        if body_ratio > 0.5 and self.analyzer.check_full_body(landmarks):
            print("Full human body detected.")
        elif body_ratio <= 0.5:
            print("Partial human body detected.")
        else:
            print("Human detected but not fully visible.")

        self.visualizer.draw_landmarks(image, landmarks)
        self.visualizer.draw_bounding_box(image, box)

        output_path = "output_with_pose.jpg"
        cv2.imwrite(output_path, image)
        print(f"Output saved to {output_path}")
        self.display_image(output_path)

    def display_image(self, path):
        image = cv2.imread(path)
        cv2.imshow('Pose Detection', image)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PoseDetectionApp("./Resources/city.jpeg")
    app.run()
