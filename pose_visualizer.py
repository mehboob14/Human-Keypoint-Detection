import cv2
import mediapipe as mp

class PoseVisualizer:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose #ref to pose modeule   

    def draw_landmarks(self, image, pose_landmarks, bounding_box):
        self.mp_drawing.draw_landmarks(
            image, pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=3, circle_radius=5),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=3, circle_radius=2)
        )
        x_min, y_min, x_max, y_max = bounding_box
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
        return image
