import cv2
import mediapipe as mp

class PoseEstimator:
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )

    def detect_pose(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.pose.process(image_rgb)

        # landmarks = results.pose_landmarks
        # landmark_nose = landmarks.landmark[0]
        # print(f"nose coordinates: ({landmark_nose.x}, {landmark_nose.y}, {landmark_nose.z}), Visibility: {landmark_nose.visibility}")
        return results


