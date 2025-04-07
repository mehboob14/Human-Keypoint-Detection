import cv2
from pose_estimator import PoseEstimator
from landmark_analyzer import LandmarkAnalyzer
from pose_visualizer import PoseVisualizer
import sys

def main(image_path):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    estimator = PoseEstimator()  
    analyzer = LandmarkAnalyzer()
    visualizer = PoseVisualizer()

    results = estimator.detect_pose(image)

    if results.pose_landmarks:
        bbox = analyzer.get_bounding_box(results.pose_landmarks, width, height)
        area_ratio = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / (width * height)
        is_full = analyzer.check_full_body(results.pose_landmarks)

        if area_ratio >= 0.4 and is_full:
            print("Full human detected.")
        elif area_ratio < 0.4:
            print(" human detected.")
        else:
            print("Human detected but body is not fully visible.")

        output_image = visualizer.draw_landmarks(image, results.pose_landmarks, bbox)
        cv2.imwrite("keypoints_overlay.jpg", output_image)
        # cv2.imshow("Keypoints_image", output_image)
        # cv2.waitKey(1000) 
        # cv2.destroyAllWindows()
    else:
        print("No human detected.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <image_path>")
    else:
        main(sys.argv[1])