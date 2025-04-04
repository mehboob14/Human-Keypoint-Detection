class LandmarkAnalyzer:
    def __init__(self):
        self.required_landmarks = [11, 12, 23, 24, 25, 26]

    def check_full_body(self, landmarks):
        for lm in self.required_landmarks:
            if landmarks.landmark[lm].visibility < 0.5:
                return False
        return True

    def get_bounding_box(self, landmarks, width, height):
        x_min = min([landmarks.landmark[i].x for i in range(len(landmarks.landmark))])
        y_min = min([landmarks.landmark[i].y for i in range(len(landmarks.landmark))])
        x_max = max([landmarks.landmark[i].x for i in range(len(landmarks.landmark))])
        y_max = max([landmarks.landmark[i].y for i in range(len(landmarks.landmark))])

        x_min_pixel = int(x_min * width)
        y_min_pixel = int(y_min * height)
        x_max_pixel = int(x_max * width)
        y_max_pixel = int(y_max * height)

        return (x_min_pixel, y_min_pixel, x_max_pixel, y_max_pixel)
