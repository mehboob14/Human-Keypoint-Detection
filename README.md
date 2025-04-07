

# 🕺 Human Pose Detection using MediaPipe

A lightweight Python script to detect human pose landmarks in an image using [MediaPipe](https://google.github.io/mediapipe/) and OpenCV. It analyzes keypoints, checks body visibility, calculates bounding box, and visualizes it all in one go!

---

## 📦 Installation

Make sure you have Python 3.7+ installed.

### Step 1: Create virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate 
Step 2: Install dependencies



1. pose_estimator #this class is responsible for detecting human poses in an image using MediaPipe Pose model

2. landmark_analyer # this class is responsible for analyzing detected pose landmarks and determining if full human body is visible and calculating bounding box around detected landmarks.

3. pose_visu #this class handles the visualization of pose landmarks on the image. it draws the detected landmarks and a bounding box around the detected human body.

4. main Program ->handler #This script ties everything together, also logic stuff e.g decides if the full body or partial body is detected 

