
import cv2
import mediapipe as mp
import numpy as np


# ==========================================
# INITIALIZE MEDIAPIPE POSE
# ==========================================

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ==========================================
# DETECT POSTURE
# ==========================================

def detect_posture(image):

    """
    Detect human body landmarks from an RGB image.

    Returns:
        annotated_image
        landmarks
        width
        height
    """

    if image is None:
        raise ValueError("Input image is empty.")

    # Ensure image is RGB
    image = np.asarray(image)

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image must have 3 RGB channels.")

    height, width = image.shape[:2]

    # Make image compatible with MediaPipe
    image = np.ascontiguousarray(
        image,
        dtype=np.uint8
    )

    # ======================================
    # MEDIAPIPE DETECTION
    # ======================================

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=0.5
    ) as pose:

        results = pose.process(image)

    # ======================================
    # CHECK DETECTION
    # ======================================

    if results.pose_landmarks is None:

        return None, None, width, height

    # ======================================
    # DRAW LANDMARKS
    # ======================================

    annotated_image = image.copy()

    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    # ======================================
    # RETURN RESULTS
    # ======================================

    return (
        annotated_image,
        results.pose_landmarks.landmark,
        width,
        height
    )


# ==========================================
# GET LANDMARK DETAILS
# ==========================================

def get_landmark_details(landmarks, width, height):

    """
    Return all 33 detected landmarks.
    """

    landmark_details = []

    for index, landmark in enumerate(landmarks):

        landmark_details.append({

            "id": index,

            "x": round(
                float(landmark.x * width), 2
            ),

            "y": round(
                float(landmark.y * height), 2
            ),

            "z": round(
                float(landmark.z), 4
            ),

            "visibility": round(
                float(landmark.visibility), 4
            )

        })

    return landmark_details
