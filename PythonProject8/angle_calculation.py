
import math
import numpy as np
import mediapipe as mp


P = mp.solutions.pose.PoseLandmark


# ==========================================
# MATHEMATICAL ANGLE CALCULATION
# ==========================================

def calculate_angle(a, b, c):

    """
    Calculate the angle at point B.

    Uses normalized 2D image coordinates.
    Returns degrees between 0 and 180.
    """

    a = np.array(a[:2], dtype=float)
    b = np.array(b[:2], dtype=float)
    c = np.array(c[:2], dtype=float)

    ba = a - b
    bc = c - b

    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)

    if mag_ba < 1e-8 or mag_bc < 1e-8:
        return None

    cosine = np.dot(ba, bc) / (mag_ba * mag_bc)

    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.degrees(np.arccos(cosine))

    return round(float(angle), 2)


# ==========================================
# LANDMARK HELPERS
# ==========================================

def get_point(landmarks, landmark_id):

    landmark = landmarks[landmark_id.value]

    return np.array([
        landmark.x,
        landmark.y,
        landmark.z
    ], dtype=float)


def get_visibility(landmarks, landmark_id):

    return float(
        landmarks[landmark_id.value].visibility
    )


def midpoint(a, b):

    return (a + b) / 2


def points_visible(landmarks, landmark_ids, threshold=0.5):

    return all(
        get_visibility(landmarks, landmark_id) >= threshold
        for landmark_id in landmark_ids
    )


# ==========================================
# JOINT ANGLES
# ==========================================

def get_joint_angles(landmarks):

    joints = {

        "left_elbow": (
            P.LEFT_SHOULDER,
            P.LEFT_ELBOW,
            P.LEFT_WRIST
        ),

        "right_elbow": (
            P.RIGHT_SHOULDER,
            P.RIGHT_ELBOW,
            P.RIGHT_WRIST
        ),

        "left_shoulder": (
            P.LEFT_ELBOW,
            P.LEFT_SHOULDER,
            P.LEFT_HIP
        ),

        "right_shoulder": (
            P.RIGHT_ELBOW,
            P.RIGHT_SHOULDER,
            P.RIGHT_HIP
        ),

        "left_hip": (
            P.LEFT_SHOULDER,
            P.LEFT_HIP,
            P.LEFT_KNEE
        ),

        "right_hip": (
            P.RIGHT_SHOULDER,
            P.RIGHT_HIP,
            P.RIGHT_KNEE
        ),

        "left_knee": (
            P.LEFT_HIP,
            P.LEFT_KNEE,
            P.LEFT_ANKLE
        ),

        "right_knee": (
            P.RIGHT_HIP,
            P.RIGHT_KNEE,
            P.RIGHT_ANKLE
        ),

        "left_ankle": (
            P.LEFT_KNEE,
            P.LEFT_ANKLE,
            P.LEFT_FOOT_INDEX
        ),

        "right_ankle": (
            P.RIGHT_KNEE,
            P.RIGHT_ANKLE,
            P.RIGHT_FOOT_INDEX
        )
    }

    results = {}

    for name, ids in joints.items():

        if not points_visible(landmarks, ids):
            results[name] = None
            continue

        a = get_point(landmarks, ids[0])
        b = get_point(landmarks, ids[1])
        c = get_point(landmarks, ids[2])

        results[name] = calculate_angle(a, b, c)

    return results


# ==========================================
# SHOULDER AND HIP ALIGNMENT
# ==========================================

def calculate_alignment(landmarks):

    pairs = {

        "shoulders": (
            P.LEFT_SHOULDER,
            P.RIGHT_SHOULDER
        ),

        "hips": (
            P.LEFT_HIP,
            P.RIGHT_HIP
        )
    }

    results = {}

    for name, (left_id, right_id) in pairs.items():

        if not points_visible(
            landmarks, (left_id, right_id)
        ):
            results[name] = None
            continue

        left = get_point(landmarks, left_id)
        right = get_point(landmarks, right_id)

        dx = right[0] - left[0]
        dy = right[1] - left[1]

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        results[name] = {

            "left_right_height_difference": round(
                float(abs(dy)), 4
            ),

            "line_inclination_degrees": round(
                float(angle), 2
            )
        }

    return results


# ==========================================
# BODY SYMMETRY
# ==========================================

def calculate_symmetry(landmarks):

    measurements = {

        "shoulder_height_difference": (
            P.LEFT_SHOULDER,
            P.RIGHT_SHOULDER
        ),

        "hip_height_difference": (
            P.LEFT_HIP,
            P.RIGHT_HIP
        ),

        "knee_height_difference": (
            P.LEFT_KNEE,
            P.RIGHT_KNEE
        ),

        "ankle_height_difference": (
            P.LEFT_ANKLE,
            P.RIGHT_ANKLE
        )
    }

    results = {}

    for name, (left_id, right_id) in measurements.items():

        if not points_visible(
            landmarks, (left_id, right_id)
        ):
            results[name] = None
            continue

        left = get_point(landmarks, left_id)
        right = get_point(landmarks, right_id)

        results[name] = round(
            float(abs(left[1] - right[1])), 4
        )

    return results


# ==========================================
# HEAD POSITION
# ==========================================

def calculate_head_position(landmarks):

    required = [
        P.LEFT_EAR,
        P.RIGHT_EAR,
        P.LEFT_SHOULDER,
        P.RIGHT_SHOULDER,
        P.LEFT_HIP,
        P.RIGHT_HIP
    ]

    if not points_visible(landmarks, required):
        return None

    left_ear = get_point(landmarks, P.LEFT_EAR)
    right_ear = get_point(landmarks, P.RIGHT_EAR)

    left_shoulder = get_point(
        landmarks, P.LEFT_SHOULDER
    )

    right_shoulder = get_point(
        landmarks, P.RIGHT_SHOULDER
    )

    left_hip = get_point(landmarks, P.LEFT_HIP)
    right_hip = get_point(landmarks, P.RIGHT_HIP)

    ear_mid = midpoint(left_ear, right_ear)

    shoulder_mid = midpoint(
        left_shoulder, right_shoulder
    )

    hip_mid = midpoint(left_hip, right_hip)

    torso_length = np.linalg.norm(
        shoulder_mid[:2] - hip_mid[:2]
    )

    if torso_length < 1e-8:
        return None

    return {

        "head_horizontal_offset": round(
            float(ear_mid[0] - shoulder_mid[0]), 4
        ),

        "head_vertical_offset": round(
            float(ear_mid[1] - shoulder_mid[1]), 4
        ),

        "normalized_head_offset": round(
            float(
                np.linalg.norm(
                    ear_mid[:2] - shoulder_mid[:2]
                ) / torso_length
            ), 4
        )
    }


# ==========================================
# TRUNK INCLINATION
# ==========================================

def calculate_trunk_inclination(landmarks):

    required = [
        P.LEFT_SHOULDER,
        P.RIGHT_SHOULDER,
        P.LEFT_HIP,
        P.RIGHT_HIP
    ]

    if not points_visible(landmarks, required):
        return None

    shoulder_mid = midpoint(
        get_point(landmarks, P.LEFT_SHOULDER),
        get_point(landmarks, P.RIGHT_SHOULDER)
    )

    hip_mid = midpoint(
        get_point(landmarks, P.LEFT_HIP),
        get_point(landmarks, P.RIGHT_HIP)
    )

    vector = shoulder_mid[:2] - hip_mid[:2]

    angle = math.degrees(
        math.atan2(vector[0], -vector[1])
    )

    return round(float(angle), 2)


# ==========================================
# MAIN ANALYSIS
# ==========================================

def analyze_posture(landmarks):

    if landmarks is None:
        return None

    return {

        "joint_angles_degrees": get_joint_angles(
            landmarks
        ),

        "body_alignment": calculate_alignment(
            landmarks
        ),

        "body_symmetry": calculate_symmetry(
            landmarks
        ),

        "head_position": calculate_head_position(
            landmarks
        ),

        "trunk_inclination_degrees": (
            calculate_trunk_inclination(landmarks)
        ),

        "landmark_visibility": {

            "average_visibility": round(
                float(np.mean([
                    landmark.visibility
                    for landmark in landmarks
                ])), 4
            ),

            "visible_landmarks": sum(
                landmark.visibility >= 0.5
                for landmark in landmarks
            ),

            "total_landmarks": len(landmarks)
        },

        "measurement_information": {

            "coordinate_system": "2D image plane",

            "camera_view": "Unknown",

            "medical_diagnosis": False,

            "limitations": [
                "Camera perspective may affect measurements.",
                "A single photograph cannot show movement over time.",
                "Front and side views provide different information.",
                "Some landmarks may be hidden or incorrectly detected."
            ]
        }
    }
