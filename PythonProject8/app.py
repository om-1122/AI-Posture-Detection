
import streamlit as st
import numpy as np
import json

from PIL import Image
from dotenv import load_dotenv

from posture_det import (
    detect_posture,
    get_landmark_details
)

from angle_calculation import (
    analyze_posture
)

from ai_analysis import (
    generate_posture_report
)


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv()


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI Human Posture Detection",
    page_icon="🧍",
    layout="wide"
)


# ==========================================
# HEADER
# ==========================================

st.title("🧍 AI Human Posture Detection")

st.write(
    "An AI-based system for human posture detection, "
    "mathematical measurements and AI posture analysis."
)

st.info(
    "Upload a clear photograph of a person. "
    "Full-body images generally provide more landmarks."
)


# ==========================================
# IMAGE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload a person's photograph",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:

        # ==================================
        # READ ORIGINAL IMAGE
        # ==================================

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        image_array = np.array(image)

        st.divider()

        # ==================================
        # MEDIAPIPE DETECTION
        # ==================================

        with st.spinner(
            "Detecting human body landmarks..."
        ):

            annotated_image, landmarks, width, height = (
                detect_posture(image_array)
            )

        # ==================================
        # DISPLAY BOTH IMAGES
        # ==================================

        st.header("📸 Posture Detection Images")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Original Photograph")

            st.image(
                image_array,
                channels="RGB",
                use_container_width=True
            )

        with col2:

            st.subheader("Recognized Photograph")

            if annotated_image is not None:

                st.image(
                    annotated_image,
                    channels="RGB",
                    caption="MediaPipe Pose Detection",
                    use_container_width=True
                )

            else:

                st.warning(
                    "No human body detected."
                )

        # ==================================
        # IMAGE DEBUG INFORMATION
        # ==================================

        with st.expander("Image Information"):

            st.write(
                "Original image dimensions:",
                image_array.shape
            )

            if annotated_image is not None:

                st.write(
                    "Recognized image dimensions:",
                    annotated_image.shape
                )

            st.write(
                "Landmarks detected:",
                landmarks is not None
            )

        # ==================================
        # CHECK LANDMARKS
        # ==================================

        if landmarks is None:

            st.error(
                "No person detected. "
                "Please upload another clear photograph."
            )

        else:

            st.success(
                "Human body landmarks detected successfully!"
            )

            # ==================================
            # LANDMARK INFORMATION
            # ==================================

            details = get_landmark_details(
                landmarks,
                width,
                height
            )

            st.divider()

            st.header("📍 Detected Body Landmarks")

            st.write(
                "Total landmarks:",
                len(details)
            )

            st.dataframe(
                details,
                use_container_width=True
            )

            # ==================================
            # MATHEMATICAL CALCULATIONS
            # ==================================

            st.divider()

            st.header(
                "📐 Mathematical Posture Measurements"
            )

            analysis = analyze_posture(
                landmarks
            )

            # ==================================
            # JOINT ANGLES
            # ==================================

            st.subheader("Joint Angles")

            joint_angles = analysis[
                "joint_angles_degrees"
            ]

            angle_rows = []

            for joint, value in joint_angles.items():

                angle_rows.append({

                    "Joint": joint.replace(
                        "_", " "
                    ).title(),

                    "Angle": (
                        f"{value}°"
                        if value is not None
                        else "Not visible"
                    )
                })

            st.dataframe(
                angle_rows,
                use_container_width=True
            )

            # ==================================
            # BODY ALIGNMENT
            # ==================================

            st.subheader("Body Alignment")

            st.json(
                analysis["body_alignment"]
            )

            # ==================================
            # BODY SYMMETRY
            # ==================================

            st.subheader("Body Symmetry")

            st.json(
                analysis["body_symmetry"]
            )

            # ==================================
            # HEAD POSITION
            # ==================================

            st.subheader("Head Position")

            st.json(
                analysis["head_position"]
            )

            # ==================================
            # TRUNK INCLINATION
            # ==================================

            st.subheader("Trunk Inclination")

            trunk_angle = analysis[
                "trunk_inclination_degrees"
            ]

            st.metric(
                "Trunk inclination",
                (
                    f"{trunk_angle}°"
                    if trunk_angle is not None
                    else "Not available"
                )
            )

            # ==================================
            # DOWNLOAD MATHEMATICAL JSON
            # ==================================

            st.download_button(

                label="Download Mathematical Data (JSON)",

                data=json.dumps(
                    analysis,
                    indent=4
                ),

                file_name="posture_measurements.json",

                mime="application/json"
            )

            # ==================================
            # GROQ AI ANALYSIS
            # ==================================

            st.divider()

            st.header("🤖 Groq AI Posture Analysis")

            st.write(
                "Generate an intelligent posture report "
                "using the calculated mathematical measurements."
            )

            if st.button(
                "Generate AI Posture Report",
                type="primary"
            ):

                try:

                    with st.spinner(
                        "Groq AI is analyzing your posture measurements..."
                    ):

                        report = generate_posture_report(
                            analysis
                        )

                    st.session_state[
                        "posture_report"
                    ] = report

                    st.success(
                        "AI posture report generated successfully!"
                    )

                except Exception as api_error:

                    st.error(
                        f"Groq API Error: {api_error}"
                    )

            # ==================================
            # DISPLAY AI REPORT
            # ==================================

            if "posture_report" in st.session_state:

                st.divider()

                st.subheader(
                    "📋 Final AI Posture Report"
                )

                st.markdown(
                    st.session_state["posture_report"]
                )

                st.download_button(

                    label="Download AI Posture Report",

                    data=st.session_state[
                        "posture_report"
                    ],

                    file_name="AI_Posture_Report.txt",

                    mime="text/plain"
                )

    except Exception as error:

        st.error(
            f"Application Error: {error}"
        )


else:

    st.warning(
        "Please upload a photograph to begin posture detection."
    )


# ==========================================
# FOOTER
# ==========================================

st.divider()

st.caption(
    "AI Human Posture Detection System | "
    "Python + MediaPipe + OpenCV + Streamlit + Groq AI"
)
