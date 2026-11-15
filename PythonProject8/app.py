
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
            "No human body detected in this photograph."
        )
