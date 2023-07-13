import streamlit as st
import cv2
import tensorflow as tf
from lib import detect_face

def main():
    st.title("Facial Expression Recognition")

    option = st.sidebar.selectbox(
        "Choose an option to detect and classify facial expressions.",
        (["External Camera"])
    )
    frame_skip_rate = 3  # Best optimize frame

    if option == "External Camera":
        camera_address = st.text_input("Camera Address (e.g: http://192.168.137.101:4747/video)")
        if camera_address:
            vid = cv2.VideoCapture(camera_address)
            st.title('Using Mobile Camera with Streamlit')
            frame_window = st.image([])
            frame_count = 0  # Initialize frame count
            while True:
                got_frame, frame = vid.read()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if got_frame:
                    if frame_count % frame_skip_rate == 0:  # Process this frame
                        frame_window.image(detect_face(frame))

                frame_count += 1  # Increment frame count

if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
      try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
          tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
    main()
