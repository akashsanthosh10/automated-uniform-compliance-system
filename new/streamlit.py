import streamlit as st
import cv2
from detection2 import Detection
import pathlib

def main():
    st.title("Live Object Detection and Face Recognition")

    # Sidebar options
    st.sidebar.header("Settings")
    capture_index = st.sidebar.number_input("Capture Index", value=0, step=1)
    model_name = st.sidebar.text_input("Model Name", value='best.pt')

    # Placeholder for displaying the live feed
    placeholder = st.empty()
    pathlib.PosixPath = pathlib.WindowsPath
    # Create a detection object
    detector = Detection(capture_index=capture_index, model_name=model_name)

    # Start the video capture and processing
    for frame, object_results in detector.vidcap():
        # Display the frame in the Streamlit app
        placeholder.image(frame, channels="BGR")

"""        # Display the object labels and coordinates
        if object_results:
            labels, cords = object_results
            for i in range(len(labels)):
                object_name = detector.class_to_label(labels[i])
                x1, y1, x2, y2 = int(cords[i][0]), int(cords[i][1]), int(cords[i][2]), int(cords[i][3])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, object_name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)"""

if __name__ == "__main__":
    main()