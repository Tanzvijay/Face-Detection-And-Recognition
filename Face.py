import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os


def load_image(image_file):
    img = Image.open(image_file)
    return img


def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces


def draw_faces(image, faces):
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image


st.title("Face Detection App")

st.sidebar.title("Upload Image or Video")
uploaded_file = st.sidebar.file_uploader("Choose an image or video...",
                                         type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    file_type = uploaded_file.type.split('/')[0]

    if file_type == 'image':
        st.sidebar.subheader("Image")
        image = load_image(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Detecting faces...")

        image = np.array(image.convert('RGB'))
        faces = detect_faces(image)
        image_with_faces = draw_faces(image, faces)

        st.image(image_with_faces, caption='Processed Image.', use_column_width=True)
        st.write(f"Detected {len(faces)} faces")

    elif file_type == 'video':
        st.sidebar.subheader("Video")
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            faces = detect_faces(frame)
            frame_with_faces = draw_faces(frame, faces)

            stframe.image(frame_with_faces, channels="BGR")

        video.release()
        os.remove(tfile.name)
