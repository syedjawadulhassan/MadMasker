from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title='Face Mask Detector',
    page_icon='😷',
    layout='centered',
    initial_sidebar_state='expanded'
)

# ---------------- WEBCAM CLASS ----------------
class MaskDetector(VideoTransformerBase):
    def __init__(self):
        self.faceNet = cv2.dnn.readNet(
            "face_detector/deploy.prototxt",
            "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        )
        self.maskNet = load_model("mask_detector.model")

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        (h, w) = img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            img, 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )

        self.faceNet.setInput(blob)
        detections = self.faceNet.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = img[startY:endY, startX:endX]
                if face is None or face.size == 0:
                    continue

                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_rgb = cv2.resize(face_rgb, (224, 224))
                face_rgb = img_to_array(face_rgb)
                face_rgb = preprocess_input(face_rgb)
                face_rgb = np.expand_dims(face_rgb, axis=0)

                (mask, withoutMask) = self.maskNet.predict(face_rgb, verbose=0)[0]

                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                label = f"{label}: {max(mask, withoutMask)*100:.2f}%"

                cv2.putText(img, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(img, (startX, startY),
                              (endX, endY), color, 2)

        return img


# ---------------- CSS ----------------
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


# ---------------- IMAGE MODE ----------------
def mask_image():
    prototxtPath = os.path.sep.join(["face_detector", "deploy.prototxt"])
    weightsPath = os.path.sep.join(["face_detector",
                                    "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
    model = load_model("mask_detector.model")

    image = cv2.imread("./images/out.jpg")
    if image is None:
        return None

    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = image[startY:endY, startX:endX]
            if face is None or face.size == 0:
                continue

            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            (mask, withoutMask) = model.predict(face, verbose=0)[0]

            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            cv2.putText(image, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------- MAIN APP ----------------
def mask_detection():
    local_css("css/styles.css")

    st.markdown('<h1 align="center">😷 Face Mask Detection</h1>', unsafe_allow_html=True)

    activities = ["Image", "Webcam"]
    st.sidebar.markdown("# Mask Detection on?")
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    # ---------- IMAGE ----------
    if choice == 'Image':
        st.markdown('<h2 align="center">Detection on Image</h2>', unsafe_allow_html=True)
        st.markdown("### Upload your image here ⬇")

        image_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'])

        if image_file is not None:
            our_image = Image.open(image_file)
            our_image.save('./images/out.jpg')

            st.image(image_file, use_column_width=True)
            st.markdown('<h3 align="center">Image uploaded successfully!</h3>', unsafe_allow_html=True)

            if st.button('Process'):
                result = mask_image()
                if result is not None:
                    st.image(result, use_column_width=True)

    # ---------- WEBCAM ----------
    elif choice == 'Webcam':
        st.markdown('<h2 align="center">Detection on Webcam</h2>', unsafe_allow_html=True)

        webrtc_streamer(
            key="mask-detection",
            video_transformer_factory=MaskDetector
        )


# ---------------- RUN ----------------
mask_detection()