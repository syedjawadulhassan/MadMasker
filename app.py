import os

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---------------- RTC CONFIG ----------------

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ---------------- LOAD MODELS ----------------


@st.cache_resource
def load_models():
    prototxt_path = "face_detector/deploy.prototxt"
    weights_path = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"

    if not os.path.exists(prototxt_path) or not os.path.exists(weights_path):
        st.error(
            "Face detector model files not found. Make sure "
            "'face_detector/deploy.prototxt' and "
            "'face_detector/res10_300x300_ssd_iter_140000.caffemodel' "
            "are included in the deployment."
        )
        st.stop()

    if not os.path.exists("mask_detector.model"):
        st.error("'mask_detector.model' not found in the deployment.")
        st.stop()

    faceNet = cv2.dnn.readNet(prototxt_path, weights_path)
    maskNet = load_model("mask_detector.model")

    return faceNet, maskNet


faceNet, maskNet = load_models()

# ---------------- CORE DETECTION LOGIC ----------------


def run_detection(image, faceNet, maskNet):
    """
    image: BGR numpy array
    returns: BGR numpy array with boxes/labels drawn
    """
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        image,
        1.0,
        (300, 300),
        (104.0, 177.0, 123.0),
    )

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < 0.5:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(w - 1, endX)
        endY = min(h - 1, endY)

        face = image[startY:endY, startX:endX]

        if face.size == 0:
            continue

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        (mask, withoutMask) = maskNet.predict(face, verbose=0)[0]

        if mask > withoutMask:
            label = "Mask"
            color = (0, 255, 0)
        else:
            label = "No Mask"
            color = (0, 0, 255)

        text = f"{label}: {max(mask, withoutMask) * 100:.2f}%"

        cv2.putText(
            image,
            text,
            (startX, max(0, startY - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )

        cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

    return image


# ---------------- WEBCAM PROCESSOR ----------------


class MaskDetector(VideoProcessorBase):
    def __init__(self):
        self.faceNet = faceNet
        self.maskNet = maskNet

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        image = run_detection(image, self.faceNet, self.maskNet)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


# ---------------- CSS (OPTIONAL) ----------------


def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ---------------- MAIN APP ----------------


def mask_detection():
    local_css("css/styles.css")

    st.markdown(
        "<h1 style='text-align: center;'>😷 Face Mask Detection</h1>",
        unsafe_allow_html=True,
    )

    option = st.sidebar.selectbox("Choose Detection Mode", ["Image", "Webcam"])

    # ---------------- IMAGE ----------------

    if option == "Image":
        st.subheader("Upload an Image")

        uploaded_file = st.file_uploader(
            "Choose an image", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Detect Mask"):
                image_np = np.array(image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

                result_bgr = run_detection(image_bgr, faceNet, maskNet)
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

                st.image(result_rgb, caption="Prediction", use_container_width=True)

    # ---------------- WEBCAM ----------------

    else:
        st.subheader("Real-Time Webcam Detection")

        webrtc_streamer(
            key="mask-detection",
            video_processor_factory=MaskDetector,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )


# ---------------- RUN ----------------

if __name__ == "__main__":
    mask_detection()
