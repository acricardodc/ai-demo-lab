import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from core.model_loader import load_yolo_model


class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_yolo_model("yolo11n.pt")
        self.confidence = 0.35

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        small = cv2.resize(image, (640, 480))

        results = self.model.predict(
            small,
            conf=self.confidence,
            verbose=False
        )

        annotated = results[0].plot()

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


def run_webcam_detection():
    st.header("📹 Webcam en tiempo real con YOLO")
    st.write("Activa la cámara y permite el acceso cuando el navegador lo solicite.")

    confidence = st.slider(
        "Confianza mínima",
        min_value=0.10,
        max_value=0.95,
        value=0.35,
        step=0.05
    )

    st.info(
        "La webcam depende del navegador, permisos de cámara y conexión WebRTC. "
        "Si falla en una red corporativa, puede requerir TURN."
    )

    ctx = webrtc_streamer(
        key="yolo-webcam",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=YOLOVideoProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
        async_processing=True
    )

    if ctx.video_processor:
        ctx.video_processor.confidence = confidence