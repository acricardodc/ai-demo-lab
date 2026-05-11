import os
import tempfile
from collections import Counter

import cv2
import pandas as pd
import streamlit as st

from core.model_loader import load_yolo_model


def run_video_detection():
    st.header("🎥 Detección de objetos en video")
    st.write(
        "Sube un video MP4. El sistema detectará vehículos, personas y objetos frame por frame."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        model_name = st.selectbox(
            "Modelo YOLO",
            ["yolo11n.pt", "yolo11s.pt"],
            index=0,
            key="video_model"
        )

    with col2:
        confidence = st.slider(
            "Confianza mínima",
            0.10,
            0.95,
            0.35,
            0.05,
            key="video_conf"
        )

    with col3:
        frame_skip = st.slider(
            "Procesar cada N frames",
            1,
            10,
            3
        )

    uploaded_video = st.file_uploader(
        "Sube un video",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video is None:
        st.info("Sube un video para iniciar.")
        return

    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_video.read())
    temp_input.close()

    st.video(temp_input.name)

    if st.button("🚀 Analizar video", width="stretch"):
        model = load_yolo_model(model_name)

        cap = cv2.VideoCapture(temp_input.name)

        if not cap.isOpened():
            st.error("No se pudo abrir el video.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = os.path.join("outputs", "videos", "video_annotated.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        progress = st.progress(0)
        status = st.empty()

        class_counter = Counter()
        processed = 0
        frame_index = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            frame_index += 1

            if frame_index % frame_skip != 0:
                writer.write(frame)
                continue

            results = model.predict(frame, conf=confidence, verbose=False)
            result = results[0]
            annotated_frame = result.plot()

            for box in result.boxes:
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]
                class_counter[class_name] += 1

            writer.write(annotated_frame)

            processed += 1
            progress.progress(min(frame_index / total_frames, 1.0))
            status.write(f"Procesando frame {frame_index} de {total_frames}")

        cap.release()
        writer.release()

        st.success("Video procesado correctamente.")
        ##st.video(output_path)

        st.subheader("Resumen de objetos detectados")

        if class_counter:
            df = pd.DataFrame(
                class_counter.items(),
                columns=["Objeto", "Cantidad aproximada"]
            ).sort_values("Cantidad aproximada", ascending=False)

            st.dataframe(df, width="stretch")

            c1, c2, c3 = st.columns(3)
            c1.metric("Frames procesados", processed)
            c2.metric("Tipos de objetos", len(class_counter))
            c3.metric("Detecciones acumuladas", sum(class_counter.values()))
        else:
            st.warning("No se detectaron objetos.")