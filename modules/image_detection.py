import tempfile
import pandas as pd
import streamlit as st
from PIL import Image

from core.model_loader import load_yolo_model


def run_image_detection():
    st.header("👁️ Detección de objetos en imágenes")
    st.write(
        "Sube una imagen y la IA detectará objetos como personas, vehículos, animales, "
        "productos u otros elementos visuales."
    )

    col_a, col_b = st.columns([1, 2])

    with col_a:
        model_name = st.selectbox(
            "Modelo YOLO",
            ["yolo11n.pt", "yolo11s.pt"],
            index=0
        )

        confidence = st.slider(
            "Confianza mínima",
            min_value=0.10,
            max_value=0.95,
            value=0.35,
            step=0.05
        )

        uploaded_file = st.file_uploader(
            "Sube una imagen",
            type=["jpg", "jpeg", "png", "webp"]
        )

    with col_b:
        st.markdown(
            """
            <div class="status-ok">
                Demo para clientes de seguridad, retail, logística,
                publicidad exterior, inventario y monitoreo visual.
            </div>
            """,
            unsafe_allow_html=True
        )

    if uploaded_file is None:
        st.info("Carga una imagen para iniciar la detección.")
        return

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Imagen original")
    st.image(image, width="stretch")

    if st.button("🚀 Ejecutar detección", width="stretch"):
        model = load_yolo_model(model_name)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            results = model.predict(tmp.name, conf=confidence)

        result = results[0]
        annotated_image = result.plot()

        st.subheader("Resultado con detecciones")
        st.image(annotated_image, width="stretch")

        rows = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            score = float(box.conf[0])
            rows.append(
                {
                    "Objeto": class_name,
                    "Confianza": round(score, 3)
                }
            )

        st.subheader("Objetos detectados")

        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch")
            st.metric("Total de objetos detectados", len(rows))
        else:
            st.warning("No se detectaron objetos con la confianza seleccionada.")