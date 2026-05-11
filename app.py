import streamlit as st

from core.styles import load_global_styles, hero
from modules.image_detection import run_image_detection
from modules.video_detection import run_video_detection
from modules.webcam_detection import run_webcam_detection
from modules.statistics_demo import run_statistics_demo
from modules.decision_tree_demo import run_decision_tree_demo
from modules.credit_scoring_demo import run_credit_scoring_demo
from modules.predictive_maintenance_demo import run_predictive_maintenance_demo


st.set_page_config(
    page_title="AI Command Center Demo Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_global_styles()

with st.sidebar:
    st.title("🤖 AI Demo Lab")
    st.caption("Python + IA + YOLO + Datos + Webcam")

    menu = st.radio(
        "Selecciona una demo",
        [
            "Inicio",
            "Detección de imágenes",
            "Detección en video",
            "Webcam en tiempo real",
            "Estadística con CSV",
            "Árbol de decisión",
            "Score crediticio demo",
            "Mantenimiento predictivo"
        ]
    )

hero()

if menu == "Inicio":
    st.subheader("Panel principal")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div class="ai-card">
                <h3>👁️ Visión Artificial</h3>
                <p>Detección de imágenes, video y webcam en tiempo real usando YOLO.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div class="ai-card">
                <h3>📊 Decisiones con Datos</h3>
                <p>Score crediticio, árboles de decisión, estadística y mantenimiento predictivo.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div class="ai-card">
                <h3>🚀 Efecto Wow</h3>
                <p>Una plataforma visual para demostrar IA aplicada a negocio.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

elif menu == "Detección de imágenes":
    run_image_detection()

elif menu == "Detección en video":
    run_video_detection()

elif menu == "Webcam en tiempo real":
    run_webcam_detection()

elif menu == "Estadística con CSV":
    run_statistics_demo()

elif menu == "Árbol de decisión":
    run_decision_tree_demo()

elif menu == "Score crediticio demo":
    run_credit_scoring_demo()

elif menu == "Mantenimiento predictivo":
    run_predictive_maintenance_demo()