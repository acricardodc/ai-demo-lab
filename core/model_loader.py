import streamlit as st
from ultralytics import YOLO


@st.cache_resource
def load_yolo_model(model_name: str = "yolo11n.pt"):
    return YOLO(model_name)