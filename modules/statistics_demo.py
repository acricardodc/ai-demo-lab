import pandas as pd
import plotly.express as px
import streamlit as st


def run_statistics_demo():
    st.header("📊 Estadística con datasets")
    st.write(
        "Carga un CSV y genera análisis descriptivo, histogramas, correlaciones y métricas rápidas."
    )

    uploaded_file = st.file_uploader(
        "Sube un archivo CSV",
        type=["csv"],
        key="csv_stats"
    )

    if uploaded_file is None:
        st.info("Sube un CSV para iniciar el análisis.")
        return

    df = pd.read_csv(uploaded_file)

    st.subheader("Vista previa")
    st.dataframe(df.head(20), width="stretch")

    rows, columns = df.shape

    c1, c2, c3 = st.columns(3)
    c1.metric("Filas", rows)
    c2.metric("Columnas", columns)
    c3.metric("Valores nulos", int(df.isnull().sum().sum()))

    st.subheader("Resumen estadístico")
    st.dataframe(df.describe(include="all").transpose(), width="stretch")

    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if not numeric_columns:
        st.warning("El dataset no contiene columnas numéricas.")
        return

    selected_column = st.selectbox(
        "Variable numérica para analizar",
        numeric_columns
    )

    fig_hist = px.histogram(
        df,
        x=selected_column,
        title=f"Distribución de {selected_column}",
        nbins=30
    )
    st.plotly_chart(fig_hist, width="stretch")

    if len(numeric_columns) >= 2:
        st.subheader("Matriz de correlación")
        corr = df[numeric_columns].corr()

        fig_corr = px.imshow(
            corr,
            text_auto=True,
            title="Correlación entre variables"
        )
        st.plotly_chart(fig_corr, width="stretch")