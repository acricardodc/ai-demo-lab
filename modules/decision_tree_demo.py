import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


def run_decision_tree_demo():
    st.header("🌳 Árbol de decisión")
    st.write(
        "Demo educativa para mostrar cómo un modelo aprende reglas de decisión a partir de datos."
    )

    data = load_iris()

    X = data.data
    y = data.target

    df = pd.DataFrame(X, columns=data.feature_names)
    df["target"] = y
    df["target_name"] = df["target"].apply(lambda value: data.target_names[value])

    st.subheader("Dataset de ejemplo")
    st.dataframe(df.head(20), width="stretch")

    col1, col2 = st.columns(2)

    with col1:
        max_depth = st.slider("Profundidad máxima", 1, 10, 3)

    with col2:
        test_size = st.slider("Porcentaje de prueba", 0.1, 0.5, 0.2)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42
    )

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    st.metric("Precisión del modelo", f"{accuracy:.2%}")

    st.subheader("Matriz de confusión")
    cm = confusion_matrix(y_test, predictions)
    st.dataframe(cm, width="stretch")

    st.subheader("Reporte de clasificación")
    report = classification_report(
        y_test,
        predictions,
        target_names=data.target_names,
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose(), width="stretch")

    st.subheader("Visualización del árbol")

    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        model,
        feature_names=data.feature_names,
        class_names=data.target_names,
        filled=True,
        rounded=True,
        ax=ax
    )
    st.pyplot(fig)

    st.info(
        "Este ejemplo permite explicar entrenamiento, prueba, precisión, "
        "reglas de decisión y sobreajuste."
    )