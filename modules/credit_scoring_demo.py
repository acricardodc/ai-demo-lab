import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def generate_credit_dataset(n=800):
    np.random.seed(42)

    income = np.random.normal(1800, 650, n).clip(400, 6000)
    debt_ratio = np.random.uniform(0.05, 0.9, n)
    credit_history_months = np.random.randint(0, 180, n)
    late_payments = np.random.poisson(1.2, n)
    requested_amount = np.random.normal(3500, 1800, n).clip(500, 15000)
    employment_years = np.random.uniform(0, 20, n)

    risk_score = (
        0.35 * debt_ratio
        + 0.25 * (late_payments / 8)
        + 0.20 * (requested_amount / income / 10)
        - 0.15 * (credit_history_months / 180)
        - 0.10 * (employment_years / 20)
    )

    approved = (risk_score < np.percentile(risk_score, 62)).astype(int)

    df = pd.DataFrame(
        {
            "income": income.round(2),
            "debt_ratio": debt_ratio.round(2),
            "credit_history_months": credit_history_months,
            "late_payments": late_payments,
            "requested_amount": requested_amount.round(2),
            "employment_years": employment_years.round(1),
            "approved": approved
        }
    )

    return df


def run_credit_scoring_demo():
    st.header("🧠 Score crediticio demo")
    st.write(
        "Simulación de decisión basada en datos para explicar modelos predictivos, "
        "riesgo y variables influyentes."
    )

    df = generate_credit_dataset()

    st.subheader("Dataset simulado")
    st.dataframe(df.head(20), width="stretch")

    features = [
        "income",
        "debt_ratio",
        "credit_history_months",
        "late_payments",
        "requested_amount",
        "employment_years"
    ]

    X = df[features]
    y = df["approved"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=5,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    st.metric("Precisión estimada del modelo demo", f"{accuracy:.2%}")

    st.subheader("Simular solicitud de crédito")

    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.number_input("Ingreso mensual", 400, 10000, 1800)
        debt_ratio = st.slider("Relación deuda / ingreso", 0.0, 1.0, 0.35)

    with col2:
        credit_history_months = st.slider("Historial crediticio meses", 0, 240, 48)
        late_payments = st.slider("Pagos tardíos", 0, 12, 1)

    with col3:
        requested_amount = st.number_input("Monto solicitado", 500, 50000, 3500)
        employment_years = st.slider("Años de empleo", 0.0, 30.0, 3.0)

    input_data = pd.DataFrame(
        [
            {
                "income": income,
                "debt_ratio": debt_ratio,
                "credit_history_months": credit_history_months,
                "late_payments": late_payments,
                "requested_amount": requested_amount,
                "employment_years": employment_years
            }
        ]
    )

    probability = model.predict_proba(input_data)[0][1]
    decision = "Aprobación sugerida" if probability >= 0.5 else "Revisión / rechazo sugerido"

    st.subheader("Resultado de la evaluación")

    c1, c2, c3 = st.columns(3)

    c1.metric("Probabilidad de aprobación", f"{probability:.2%}")
    c2.metric("Decisión", decision)
    c3.metric("Nivel de riesgo", "Bajo" if probability >= 0.7 else "Medio" if probability >= 0.45 else "Alto")

    st.subheader("Factores más influyentes")

    importance = pd.DataFrame(
        {
            "Variable": features,
            "Importancia": model.feature_importances_
        }
    ).sort_values("Importancia", ascending=False)

    fig = px.bar(
        importance,
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Importancia de variables en el modelo"
    )

    st.plotly_chart(fig, width="stretch")

    st.warning(
        "Un sistema real de crédito requiere "
        "validación legal, auditoría de sesgos, explicabilidad, protección de datos "
        "y revisión humana."
    )