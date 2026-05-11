import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


def generate_demo_dataset(n=300):
    np.random.seed(42)

    machine_types = [
        "Compresor",
        "Bomba",
        "Motor eléctrico",
        "Banda transportadora",
        "CNC",
        "Turbina",
        "Prensa hidráulica"
    ]

    plants = ["Planta Norte", "Planta Sur", "Planta Central"]

    rows = []

    for i in range(1, n + 1):
        machine_type = np.random.choice(machine_types)
        plant = np.random.choice(plants)

        days_since_maintenance = int(np.random.gamma(4.5, 12))
        operating_hours = int(days_since_maintenance * np.random.uniform(12, 22))
        avg_temp_c = np.random.normal(65 + days_since_maintenance * 0.08, 8)
        vibration_mm_s = np.random.gamma(2.2, 1.1) + days_since_maintenance * 0.015
        oil_quality_pct = np.clip(
            100 - days_since_maintenance * np.random.uniform(0.45, 0.9),
            5,
            100
        )
        failures_last_90d = np.random.poisson(max(0.05, days_since_maintenance / 55))
        downtime_hours_last_30d = max(
            0,
            np.random.gamma(1.4, 1.8) + failures_last_90d * np.random.uniform(0.7, 2.5)
        )

        risk = 0
        risk += 0.18 * min(days_since_maintenance / 90, 1.5)
        risk += 0.22 * min(vibration_mm_s / 8, 1.5)
        risk += 0.18 * min(max(avg_temp_c - 65, 0) / 35, 1.5)
        risk += 0.18 * min((100 - oil_quality_pct) / 80, 1.3)
        risk += 0.12 * min(failures_last_90d / 4, 1.4)
        risk += 0.12 * min(downtime_hours_last_30d / 12, 1.3)
        risk = float(np.clip(risk, 0, 1))

        if risk >= 0.80:
            priority = "Crítica"
            recommended_within = "1 a 3 días"
        elif risk >= 0.65:
            priority = "Alta"
            recommended_within = "3 a 7 días"
        elif risk >= 0.55:
            priority = "Media"
            recommended_within = "7 a 15 días"
        else:
            priority = "Normal"
            recommended_within = "15 a 45 días"

        if vibration_mm_s > 6.5:
            recommended_action = "Revisar rodamientos, alineación y vibración"
            component = "Rodamiento"
        elif oil_quality_pct < 45:
            recommended_action = "Cambiar aceite/lubricante y revisar contaminación"
            component = "Aceite"
        elif avg_temp_c > 85:
            recommended_action = "Inspeccionar sobrecalentamiento y ventilación"
            component = "Sistema térmico"
        elif failures_last_90d >= 3:
            recommended_action = "Ejecutar inspección preventiva completa por fallas recurrentes"
            component = "Sistema general"
        else:
            recommended_action = "Inspección preventiva general"
            component = "Componente crítico"

        rows.append({
            "machine_id": f"MQ-{i:04d}",
            "plant": plant,
            "machine_type": machine_type,
            "component": component,
            "days_since_maintenance": days_since_maintenance,
            "operating_hours": operating_hours,
            "avg_temp_c": round(avg_temp_c, 2),
            "vibration_mm_s": round(vibration_mm_s, 2),
            "oil_quality_pct": round(oil_quality_pct, 2),
            "failures_last_90d": failures_last_90d,
            "downtime_hours_last_30d": round(downtime_hours_last_30d, 2),
            "risk_score": round(risk, 3),
            "maintenance_required": 1 if risk >= 0.55 else 0,
            "priority": priority,
            "recommended_within": recommended_within,
            "recommended_action": recommended_action
        })

    return pd.DataFrame(rows)


def prepare_dataset(df):
    required_columns = [
        "machine_id",
        "plant",
        "machine_type",
        "component",
        "days_since_maintenance",
        "operating_hours",
        "avg_temp_c",
        "vibration_mm_s",
        "oil_quality_pct",
        "failures_last_90d",
        "downtime_hours_last_30d"
    ]

    for col in required_columns:
        if col not in df.columns:
            if col == "machine_id":
                df[col] = [f"MQ-{i:04d}" for i in range(1, len(df) + 1)]
            elif col in ["plant", "machine_type", "component"]:
                df[col] = "No especificado"
            else:
                df[col] = 0

    df["risk_score"] = (
        0.18 * (df["days_since_maintenance"] / 90).clip(0, 1.5)
        + 0.22 * (df["vibration_mm_s"] / 8).clip(0, 1.5)
        + 0.18 * ((df["avg_temp_c"] - 65).clip(0) / 35).clip(0, 1.5)
        + 0.18 * ((100 - df["oil_quality_pct"]) / 80).clip(0, 1.3)
        + 0.12 * (df["failures_last_90d"] / 4).clip(0, 1.4)
        + 0.12 * (df["downtime_hours_last_30d"] / 12).clip(0, 1.3)
    ).clip(0, 1)

    df["maintenance_required"] = df["risk_score"].apply(lambda x: 1 if x >= 0.55 else 0)

    def priority(risk):
        if risk >= 0.80:
            return "Crítica"
        if risk >= 0.65:
            return "Alta"
        if risk >= 0.55:
            return "Media"
        return "Normal"

    def days(risk):
        if risk >= 0.80:
            return "1 a 3 días"
        if risk >= 0.65:
            return "3 a 7 días"
        if risk >= 0.55:
            return "7 a 15 días"
        return "15 a 45 días"

    def action(row):
        if row["vibration_mm_s"] > 6.5:
            return "Revisar rodamientos, alineación y vibración"
        if row["oil_quality_pct"] < 45:
            return "Cambiar aceite/lubricante y revisar contaminación"
        if row["avg_temp_c"] > 85:
            return "Inspeccionar sobrecalentamiento y ventilación"
        if row["failures_last_90d"] >= 3:
            return "Ejecutar inspección preventiva completa por fallas recurrentes"
        return "Inspección preventiva general"

    df["priority"] = df["risk_score"].apply(priority)
    df["recommended_within"] = df["risk_score"].apply(days)
    df["recommended_action"] = df.apply(action, axis=1)

    return df


def run_predictive_maintenance_demo():
    st.header("🏭 Mantenimiento predictivo industrial")
    st.write(
        "Carga un dataset de máquinas industriales o usa datos simulados. "
        "El sistema calcula riesgo, prioridad, cuándo hacer mantenimiento y qué acción ejecutar."
    )

    uploaded_file = st.file_uploader(
        "Sube un CSV de máquinas industriales",
        type=["csv"],
        key="industrial_machines_csv"
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = prepare_dataset(df)
        st.success("Dataset cargado correctamente.")
    else:
        df = generate_demo_dataset()
        st.info("Usando dataset simulado de máquinas industriales.")

    st.subheader("Vista previa")
    st.dataframe(df.head(30), width="stretch")

    total_machines = len(df)
    maintenance_count = int(df["maintenance_required"].sum())
    critical_count = int((df["priority"] == "Crítica").sum())
    avg_risk = df["risk_score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Máquinas analizadas", total_machines)
    c2.metric("Requieren mantenimiento", maintenance_count)
    c3.metric("Prioridad crítica", critical_count)
    c4.metric("Riesgo promedio", f"{avg_risk:.2%}")

    st.subheader("Máquinas priorizadas")

    prioritized = df.sort_values("risk_score", ascending=False)

    columns_to_show = [
        "machine_id",
        "plant",
        "machine_type",
        "component",
        "risk_score",
        "priority",
        "recommended_within",
        "recommended_action",
        "days_since_maintenance",
        "operating_hours",
        "avg_temp_c",
        "vibration_mm_s",
        "oil_quality_pct",
        "failures_last_90d",
        "downtime_hours_last_30d"
    ]

    st.dataframe(prioritized[columns_to_show].head(50), width="stretch")

    st.subheader("Top 15 máquinas con mayor riesgo")

    top_risk = prioritized.head(15)

    fig_top = px.bar(
        top_risk,
        x="machine_id",
        y="risk_score",
        color="priority",
        hover_data=["machine_type", "component", "recommended_action"],
        title="Top 15 máquinas con mayor riesgo"
    )

    st.plotly_chart(fig_top, width="stretch")

    st.subheader("Consulta individual de máquina")

    selected_machine = st.selectbox(
        "Selecciona una máquina",
        df["machine_id"].tolist()
    )

    machine = df[df["machine_id"] == selected_machine].iloc[0]

    m1, m2, m3 = st.columns(3)
    m1.metric("Riesgo", f"{machine['risk_score']:.2%}")
    m2.metric("Prioridad", machine["priority"])
    m3.metric("Mantenimiento sugerido", machine["recommended_within"])

    st.success(
        f"""
        Máquina: {machine['machine_id']}  
        Tipo: {machine['machine_type']}  
        Componente: {machine['component']}  
        Acción recomendada: {machine['recommended_action']}
        """
    )

    csv = prioritized.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="⬇️ Descargar análisis priorizado CSV",
        data=csv,
        file_name="analisis_mantenimiento_predictivo.csv",
        mime="text/csv"
    )