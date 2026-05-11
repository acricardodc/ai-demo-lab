import streamlit as st


def load_global_styles():
    st.markdown(
        """
        <style>
        .main {
            background-color: #020617;
        }

        .hero {
            padding: 30px;
            border-radius: 24px;
            background: linear-gradient(135deg, #0F172A, #1E293B, #0EA5E9);
            box-shadow: 0px 10px 30px rgba(14, 165, 233, 0.25);
            margin-bottom: 25px;
        }

        .hero-title {
            font-size: 46px;
            font-weight: 900;
            color: #FFFFFF;
            margin-bottom: 5px;
        }

        .hero-subtitle {
            font-size: 18px;
            color: #CBD5E1;
        }

        .ai-card {
            padding: 24px;
            border-radius: 22px;
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0px 8px 24px rgba(0,0,0,0.35);
            min-height: 170px;
        }

        .ai-card h3 {
            color: #F8FAFC;
            font-size: 22px;
            margin-bottom: 8px;
        }

        .ai-card p {
            color: #CBD5E1;
            font-size: 15px;
        }

        .metric-box {
            padding: 20px;
            border-radius: 20px;
            background: linear-gradient(135deg, #111827, #1E293B);
            border: 1px solid rgba(56, 189, 248, 0.35);
            text-align: center;
        }

        .metric-number {
            font-size: 34px;
            font-weight: 900;
            color: #38BDF8;
        }

        .metric-label {
            color: #CBD5E1;
            font-size: 14px;
        }

        .status-ok {
            padding: 12px 16px;
            border-radius: 14px;
            background: rgba(34, 197, 94, 0.15);
            border: 1px solid rgba(34, 197, 94, 0.45);
            color: #BBF7D0;
        }

        .status-warning {
            padding: 12px 16px;
            border-radius: 14px;
            background: rgba(245, 158, 11, 0.15);
            border: 1px solid rgba(245, 158, 11, 0.45);
            color: #FDE68A;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def hero():
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">AI Command Center Demo Lab</div>
            <div class="hero-subtitle">
                Laboratorio visual de inteligencia artificial con visión en tiempo real,
                análisis de datos, modelos predictivos y demostraciones empresariales.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )