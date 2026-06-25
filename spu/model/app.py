from pathlib import Path

import pandas as pd
import streamlit as st

from vaccine_predictor import VaccineCharCNNPredictor


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_charcnn_model.pt"


st.set_page_config(
    page_title="Vaccine Prediction",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f7f9fb;
        color: #16202a;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 249, 251, 0.88);
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.4rem;
        padding-bottom: 2.5rem;
    }

    .hero {
        display: grid;
        grid-template-columns: 1.1fr 0.9fr;
        gap: 32px;
        align-items: center;
        margin-bottom: 28px;
        padding: 8px 0 12px;
    }

    .hero h1 {
        margin: 0 0 10px;
        font-size: 2.25rem;
        line-height: 1.15;
        color: #0f2537;
        letter-spacing: 0;
    }

    .hero p {
        margin: 0;
        color: #4c5c68;
        font-size: 1.02rem;
        line-height: 1.7;
    }

    .hero-visual {
        min-height: 190px;
        border-radius: 8px;
        background:
            linear-gradient(135deg, rgba(12, 82, 98, 0.86), rgba(34, 134, 118, 0.78)),
            url("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?auto=format&fit=crop&w=1000&q=80");
        background-size: cover;
        background-position: center;
        box-shadow: 0 18px 40px rgba(15, 37, 55, 0.14);
    }

    .result-box {
        border: 1px solid #d8e2e7;
        border-radius: 8px;
        padding: 22px 24px;
        background: #ffffff;
        box-shadow: 0 10px 24px rgba(15, 37, 55, 0.06);
    }

    .result-label {
        margin: 0 0 8px;
        color: #60717d;
        font-size: 0.9rem;
    }

    .result-title {
        margin: 0;
        color: #0f5c6b;
        font-size: 1.75rem;
        line-height: 1.25;
        font-weight: 750;
    }

    .small-note {
        color: #687986;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    div[data-testid="stMetricValue"] {
        color: #0f5c6b;
    }

    .stButton > button {
        width: 100%;
        min-height: 46px;
        border-radius: 8px;
        border: 0;
        background: #0f5c6b;
        color: #ffffff;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: #0b4a56;
        color: #ffffff;
        border: 0;
    }

    @media (max-width: 760px) {
        .hero {
            grid-template-columns: 1fr;
            gap: 18px;
        }

        .hero h1 {
            font-size: 1.7rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_predictor():
    return VaccineCharCNNPredictor(str(MODEL_PATH))


st.markdown(
    """
    <section class="hero">
        <div>
            <h1>Vaccine Outcome Prediction</h1>
            <p>
                ระบบช่วยคาดการณ์วัคซีนที่เกี่ยวข้องจากอาการและอายุของผู้ป่วย
                โดยใช้โมเดล Character CNN ที่คุณฝึกไว้
            </p>
        </div>
        <div class="hero-visual" aria-hidden="true"></div>
    </section>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.subheader("Model")
    st.caption(str(MODEL_PATH.name))
    top_k = st.slider("จำนวนผลลัพธ์", min_value=1, max_value=5, value=3)
    st.divider()
    st.caption(
        "ผลลัพธ์เป็นการคาดการณ์จากโมเดล ควรใช้ร่วมกับการประเมินทางการแพทย์"
    )


left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.subheader("ข้อมูลนำเข้า")
    symptom_input = st.text_area(
        "อาการ",
        value="ไอเป็นชุด หายใจมีเสียงวู๊ป ต้องการทราบวัคซีนที่ถึงกำหนด",
        height=150,
        placeholder="พิมพ์อาการหรือคำถามเกี่ยวกับวัคซีน",
    )

    age_col, unit_col = st.columns([1, 1])
    with age_col:
        age_value = st.number_input(
            "อายุ",
            min_value=0.0,
            max_value=120.0,
            value=4.0,
            step=1.0,
        )
    with unit_col:
        age_unit = st.selectbox("หน่วยอายุ", ["ปี", "เดือน"])

    extra_text = st.text_input(
        "ข้อมูลเพิ่มเติม",
        placeholder="เช่น ประวัติ วัคซีนที่เคยได้รับ หรือรายละเอียดอื่น",
    )

    submitted = st.button("Predict", type="primary")


with right:
    st.subheader("ผลการทำนาย")

    if submitted:
        if not symptom_input.strip() and not extra_text.strip():
            st.warning("กรุณากรอกอาการหรือข้อมูลเพิ่มเติมก่อนทำนาย")
        elif not MODEL_PATH.exists():
            st.error(f"ไม่พบไฟล์โมเดล: {MODEL_PATH}")
        else:
            with st.spinner("กำลังประมวลผล..."):
                predictor = load_predictor()
                result = predictor.predict_one(
                    symptom_input=symptom_input,
                    age_input=f"{age_value:g}{age_unit}",
                    input_text=extra_text,
                    return_top_k=top_k,
                )

            labels = result["top_3_labels"]
            probabilities = result["top_3_probabilities"]
            best_label = labels[0]
            best_probability = probabilities[0] * 100

            st.markdown(
                f"""
                <div class="result-box">
                    <p class="result-label">คำแนะนำอันดับแรก</p>
                    <p class="result-title">{best_label}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.metric("ความมั่นใจ", f"{best_probability:.2f}%")

            chart_data = pd.DataFrame(
                {
                    "วัคซีน": labels,
                    "ความมั่นใจ (%)": [prob * 100 for prob in probabilities],
                }
            )
            st.bar_chart(chart_data, x="วัคซีน", y="ความมั่นใจ (%)")
            st.dataframe(
                chart_data,
                hide_index=True,
                use_container_width=True,
            )
    else:
        st.markdown(
            """
            <div class="result-box">
                <p class="result-label">พร้อมใช้งาน</p>
                <p class="result-title">กรอกข้อมูลแล้วกด Predict</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="small-note">โมเดลจะโหลดครั้งแรกเมื่อกดทำนาย และ Streamlit จะ cache ไว้สำหรับการใช้งานครั้งถัดไป</p>',
            unsafe_allow_html=True,
        )
