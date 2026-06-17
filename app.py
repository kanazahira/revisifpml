import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #F0F4F8; }
.block-container { padding: 2.5rem 2rem 2rem 2rem !important; max-width: 1400px; }

.hero-header {
    background: linear-gradient(135deg, #1a2f6e 0%, #2356a0 50%, #1a6eb5 100%);
    border-radius: 20px;
    padding: 32px 40px 28px 40px;
    margin-top: 8px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px rgba(26,47,110,0.18);
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 32px; font-weight: 700;
    color: #ffffff; margin: 0 0 6px 0; line-height: 1.2;
}
.hero-sub { font-size: 13.5px; color: rgba(255,255,255,0.78); margin: 0; }
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 20px; padding: 4px 14px;
    font-size: 11.5px; color: #fff; font-weight: 500;
    margin-right: 8px; margin-top: 12px;
}

.sec-header {
    font-size: 12px; font-weight: 700; color: #1a2f6e;
    letter-spacing: 1px; text-transform: uppercase;
    border-left: 3px solid #2356a0;
    padding-left: 10px; margin: 16px 0 12px 0;
}

.result-danger {
    background: linear-gradient(135deg, #c0392b, #e74c3c);
    border-radius: 14px; padding: 18px 24px;
    text-align: center; font-size: 18px; font-weight: 700;
    color: white; margin: 14px 0;
    box-shadow: 0 4px 18px rgba(192,57,43,0.30);
}
.result-safe {
    background: linear-gradient(135deg, #1b6b4a, #27ae60);
    border-radius: 14px; padding: 18px 24px;
    text-align: center; font-size: 18px; font-weight: 700;
    color: white; margin: 14px 0;
    box-shadow: 0 4px 18px rgba(27,107,74,0.30);
}

.info-box {
    background: #eef3fb; border-radius: 10px;
    padding: 13px 16px; border-left: 3px solid #2356a0;
    font-size: 13px; color: #2d3748; margin: 10px 0; line-height: 1.6;
}
.info-box-warn {
    background: #fff8e8; border-radius: 10px;
    padding: 13px 16px; border-left: 3px solid #d97706;
    font-size: 13px; color: #78350f; margin: 10px 0;
}

.shap-item {
    display: flex; align-items: center;
    background: #f7f9fc; border-radius: 10px;
    padding: 10px 14px; margin-bottom: 8px;
    border: 1px solid #e2e8f0;
}
.shap-rank { font-size: 13px; font-weight: 700; color: #2356a0; width: 22px; flex-shrink: 0; }
.shap-name { font-size: 13px; font-weight: 600; color: #1e293b; flex: 1; }
.shap-dir  { font-size: 12px; font-weight: 500; margin-left: 8px; }
.shap-val  { font-size: 12px; color: #64748b; margin-left: 8px; font-family: monospace; }

.empty-state {
    background: #f7f9fc; border-radius: 16px;
    padding: 50px 30px; text-align: center;
    border: 2px dashed #cbd5e1;
}

/* Streamlit overrides */
div[data-testid="stMetric"] {
    background: #ffffff; border-radius: 12px;
    padding: 14px 16px; border: 1px solid #e2e8f0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
div[data-testid="stMetricLabel"] p  { color: #64748b !important; font-size: 12px !important; font-weight: 500 !important; }
div[data-testid="stMetricValue"]    { color: #1a2f6e !important; font-size: 24px !important; font-weight: 700 !important; }

.stButton > button {
    background: linear-gradient(135deg, #1a2f6e, #2356a0) !important;
    color: white !important; border: none !important;
    border-radius: 10px !important; font-weight: 600 !important;
    font-size: 14px !important; padding: 12px 20px !important;
    box-shadow: 0 4px 14px rgba(35,86,160,0.30) !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; }

.stSelectbox label, .stNumberInput label {
    font-size: 13px !important; font-weight: 500 !important; color: #374151 !important;
}
div[data-baseweb="select"] > div {
    border-radius: 9px !important; border-color: #e2e8f0 !important;
    background: #ffffff !important;
}
div[data-testid="stNumberInput"] input {
    border-radius: 9px !important; border-color: #e2e8f0 !important;
}
button[data-baseweb="tab"] { font-size: 13.5px !important; font-weight: 500 !important; color: #64748b !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #1a2f6e !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

# ── Resources ──
@st.cache_resource
def load_model():
    return joblib.load("xgboost.pkl")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("heart.csv")
        df_enc = df.copy()
        df_enc["Sex"]            = (df_enc["Sex"] == "M").astype(int)
        df_enc["ExerciseAngina"] = (df_enc["ExerciseAngina"] == "Y").astype(int)
        return df, df_enc
    except:
        return None, None

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

model             = load_model()
df_raw, df_enc    = load_data()
explainer         = get_explainer(model)

FEATURE_COLS = [
    "Age","Sex","RestingBP","Cholesterol","FastingBS","MaxHR",
    "ExerciseAngina","Oldpeak",
    "ChestPainType_ASY","ChestPainType_ATA","ChestPainType_NAP","ChestPainType_TA",
    "RestingECG_LVH","RestingECG_Normal","RestingECG_ST",
    "ST_Slope_Down","ST_Slope_Flat","ST_Slope_Up"
]

def build_row(age, sex, resting_bp, cholesterol, fasting_bs,
              max_hr, exercise_angina, oldpeak, chest_pain, ecg, st_slope):
    row = {col: 0 for col in FEATURE_COLS}
    row.update({
        "Age": age, "Sex": 1 if sex == "Male" else 0,
        "RestingBP": resting_bp, "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "MaxHR": max_hr,
        "ExerciseAngina": 1 if exercise_angina == "Yes" else 0,
        "Oldpeak": oldpeak,
    })
    row[f"ChestPainType_{chest_pain}"] = 1
    row[f"RestingECG_{ecg}"]           = 1
    row[f"ST_Slope_{st_slope}"]        = 1
    return pd.DataFrame([row])

def make_gauge(val):
    if val >= 60:   bar_color, label = "#c0392b", "High Risk"
    elif val >= 40: bar_color, label = "#d97706", "Moderate Risk"
    else:           bar_color, label = "#16a34a", "Low Risk"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=val,
        number={"suffix": "%", "font": {"size": 40, "color": "#1a2f6e", "family": "Inter"}},
        title={"text": f"<b>{label}</b>",
               "font": {"size": 14, "color": bar_color, "family": "Inter"}},
        gauge={
            "axis": {"range": [0,100], "tickcolor": "#94a3b8",
                     "tickfont": {"color": "#64748b", "size": 11}, "tickwidth": 1},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "#f7f9fc", "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#dcfce7"},
                {"range": [40, 60], "color": "#fef9c3"},
                {"range": [60,100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "#1a2f6e","width": 3},
                          "thickness": 0.85, "value": val}
        }
    ))
    fig.update_layout(
        height=220, margin=dict(t=30,b=0,l=40,r=40),
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
    )
    return fig

# ── HEADER ──
st.markdown("""
<div class="hero-header">
  <p class="hero-title">❤️ Heart Disease Prediction</p>
  <p class="hero-sub">Explainable Machine Learning · XGBoost · SHAP Analysis</p>
  <span class="hero-badge">🏥 Academic Project</span>
  <span class="hero-badge">📊 Final Project ML</span>
  <span class="hero-badge">🔬 Explainable AI</span>
</div>
""", unsafe_allow_html=True)

# ── TABS ──
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Prediction & SHAP",
    "📊  Model Comparison",
    "📈  Data Insights (EDA)",
    "ℹ️  About",
])

# ══════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1.05, 1], gap="large")

    with col_input:
        st.markdown('<div class="sec-header">Patient Data Input</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age             = st.number_input("Age (years)", 1, 120, 50)
            resting_bp      = st.number_input("Resting Blood Pressure (mmHg)", 50, 250, 120)
            cholesterol     = st.number_input("Cholesterol (mg/dL)", 0, 700, 200)
            max_hr          = st.number_input("Max Heart Rate (bpm)", 50, 250, 150)
            oldpeak         = st.number_input("Oldpeak — ST Depression", 0.0, 10.0, 1.0, step=0.1)
        with c2:
            sex             = st.selectbox("Gender", ["Male","Female"])
            fasting_bs      = st.selectbox("Fasting Blood Sugar > 120 mg/dL", ["No","Yes"])
            exercise_angina = st.selectbox("Exercise Induced Angina", ["No","Yes"])
            chest_pain      = st.selectbox("Chest Pain Type", ["ASY","ATA","NAP","TA"],
                                           help="ASY: Asymptomatic · ATA: Atypical Angina · NAP: Non-Anginal · TA: Typical Angina")
            ecg             = st.selectbox("Resting ECG Result", ["Normal","LVH","ST"])
            st_slope        = st.selectbox("ST Slope", ["Flat","Up","Down"])

        predict_btn = st.button("🔍  Run Prediction", use_container_width=True)

    with col_result:
        st.markdown('<div class="sec-header">Prediction Result</div>', unsafe_allow_html=True)

        if predict_btn:
            df_pred    = build_row(age, sex, resting_bp, cholesterol, fasting_bs,
                                   max_hr, exercise_angina, oldpeak, chest_pain, ecg, st_slope)
            prob       = model.predict_proba(df_pred)[0]
            healthy    = prob[0] * 100
            disease    = prob[1] * 100
            prediction = model.predict(df_pred)[0]

            st.plotly_chart(make_gauge(disease), use_container_width=True, key="gauge")

            m1, m2 = st.columns(2)
            m1.metric("💚 Not Heart Disease", f"{healthy:.1f}%")
            m2.metric("❤️ Heart Disease Risk", f"{disease:.1f}%")

            if prediction == 1:
                risk = "High" if disease >= 70 else "Moderate"
                st.markdown(f'<div class="result-danger">❤️ Heart Disease Detected &nbsp;·&nbsp; {risk} Risk ({disease:.1f}%)</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="result-safe">💚 No Heart Disease Detected &nbsp;·&nbsp; {healthy:.1f}% Confidence</div>',
                            unsafe_allow_html=True)

            st.markdown('<div class="sec-header">Why This Prediction? (SHAP)</div>',
                        unsafe_allow_html=True)
            st.markdown('<div class="info-box">SHAP menjelaskan kontribusi setiap fitur. '
                        '<b>Merah</b> = mendorong risiko naik · <b>Biru</b> = menurunkan risiko.</div>',
                        unsafe_allow_html=True)

            shap_out = explainer(df_pred)
            sv = shap_out.values
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            sv = sv[0]
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = float(base_val[1]) if len(base_val) > 1 else float(base_val[0])

            explanation = shap.Explanation(
                values=sv, base_values=base_val,
                data=df_pred.iloc[0].values,
                feature_names=FEATURE_COLS
            )
            fig_shap, ax = plt.subplots(figsize=(7, 4.2))
            fig_shap.patch.set_facecolor("#ffffff")
            ax.set_facecolor("#ffffff")
            shap.waterfall_plot(explanation, max_display=10, show=False)
            for txt in ax.texts:   txt.set_color("#1e293b")
            ax.tick_params(colors="#1e293b")
            ax.xaxis.label.set_color("#1e293b")
            for sp in ax.spines.values(): sp.set_edgecolor("#e2e8f0")
            plt.tight_layout()
            st.pyplot(fig_shap, use_container_width=True)
            plt.close()

            st.markdown('<div class="sec-header">Top Influencing Factors</div>', unsafe_allow_html=True)
            top_idx = np.argsort(np.abs(sv))[::-1][:3]
            for i, idx in enumerate(top_idx):
                direction  = "🔴 Increases risk" if sv[idx] > 0 else "🟢 Reduces risk"
                dir_color  = "#c0392b" if sv[idx] > 0 else "#16a34a"
                st.markdown(f"""
                <div class="shap-item">
                  <span class="shap-rank">{i+1}.</span>
                  <span class="shap-name">{FEATURE_COLS[idx]}</span>
                  <span class="shap-dir" style="color:{dir_color};">{direction}</span>
                  <span class="shap-val">SHAP: {sv[idx]:+.3f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty-state">
              <div style="font-size:52px;margin-bottom:14px;">🫀</div>
              <div style="font-size:16px;font-weight:600;color:#1e293b;margin-bottom:6px;">Ready to Analyze</div>
              <div style="font-size:13px;color:#64748b;">Fill in patient data on the left,<br>then click <b>Run Prediction</b></div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 2 — MODEL COMPARISON
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-header">Model Performance Comparison</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
    Hasil evaluasi <b>Logistic Regression dan XGBoost pada data Heart Disease.</b>
    <b>PR-AUC</b> adalah metrik utama — lebih sensitif terhadap kemampuan deteksi kelas positif.
    </div>""", unsafe_allow_html=True)

    results_data = {
        "Model": [
            "⭐ XGBoost",
            "Logistic Regression",
        ],
          "PR-AUC":    [0.9580, 0.9450],
          "ROC-AUC":   [0.9482, 0.9347],
          "Accuracy":  [0.9216, 0.9126],
          "F1-Score":  [0.9216, 0.9126],
          "Precision": [0.9216, 0.9038],
          "Recall":    [0.9216, 0.9216]
    }
    df_res = pd.DataFrame(results_data)

    def highlight_best(row):
        if "⭐" in str(row["Model"]):
            return ["background-color:#dbeafe; font-weight:600; color:#1e3a8a"] * len(row)
        return ["color:#1e293b"] * len(row)

    styled = (df_res.style
        .apply(highlight_best, axis=1)
        .format({k:"{:.4f}" for k in ["PR-AUC","ROC-AUC","Accuracy","F1-Score","Precision","Recall"]})
        .bar(subset=["PR-AUC"], color=["#fee2e2","#dbeafe"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True, height=105)
    st.caption("⭐ Model terbaik yang digunakan pada tab Prediction.")

    cb1, cb2 = st.columns(2, gap="large")
    with cb1:
        st.markdown('<div class="sec-header">PR-AUC per Model</div>', unsafe_allow_html=True)
        colors = ["#1a2f6e" if "⭐" in m else "#93c5fd" for m in df_res["Model"]]
        fig_bar = go.Figure(go.Bar(
            x=df_res["PR-AUC"],
            y=[m.replace("⭐ ","") for m in df_res["Model"]],
            orientation="h", marker_color=colors, marker_line_width=0,
            text=[f"{v:.4f}" for v in df_res["PR-AUC"]],
            textposition="outside",
            textfont=dict(color="#1e293b", size=11, family="Inter")
        ))
        fig_bar.update_layout(
            height=280, margin=dict(t=10,b=10,l=10,r=60),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            xaxis=dict(color="#64748b", range=[0.88, 0.97], gridcolor="#f1f5f9"),
            yaxis=dict(color="#374151", autorange="reversed"),
            font=dict(family="Inter", color="#1e293b"),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="bar_prauc")

    with cb2:
        st.markdown('<div class="sec-header">Multi-Metric Radar</div>', unsafe_allow_html=True)
        metrics = ["PR-AUC","ROC-AUC","Accuracy","F1-Score"]
        palette = ["#1a2f6e","#2356a0","#3b82f6","#93c5fd","#cbd5e1","#94a3b8"]
        fig_rad = go.Figure()
        for i, row in df_res.iterrows():
            vals = [row[m] for m in metrics] + [row[metrics[0]]]
            fig_rad.add_trace(go.Scatterpolar(
                r=vals, theta=metrics+[metrics[0]],
                fill="toself", name=row["Model"].replace("⭐ ",""),
                line_color=palette[i], opacity=0.75, line_width=2
            ))
        fig_rad.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0.83,0.96],
                                color="#64748b", gridcolor="#e2e8f0"),
                angularaxis=dict(color="#374151", gridcolor="#e2e8f0")
            ),
            height=280, margin=dict(t=20,b=10,l=10,r=10),
            paper_bgcolor="#ffffff",
            font=dict(family="Inter", color="#1e293b"),
            legend=dict(font=dict(size=9.5, color="#374151"),
                        bgcolor="rgba(255,255,255,0.9)", bordercolor="#e2e8f0", borderwidth=1)
        )
        st.plotly_chart(fig_rad, use_container_width=True, key="radar")

    st.markdown('<div class="sec-header">Interpretasi Metrik Evaluasi</div>', unsafe_allow_html=True)
    ex1, ex2, ex3 = st.columns(3, gap="medium")
    for col, title, body, color, bg in [
        (ex1,"📊  Accuracy","Menunjukkan proporsi prediksi yang benar dari seluruh data. Semakin tinggi nilainya, semakin baik performa model secara keseluruhan.","#b45309","#fffbeb"),
        (ex2,"🎯  ROC-AUC","Mengukur kemampuan model membedakan pasien dengan dan tanpa penyakit jantung pada berbagai threshold klasifikasi.","#b45309","#fffbeb"),
        (ex3,"⭐  XGBoost Terbaik","XGBoost menghasilkan ROC-AUC 0.9482 dan PR-AUC 0.9580, lebih tinggi dibanding Logistic Regression sehingga dipilih sebagai model prediksi pada dashboard ini.","#15803d","#f0fdf4"),
    ]:
        col.markdown(f"""
        <div style="background:{bg};border-radius:12px;padding:16px 18px;
                    border:1px solid #e2e8f0;height:100%;">
          <div style="font-size:13px;font-weight:700;color:{color};margin-bottom:8px;">{title}</div>
          <div style="font-size:12.5px;color:#374151;line-height:1.6;">{body}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
# TAB 3 — EDA
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if df_raw is None:
        st.markdown('<div class="info-box-warn">⚠️ File <code>heart.csv</code> tidak ditemukan. Letakkan di folder yang sama dengan app.py.</div>',
                    unsafe_allow_html=True)
    else:
        total = len(df_raw)
        n_pos = df_raw["HeartDisease"].sum()
        n_neg = total - n_pos

        s1,s2,s3,s4 = st.columns(4)
        s1.metric("Total Pasien", f"{total:,}")
        s2.metric("Sakit Jantung", f"{n_pos} ({n_pos/total*100:.1f}%)")
        s3.metric("Tidak Sakit", f"{n_neg} ({n_neg/total*100:.1f}%)")
        s4.metric("Rata-rata Usia", f"{df_raw['Age'].mean():.0f} tahun")

        st.divider()

        r1, r2 = st.columns(2, gap="large")
        with r1:
            st.markdown('<div class="sec-header">Distribusi Target</div>', unsafe_allow_html=True)
            fig_pie = go.Figure(go.Pie(
                labels=["Tidak Sakit Jantung","Sakit Jantung"],
                values=[n_neg, n_pos], hole=0.58,
                marker_colors=["#2356a0","#c0392b"],
                textinfo="label+percent",
                textfont=dict(color="#1e293b", size=12, family="Inter"),
            ))
            fig_pie.update_layout(
                height=260, margin=dict(t=10,b=10,l=10,r=10),
                paper_bgcolor="#ffffff",
                font=dict(color="#1e293b", family="Inter"),
                legend=dict(font=dict(color="#374151"))
            )
            st.plotly_chart(fig_pie, use_container_width=True, key="pie")

        with r2:
            st.markdown('<div class="sec-header">Distribusi Usia per Kelas</div>', unsafe_allow_html=True)
            fig_age = go.Figure()
            for label, color, name in [(0,"#2356a0","Tidak Sakit"),(1,"#c0392b","Sakit Jantung")]:
                fig_age.add_trace(go.Histogram(
                    x=df_raw[df_raw["HeartDisease"]==label]["Age"],
                    name=name, marker_color=color, opacity=0.72, nbinsx=20
                ))
            fig_age.update_layout(
                barmode="overlay", height=260,
                margin=dict(t=10,b=10,l=10,r=10),
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                xaxis=dict(color="#374151", title="Usia", gridcolor="#f1f5f9"),
                yaxis=dict(color="#374151", gridcolor="#f1f5f9"),
                font=dict(family="Inter", color="#1e293b"),
                legend=dict(font=dict(color="#374151"), bgcolor="rgba(255,255,255,0.9)")
            )
            st.plotly_chart(fig_age, use_container_width=True, key="age_hist")

        st.divider()
        st.markdown('<div class="sec-header">Distribusi Fitur Numerik per Kelas</div>', unsafe_allow_html=True)
        feat_sel = st.selectbox("Pilih fitur:", ["Age","RestingBP","Cholesterol","MaxHR","Oldpeak"])
        fig_box = go.Figure()
        for label, color, name in [(0,"#2356a0","Tidak Sakit"),(1,"#c0392b","Sakit Jantung")]:
            fig_box.add_trace(go.Box(
                y=df_raw[df_raw["HeartDisease"]==label][feat_sel],
                name=name, marker_color=color, boxmean=True, line_color=color
            ))
        fig_box.update_layout(
            height=300, margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            yaxis=dict(color="#374151", title=feat_sel, gridcolor="#f1f5f9"),
            xaxis=dict(color="#374151"),
            font=dict(family="Inter", color="#1e293b"),
            legend=dict(font=dict(color="#374151"), bgcolor="rgba(255,255,255,0.9)")
        )
        st.plotly_chart(fig_box, use_container_width=True, key="boxplot")

        st.divider()
        st.markdown('<div class="sec-header">Persentase Sakit Jantung per Kategori</div>', unsafe_allow_html=True)
        cat_sel = st.selectbox("Pilih fitur kategorikal:",
                               ["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope","FastingBS"])
        ct = pd.crosstab(df_raw[cat_sel], df_raw["HeartDisease"], normalize="index") * 100
        ct.columns = ["Tidak Sakit (%)","Sakit Jantung (%)"]
        ct = ct.reset_index()
        fig_cat = go.Figure()
        for col_name, color in [("Tidak Sakit (%)","#2356a0"),("Sakit Jantung (%)","#c0392b")]:
            fig_cat.add_trace(go.Bar(
                x=ct[cat_sel].astype(str), y=ct[col_name],
                name=col_name.replace(" (%)",""),
                marker_color=color, marker_line_width=0
            ))
        fig_cat.update_layout(
            barmode="group", height=300,
            margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            yaxis=dict(color="#374151", title="%", gridcolor="#f1f5f9"),
            xaxis=dict(color="#374151"),
            font=dict(family="Inter", color="#1e293b"),
            legend=dict(font=dict(color="#374151"), bgcolor="rgba(255,255,255,0.9)")
        )
        st.plotly_chart(fig_cat, use_container_width=True, key="cat_bar")

        st.divider()
        st.markdown('<div class="sec-header">Korelasi Fitur Numerik dengan Target</div>', unsafe_allow_html=True)
        num_cols = ["Age","RestingBP","Cholesterol","MaxHR","Oldpeak","FastingBS","HeartDisease"]
        corr = df_enc[num_cols].corr()
        fig_corr = px.imshow(
            corr, text_auto=".2f",
            color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto"
        )
        fig_corr.update_traces(textfont=dict(color="#1e293b", size=11))
        fig_corr.update_layout(
            height=380, margin=dict(t=10,b=10,l=10,r=10),
            paper_bgcolor="#ffffff",
            font=dict(family="Inter", color="#1e293b"),
            xaxis=dict(color="#374151"),
            yaxis=dict(color="#374151"),
            coloraxis_colorbar=dict(tickfont=dict(color="#374151"),
                                    title=dict(font=dict(color="#374151")))
        )
        st.plotly_chart(fig_corr, use_container_width=True, key="corr")

# ══════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-header">Tentang Penelitian</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2, gap="large")

    with a1:
        st.markdown("""
        <div style="background:#ffffff;border-radius:14px;padding:20px 22px;
                    border:1px solid #e2e8f0;box-shadow:0 2px 12px rgba(0,0,0,0.06);margin-bottom:14px;">
          <div style="font-size:11px;font-weight:600;color:#64748b;letter-spacing:0.8px;
                      text-transform:uppercase;margin-bottom:10px;">Judul</div>
          <div style="font-size:14px;font-weight:600;color:#1a2f6e;line-height:1.6;">
            Prediksi Risiko Penyakit Jantung dengan Pendekatan Explainable Machine Learning:
            Perbandingan Logistic Regression dan XGBoost
          </div>
        </div>
        <div style="background:#ffffff;border-radius:14px;padding:20px 22px;
                    border:1px solid #e2e8f0;box-shadow:0 2px 12px rgba(0,0,0,0.06);">
          <div style="font-size:11px;font-weight:600;color:#64748b;letter-spacing:0.8px;
                      text-transform:uppercase;margin-bottom:10px;">Dataset</div>
          <div style="font-size:13.5px;color:#374151;line-height:2.0;">
            📌 Heart Failure Prediction Dataset (Kaggle)<br>
            📌 918 baris · 11 fitur · Distribusi 55% : 45%<br>
            📌 Gabungan 5 sumber: Cleveland, Hungary, Switzerland, Long Beach, Stalog
          </div>
        </div>
        """, unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div style="background:#ffffff;border-radius:14px;padding:20px 22px;
                    border:1px solid #e2e8f0;box-shadow:0 2px 12px rgba(0,0,0,0.06);margin-bottom:14px;">
          <div style="font-size:11px;font-weight:600;color:#64748b;letter-spacing:0.8px;
                      text-transform:uppercase;margin-bottom:10px;">Metodologi</div>
          <div style="font-size:13.5px;color:#374151;line-height:2.0;">
            ✅ &nbsp;2 Model: Logistic Regression · XGBoost<br>
            ✅ &nbsp;Hyperparameter Tuning: GridSearchCV & Optuna<br>
            ✅ &nbsp;Train-Test Split: 80% · 20%<br>
            ✅ &nbsp;Evaluasi: ROC-AUC · PR-AUC · F1 · Precision · Recall<br>
            ✅ &nbsp;Explainability: SHAP
          </div>
        </div>
        <div style="background:linear-gradient(135deg,#dbeafe,#eff6ff);border-radius:14px;
                    padding:20px 22px;border:1px solid #bfdbfe;">
          <div style="font-size:11px;font-weight:600;color:#1e40af;letter-spacing:0.8px;
                      text-transform:uppercase;margin-bottom:6px;">Model Terbaik</div>
          <div style="font-size:24px;font-weight:700;color:#1a2f6e;">⭐ XGBoost</div>
          <div style="font-size:12.5px;color:#1e40af;margin-top:4px;">
            PR-AUC: 0.9580 &nbsp;·&nbsp; ROC-AUC: 0.9482 &nbsp;·&nbsp; Accuracy: 92.16%
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box-warn" style="margin-top:16px;">
    ⚠️ <b>Disclaimer:</b> Dashboard ini dibuat untuk keperluan akademik (Final Project Machine Learning).
    Model tidak dimaksudkan untuk diagnosis medis. Konsultasikan dengan tenaga medis profesional.
    </div>
    """, unsafe_allow_html=True)
