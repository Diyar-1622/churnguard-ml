"""
ChurnGuard — Customer Churn Prediction Dashboard
Run: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ChurnGuard", page_icon="shield", layout="wide")

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 800
    df = pd.DataFrame({
        "tenure": np.random.randint(1, 72, n),
        "MonthlyCharges": np.round(np.random.uniform(20, 120, n), 2),
        "TotalCharges": np.round(np.random.uniform(50, 8000, n), 2),
        "Contract": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20]),
        "InternetService": np.random.choice(["DSL", "Fiber optic", "No"], n),
        "PaymentMethod": np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
        "SeniorCitizen": np.random.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": np.random.choice(["Yes", "No"], n),
        "Dependents": np.random.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "TechSupport": np.random.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": np.random.choice(["Yes", "No", "No internet service"], n),
    })
    churn_prob = (
        0.05
        + (df["Contract"] == "Month-to-month") * 0.30
        + (df["InternetService"] == "Fiber optic") * 0.15
        + (df["MonthlyCharges"] > 80) * 0.20
        + (df["tenure"] < 12) * 0.15
        + (df["PaymentMethod"] == "Electronic check") * 0.10
        - (df["tenure"] > 48) * 0.20
    ).clip(0, 1)
    df["Churn"] = (np.random.rand(n) < churn_prob).astype(int)
    return df

@st.cache_resource
def train_model(df):
    le = LabelEncoder()
    df_enc = df.copy()
    cat_cols = ["Contract", "InternetService", "PaymentMethod", "Partner", "Dependents", "TechSupport", "StreamingTV"]
    for col in cat_cols:
        df_enc[col] = le.fit_transform(df_enc[col])
    feature_cols = [c for c in df_enc.columns if c != "Churn"]
    X, y = df_enc[feature_cols], df_enc["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, feature_cols, X_test, y_test, auc

df = load_data()
model, feature_cols, X_test, y_test, auc = train_model(df)

st.title("ChurnGuard - Customer Churn Prediction")
st.caption("Built by Diyar Batyrkhan | XGBoost + SHAP + Streamlit")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["EDA & Insights", "Predict Churn", "Model Performance"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    churn_rate = df["Churn"].mean()
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churned", f"{df['Churn'].sum():,}", f"{churn_rate:.1%}")
    col3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.0f}")
    col4.metric("Avg Tenure", f"{df['tenure'].mean():.0f} months")
    st.markdown("### Churn by Contract Type")
    churn_by_contract = df.groupby("Contract")["Churn"].mean().reset_index()
    fig = px.bar(churn_by_contract, x="Contract", y="Churn", color="Churn",
                 color_continuous_scale="RdYlGn_r", text_auto=".1%")
    st.plotly_chart(fig, use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### Monthly Charges Distribution")
        fig2 = px.histogram(df, x="MonthlyCharges",
                            color=df["Churn"].map({0: "Retained", 1: "Churned"}),
                            nbins=30, barmode="overlay", opacity=0.7,
                            color_discrete_map={"Retained": "#2ecc71", "Churned": "#e74c3c"})
        st.plotly_chart(fig2, use_container_width=True)
    with col_b:
        st.markdown("### Feature Importances")
        importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig4 = px.bar(importances.tail(8), orientation="h", color=importances.tail(8), color_continuous_scale="Blues")
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.markdown("### Enter Customer Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        tenure = st.slider("Tenure (months)", 1, 72, 12)
        monthly = st.number_input("Monthly Charges ($)", 20.0, 120.0, 65.0, step=5.0)
        senior = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
    with col2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    with col3:
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

    if st.button("Predict Churn Risk", type="primary", use_container_width=True):
        def encode_cat(val, col_name):
            mapping = {v: i for i, v in enumerate(sorted(set(df[col_name].tolist())))}
            return mapping.get(val, 0)
        input_dict = {
            "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": tenure * monthly,
            "SeniorCitizen": senior, "Contract": encode_cat(contract, "Contract"),
            "InternetService": encode_cat(internet, "InternetService"),
            "PaymentMethod": encode_cat(payment, "PaymentMethod"),
            "Partner": encode_cat(partner, "Partner"), "Dependents": encode_cat(dependents, "Dependents"),
            "TechSupport": encode_cat(tech_support, "TechSupport"),
            "StreamingTV": encode_cat("No", "StreamingTV"),
        }
        input_df = pd.DataFrame([input_dict])[feature_cols]
        prob = model.predict_proba(input_df)[0][1]
        color = "#e74c3c" if prob > 0.6 else "#f39c12" if prob > 0.35 else "#2ecc71"
        risk = "HIGH RISK" if prob > 0.6 else "MEDIUM RISK" if prob > 0.35 else "LOW RISK"
        st.markdown(
            "<div style='text-align:center;padding:30px;border-radius:12px;background:" + color + "22;border:2px solid " + color + "'>"
            "<div style='font-size:52px;font-weight:bold;color:" + color + "'>" + f"{prob:.0%}" + "</div>"
            "<div style='font-size:22px;color:" + color + "'>" + risk + "</div></div>",
            unsafe_allow_html=True
        )

with tab3:
    st.markdown(f"### Gradient Boosting | AUC: **{auc:.3f}**")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={auc:.3f}", line=dict(color="#3498db", width=2)))
    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash="dash", color="#999")))
    fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=400)
    st.plotly_chart(fig_roc, use_container_width=True)

st.caption("Built by Diyar Batyrkhan | github.com/Diyar-1622/churnguard-ml")
