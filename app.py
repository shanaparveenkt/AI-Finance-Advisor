import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Finance Advisor",
    page_icon="💰",
    layout="wide"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

/* Background */
.main {
    background-color: #f5f7fa;
}

/* Metric Card */
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    padding: 15px !important;
    border-radius: 12px !important;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08) !important;
}

/* Force ALL text inside metric to dark */
div[data-testid="stMetric"] * {
    color: #000000 !important;
}

/* Label (small text) */
div[data-testid="stMetricLabel"] {
    color: #555 !important;
    font-weight: 600 !important;
}

/* Value (big number) */
div[data-testid="stMetricValue"] {
    color: #000000 !important;
    font-size: 26px !important;
    font-weight: bold !important;
}

/* Delta (if appears) */
div[data-testid="stMetricDelta"] {
    color: #333 !important;
}

/* Headings */
h1, h2, h3 {
    color: #2c3e50;
}

</style>
""", unsafe_allow_html=True)
# ------------------ SIDEBAR ------------------
st.sidebar.title("💼 Finance Dashboard")
st.sidebar.markdown("Upload your dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# ------------------ TITLE ------------------
st.markdown("## 💰 AI-Powered Personal Finance Advisor")
st.markdown("### Smart insights • Predictions • Financial Health")
st.divider()

# ------------------ MAIN APP ------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # ------------------ DATA CLEANING ------------------
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')

    df['month'] = df['date'].dt.month
    df['day_name'] = df['date'].dt.day_name()

    # ------------------ CATEGORIZATION ------------------
    def categorize(desc):
        desc = desc.lower()
        if "swiggy" in desc or "zomato" in desc:
            return "Food"
        elif "uber" in desc or "ola" in desc:
            return "Travel"
        elif "amazon" in desc or "flipkart" in desc:
            return "Shopping"
        elif "milk" in desc or "groceries" in desc:
            return "Essentials"
        elif "electricity" in desc or "recharge" in desc:
            return "Bills"
        elif "salary" in desc:
            return "Income"
        else:
            return "Others"

    df['category'] = df['description'].apply(categorize)

    expenses = df[df['category'] != "Income"]

    # ------------------ KPI METRICS ------------------
    total_spending = expenses['amount'].sum()
    total_income = df['income'].sum()
    savings = total_income - total_spending

    col1, col2, col3 = st.columns(3)

    col1.metric("💸 Total Spending", f"₹{total_spending}")
    col2.metric("💰 Total Income", f"₹{total_income}")
    col3.metric("📈 Savings", f"₹{savings}")

    st.divider()

    # ------------------ CATEGORY ANALYSIS ------------------
    st.markdown("### 📊 Category-wise Spending")

    category_spending = expenses.groupby('category')['amount'].sum()
    st.bar_chart(category_spending)

    # ------------------ WEEKEND VS WEEKDAY ------------------
    weekend = expenses[expenses['day_name'].isin(['Saturday','Sunday'])]
    weekday = expenses[~expenses['day_name'].isin(['Saturday','Sunday'])]

    weekend_avg = weekend['amount'].mean()
    weekday_avg = weekday['amount'].mean()

    st.markdown("### 📅 Spending Behavior")

    col1, col2 = st.columns(2)
    col1.metric("Weekend Avg", round(weekend_avg, 2))
    col2.metric("Weekday Avg", round(weekday_avg, 2))

    st.divider()

    # ------------------ PREDICTION ------------------
    st.markdown("### 📈 Future Spending Prediction")

    model_df = expenses.copy().sort_values('date')
    model_df['day_index'] = range(len(model_df))

    X = model_df[['day_index']]
    y = model_df['amount']

    model = LinearRegression()
    model.fit(X, y)

    future_days = np.array(range(len(model_df), len(model_df)+7)).reshape(-1,1)
    predictions = model.predict(future_days)

    chart_data = list(y) + list(predictions)
    st.line_chart(chart_data)

    st.divider()

    # ------------------ RECOMMENDATIONS ------------------
    st.markdown("### 💡 AI Recommendations")

    recommendations = []

    if total_spending > total_income:
        recommendations.append("⚠️ Expenses exceed income. Control spending immediately.")

    food_spend = category_spending.get('Food', 0)
    if food_spend / total_spending > 0.3:
        recommendations.append("🍔 High food spending detected. Try reducing online orders.")

    if weekend['amount'].sum() > weekday['amount'].sum():
        recommendations.append("📅 You spend more on weekends. Plan a budget.")

    if savings > total_income * 0.2:
        recommendations.append("🌟 Excellent saving habit! Keep it up.")

    if savings < 0:
        recommendations.append("💸 You are running a deficit. Immediate action needed.")

    for r in recommendations:
        st.success(r)

    st.divider()

    # ------------------ FINANCIAL SCORE ------------------
    st.markdown("### 💰 Financial Health Score")

    score = 100

    if total_spending > total_income:
        score -= 30

    if food_spend / total_spending > 0.4:
        score -= 15

    if savings < 0:
        score -= 20

    st.progress(score / 100)
    st.write(f"### {score} / 100")

else:
    st.info("👈 Upload your CSV file from the sidebar to get started")