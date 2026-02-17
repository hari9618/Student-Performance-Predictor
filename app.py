import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(page_title="Student Performance Predictor", layout="wide")

st.title("🎓 Student Marks Predictor (ML Web App)")
st.markdown("Upload dataset → Train model → Predict marks → Evaluate performance")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------
st.sidebar.header("Upload Dataset")
file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

if file is not None:

    df = pd.read_csv(file)
    df.columns = df.columns.str.strip()

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)

    # ---------------------------------------------------
    # COLUMN SELECTION
    # ---------------------------------------------------
    st.sidebar.subheader("Select Columns")

    feature_col = st.sidebar.selectbox("Feature (Study Hours)", df.columns)
    target_col = st.sidebar.selectbox("Target (Marks)", df.columns, index=1)

    X = df[[feature_col]]
    y = df[target_col]

    # ---------------------------------------------------
    # TRAIN MODEL
    # ---------------------------------------------------
    model = LinearRegression()
    model.fit(X, y)

    predictions = model.predict(X)

    r2 = r2_score(y, predictions)
    mae = mean_absolute_error(y, predictions)

    # ---------------------------------------------------
    # METRICS
    # ---------------------------------------------------
    st.subheader("📊 Model Performance")

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("Mean Absolute Error", f"{mae:.2f}")

    # ---------------------------------------------------
    # PREDICTION
    # ---------------------------------------------------
    st.subheader("🔮 Predict Marks")

    hours = st.slider("Enter Study Hours", 0.0, 16.0, 4.0)

    predicted_mark = model.predict([[hours]])[0]

    st.success(f"Predicted Marks: {predicted_mark:.2f}")

    # ---------------------------------------------------
    # GRAPH
    # ---------------------------------------------------
    st.subheader("📈 Regression Visualization")

    fig, ax = plt.subplots()

    ax.scatter(X, y, label="Actual Data")

    x_range = np.linspace(X.min(), X.max(), 100)
    y_range = model.predict(x_range.reshape(-1,1))

    ax.plot(x_range, y_range, color="red", label="Regression Line")
    ax.scatter(hours, predicted_mark, color="green", s=120, label="Your Prediction")

    ax.set_xlabel(feature_col)
    ax.set_ylabel(target_col)
    ax.legend()

    st.pyplot(fig)

    # ---------------------------------------------------
    # DOWNLOAD MODEL
    # ---------------------------------------------------
    st.subheader("💾 Download Trained Model")

    model_bytes = pickle.dumps(model)
    st.download_button(
        label="Download Model (.pkl)",
        data=model_bytes,
        file_name="student_marks_model.pkl"
    )

else:
    st.info("👆 Upload a CSV file to start")

