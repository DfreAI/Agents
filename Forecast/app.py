import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
import os
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# **🎨 Streamlit UI Styling**
st.set_page_config(page_title="Revenue Forecasting with Prophet", page_icon="📊", layout="wide")

# **File Upload Interface**
st.title("Revenue Forecasting using Prophet Algorithm")

uploaded_file = st.file_uploader("Upload Excel File with Date and Revenue", type=["xlsx"])

if uploaded_file is not None:
    try:
        # Load the Excel file into a DataFrame
        df = pd.read_excel(uploaded_file)

        # Check if necessary columns exist
        if 'Date' not in df.columns or 'Revenue' not in df.columns:
            st.error("🚨 The uploaded file must contain 'Date' and 'Revenue' columns.")
            st.stop()

        # Preprocess the data
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.rename(columns={'Date': 'ds', 'Revenue': 'y'})

        # **Prophet Forecasting**
        model = Prophet()
        model.fit(df)

        # Make future predictions
        future = model.make_future_dataframe(df, periods=365)  # Forecast 365 days ahead
        forecast = model.predict(future)

        # **Display Results**
        st.subheader("Forecasting Results")
        st.write("The model forecasts revenue trends for the next year.")

        # Plot forecast
        fig = model.plot(forecast)
        st.pyplot(fig)

        # Plot forecast components
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"🚨 An error occurred while processing the file: {e}")
        st.stop()

# Additional Code for Commentary (similar to FP&A example)
# You can integrate this part to provide insights based on the forecast results

# For instance, generate an AI commentary based on forecasting insights
# st.subheader("🤖 AI-Generated Forecast Commentary")
# ai_commentary = "Your insights would be generated here based on the results"
# st.write(ai_commentary)
