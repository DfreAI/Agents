import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# File Upload Interface
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

        # Initialize Prophet model
        model = Prophet()

        # Fit the model
        model.fit(df)

        # Create future dataframe - forecast 365 days ahead (ensure you only pass 'periods' once)
        future = model.make_future_dataframe(df, periods=365)  # Forecast 365 days ahead

        # Predict future values
        forecast = model.predict(future)

        # Display the results
        st.subheader("Forecasting Results")
        st.write("The model forecasts revenue trends for the next year.")

        # Plot forecast
        fig = model.plot(forecast)
        st.pyplot(fig)

        # Plot forecast components (trends, seasonality, etc.)
        fig2 = model.plot_components(forecast)
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"🚨 An error occurred while processing the file: {e}")
        st.stop()
