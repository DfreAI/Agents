import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet  # Updated package

# Set up page configuration
st.set_page_config(page_title="Revenue Forecasting with Prophet", page_icon="📊", layout="wide")

# Title of the app
st.title("🚀 AI-Powered Revenue Forecasting with Prophet Algorithm")

# Upload Excel file
uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx", "xls"])

if uploaded_file is not None:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)

    # Check if required columns 'Date' and 'Revenue' exist
    if 'Date' in df.columns and 'Revenue' in df.columns:
        # Convert 'Date' to datetime and 'Revenue' to numeric
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Revenue'] = pd.to_numeric(df['Revenue'], errors='coerce')

        # Drop rows with missing values
        df.dropna(subset=['Date', 'Revenue'], inplace=True)

        # Rename columns for Prophet compatibility
        df = df.rename(columns={'Date': 'ds', 'Revenue': 'y'})

        # Display data preview
        st.subheader("Data Preview")
        st.write(df.head())

        # Plot the time series data
        st.subheader("Time Series Plot")
        plt.figure(figsize=(10, 6))
        plt.plot(df['ds'], df['y'], marker='o', linestyle='-', color='b')
        plt.title("Revenue over Time")
        plt.xlabel("Date")
        plt.ylabel("Revenue")
        st.pyplot(plt)

        # Prophet model for forecasting
        st.subheader("Forecasting with Prophet Algorithm")

        try:
            # Initialize Prophet model
            model = Prophet(daily_seasonality=True, yearly_seasonality=True)
            model.fit(df)

            # Make future data frame for prediction (ensure periods is passed correctly)
            future = model.make_future_dataframe(df, periods=365)  # Forecasting for the next year

            # Predict
            forecast = model.predict(future)

            # Plot the forecast
            st.subheader("Forecasted Revenue")
            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            # Display forecast components (trend, yearly, weekly)
            st.subheader("Forecast Components")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"An error occurred while making predictions: {str(e)}")

    else:
        st.error("The uploaded file must contain 'Date' and 'Revenue' columns.")

else:
    st.info("Please upload an Excel file to begin.")
