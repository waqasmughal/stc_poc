import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime, timedelta
import openai

import os
from dotenv import load_dotenv

load_dotenv()  # Load .env variables
openai.api_key = os.getenv("OPENAPI_KEY")
number = 4

# Function to fetch stock data from Yahoo Finance
def fetch_data_from_yf(company, start_date):
    df = yf.download(company, start=start_date)
    df = df[['Close']]  
    return df

# Function to create sequences for the LSTM model
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])  
        y.append(data[i, 0])  
    return np.array(X), np.array(y)

# Function to scale data using the saved MinMaxScaler
def scale_data(df, scaler):
    return scaler.transform(df)

# Function to predict next 40 days' stock prices
def predict_next_40_days(company, model_path, scaler_path, number, seq_length=60):
    start_date = '2024-11-01'  
    df = fetch_data_from_yf(company, start_date)

    scaler = joblib.load(scaler_path)
    model = load_model(model_path)

    scaled_data = scale_data(df, scaler)
    X, _ = create_sequences(scaled_data)

    input_sequence = scaled_data[-seq_length:]
    input_sequence = np.reshape(input_sequence, (1, seq_length, 1))  

    last_date = df.index[-1]

    predictions, prediction_dates = [], []

    for i in range(number + 1):
        prediction = model.predict(input_sequence)
        predictions.append(prediction[0][0])

        next_day = last_date + timedelta(days=i+1)
        prediction_dates.append(next_day.strftime('%Y-%m-%d'))

        prediction_reshaped = np.reshape(prediction, (1, 1, 1))
        input_sequence = np.append(input_sequence[:, 1:, :], prediction_reshaped, axis=1)

    predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return prediction_dates, predicted_prices.flatten()

# Function to generate investment advice
def generate_investment_advice(predictions, company_name):
    trend = "upward" if predictions[-1] > predictions[0] else "downward"

    prompt = (
        f"The predicted closing prices for {company_name} for the next {number} days are: "
        f"{', '.join([f'${price:.2f}' for price in predictions])}. "
        "Based on this prediction, should investors buy, hold, or sell the stock? "
        f"The stock price trend appears to be {trend}. Provide specific reasons for your recommendation."
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a financial market analyst."},
                  {"role": "user", "content": prompt}],
        max_tokens=300,
    )
    
    return response['choices'][0]['message']['content'].strip()
