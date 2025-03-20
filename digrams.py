import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import yfinance as yf
import numpy as np
import pandas as pd
import os

matplotlib.use('Agg')  # Use non-interactive backend

# Function to fetch stock data
def fetch_data_from_yf_for_images(company, start_date):
    df = yf.download(company, start=start_date)

    # Make sure to set the directory for saving images
    image_dir = f"static/images/{company}"

    # Ensure the image directory exists
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # Fix multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(company, axis=1, level=1, drop_level=True)

    if df.empty:
        raise ValueError("Stock data is empty. Check the ticker symbol or date range.")

    if "Close" not in df.columns or "Volume" not in df.columns:
        raise KeyError("Missing expected columns in stock data")

    df = df[["Close", "Volume"]].copy()

    df["Close"] = df["Close"].squeeze()
    df["Volume"] = df["Volume"].squeeze()

    # Save the plots as images
    plot_histogram(df, company, image_dir)
    detect_outliers(df, company, image_dir)
    plot_moving_average(df, company, image_dir)
    detect_volume_outliers(df, company, image_dir)

    return df


# Save the histogram plot
def plot_histogram(df, company, image_dir):
    plt.figure(figsize=(10, 5))
    sns.histplot(df["Close"], bins=30, kde=True, color="blue", alpha=0.7)
    plt.xlabel("Closing Price")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {company} Stock Closing Prices")
    plt.savefig(f"{image_dir}/histogram_{company}.png")
    plt.close()


# Save the outliers plot
def detect_outliers(df, company, image_dir):
    Q1 = df["Close"].quantile(0.25)
    Q3 = df["Close"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df["Outlier"] = (df["Close"] < lower_bound) | (df["Close"] > upper_bound)

    plt.figure(figsize=(12, 5))
    sns.scatterplot(x=df.index, y=df["Close"], hue=df["Outlier"], palette={True: "red", False: "blue"}, alpha=0.6)
    plt.axhline(y=upper_bound, color="red", linestyle="dashed", label="Upper Bound")
    plt.axhline(y=lower_bound, color="red", linestyle="dashed", label="Lower Bound")
    plt.title(f"Stock Closing Prices with Outliers Highlighted for {company}")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend(["Upper Bound", "Lower Bound", "Outlier", "Normal"])
    plt.savefig(f"{image_dir}/outliers_{company}.png")
    plt.close()


# Save the moving averages plot
def plot_moving_average(df, company, image_dir):
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["SMA_50"] = df["Close"].rolling(window=50).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], label="Closing Price", color="blue", alpha=0.6)
    plt.plot(df.index, df["SMA_10"], label="10-Day SMA", color="red", linestyle="dashed")
    plt.plot(df.index, df["SMA_50"], label="50-Day SMA", color="green", linestyle="dashed")
    plt.title(f"Stock Closing Price with Moving Averages for {company}")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.savefig(f"{image_dir}/moving_average_{company}.png")
    plt.close()


# Save the volume outliers plot
def detect_volume_outliers(df, company, image_dir):
    Q1 = df["Volume"].quantile(0.25)
    Q3 = df["Volume"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df["Volume_Outlier"] = (df["Volume"] < lower_bound) | (df["Volume"] > upper_bound)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=df.index, y=df["Volume"], hue=df["Volume_Outlier"], palette={True: "red", False: "blue"}, alpha=0.6)
    plt.axhline(y=upper_bound, color="red", linestyle="dashed", label="Upper Bound")
    plt.axhline(y=lower_bound, color="red", linestyle="dashed", label="Lower Bound")
    plt.title(f"Stock Volume with Outliers Highlighted for {company}")
    plt.xlabel("Date")
    plt.ylabel("Trade Volume")
    plt.legend(["Upper Bound", "Lower Bound", "Outlier", "Normal"])
    plt.savefig(f"{image_dir}/volume_outliers_{company}.png")
    plt.close()


# Example: Fetch Data and Generate Plots for a Company
# company = "AAPL"  # Apple stock
# start_date = "2024-01-01"
# fetch_data_from_yf_for_images(company, start_date)
