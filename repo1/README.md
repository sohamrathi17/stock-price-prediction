# Stock Price Prediction with LSTM

This project predicts the next-day closing price of a stock ticker using a stacked LSTM neural network trained on historical OHLCV data and engineered time-series features.

---

## Problem Statement

Given the last 60 days of stock market data, the goal is to predict the next day’s closing price and evaluate whether the model performs better than a naive baseline where tomorrow’s price is assumed to be equal to today’s price.

---

## Project Structure

stock-price-prediction/
├── stock_price_prediction_code.ipynb
├── requirements.txt
└── README.md

- stock_price_prediction_code.ipynb contains the full workflow including data analysis, feature engineering, model training, and evaluation
- requirements.txt lists all required Python dependencies

---

## Workflow Overview

The pipeline begins by fetching stock data using yfinance for a selected ticker such as AAPL from 2020 to the present.

Feature engineering is then applied to generate additional signals such as returns, moving averages, and volatility.

The dataset is split into training, validation, and test sets in a 70/15/15 ratio.

A StandardScaler is fitted on the training data and applied to validation and test sets to prevent data leakage.

The data is then converted into sequences using a 60-day lookback window.

These sequences are used to train a stacked LSTM model implemented in TensorFlow/Keras.

Finally, the model is evaluated and compared against a naive baseline.

---

## Features Engineered

Feature           Description
Close             Raw closing price
Volume            Trading volume
Return_1d         Daily percentage return
MA_7              7-day moving average
MA_21             21-day moving average
Volatility_7      7-day rolling return standard deviation

---

## Model Architecture

The model takes an input of 60 timesteps with 6 features and processes it through two LSTM layers followed by dense layers.

- LSTM layer with 64 units and return sequences enabled
- Dropout layer with rate 0.2
- LSTM layer with 32 units
- Dropout layer with rate 0.2
- Dense layer with 64 units and ReLU activation
- Dropout layer with rate 0.2
- Final Dense layer with 1 unit to predict the next-day closing price

Training configuration:
- Optimizer: Adam
- Loss function: Mean Squared Error
- Callbacks: EarlyStopping with patience 10, ModelCheckpoint

---

## Evaluation Metrics

Metric                 Description
RMSE                   Root Mean Squared Error
MAE                    Mean Absolute Error
MAPE                   Mean Absolute Percentage Error
Directional Accuracy   Percentage of correctly predicted price movement direction

---

## How to Run

Clone the repository:
git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
cd stock-price-prediction

Install dependencies:
pip install -r requirements.txt

Launch the notebook:
jupyter notebook stock_price_prediction_code.ipynb

To change the stock ticker, update the following line in the notebook:
ticker = "TSLA"

---

## Requirements

numpy
pandas
matplotlib
seaborn
yfinance
scikit-learn
tensorflow>=2.12

---

## Key Design Decisions

- StandardScaler is fitted only on training data to avoid data leakage
- Sequence overlap is maintained across splits to avoid cold-start issues
- The best model is saved as best_lstm_stock_model.keras using ModelCheckpoint

---

## Author

Soham Rathi
sohamrathi.com
