# 📈 Stock Price Prediction with LSTM

Predicts the **next-day closing price** of any stock ticker using a stacked LSTM neural network, trained on historical OHLCV data and engineered time-series features.

---

## 📌 Problem Statement

Given the last 60 days of stock market data, can we predict tomorrow's closing price — and beat a naive *"today's price = tomorrow's price"* baseline?

---

## 🗂️ Project Structure

```
stock-price-prediction/
├── stock_price_prediction_code.ipynb   # Full notebook: EDA → features → training → evaluation
├── requirements.txt                    # Python dependencies
└── README.md
```

---

## 🔄 Workflow Overview

```
Yahoo Finance (yfinance)
        ↓
  Raw OHLCV Data (AAPL, 2020–today)
        ↓
  Feature Engineering
  (Returns, MA-7, MA-21, Volatility-7)
        ↓
  Train / Validation / Test Split (70/15/15)
        ↓
  StandardScaler (fit on train only)
        ↓
  60-day Lookback Sequences
        ↓
  Stacked LSTM Model (TensorFlow/Keras)
        ↓
  Evaluation vs. Naive Baseline
```

---

## ⚙️ Features Engineered

| Feature | Description |
|---|---|
| `Close` | Raw closing price |
| `Volume` | Trading volume |
| `Return_1d` | Daily percentage return |
| `MA_7` | 7-day moving average |
| `MA_21` | 21-day moving average |
| `Volatility_7` | 7-day rolling return standard deviation |

---

## 🧠 Model Architecture

```
Input (60 timesteps × 6 features)
  → LSTM(64, return_sequences=True)
  → Dropout(0.2)
  → LSTM(32)
  → Dropout(0.2)
  → Dense(64, relu)
  → Dropout(0.2)
  → Dense(1)           ← predicted next-day close price
```

- **Optimizer:** Adam
- **Loss:** Mean Squared Error (MSE)
- **Callbacks:** EarlyStopping (patience=10), ModelCheckpoint

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| RMSE | Root Mean Squared Error |
| MAE | Mean Absolute Error |
| MAPE | Mean Absolute Percentage Error |
| Directional Accuracy | % of correctly predicted price movement direction |

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/stock-price-prediction.git
cd stock-price-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the notebook**
```bash
jupyter notebook stock_price_prediction_code.ipynb
```

**4. (Optional) Change the ticker**

In Cell 2, update:
```python
ticker = "TSLA"   # or "GOOG", "MSFT", etc.
```

---

## 📦 Requirements

```
numpy
pandas
matplotlib
seaborn
yfinance
scikit-learn
tensorflow>=2.12
```

---

## 📝 Key Design Decisions

- **No data leakage:** `StandardScaler` is fit only on training data, then applied to val/test.
- **Sequence overlap:** Validation and test sequences include the tail of the previous split to avoid cold-start gaps.
- **Best model saved** as `best_lstm_stock_model.keras` via `ModelCheckpoint`.

---

## 👤 Author

> Soham Rathi
sohamrathi.com
