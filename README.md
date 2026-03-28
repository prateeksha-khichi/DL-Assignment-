# DL-Assignment-
The project covers the full machine learning pipeline — from data collection and preprocessing to model training, evaluation, and interpretation — aligned with the five tasks outlined in the assignment.


# 📈 Deep Learning for Stock Price Prediction
### LSTM & Hybrid LSTM + NLP Sentiment Analysis

> **Final Assignment — Deep Learning Subject**  
> Two deep learning models applied to real-world financial time-series data

---

## 🧠 Project Overview

This project implements and evaluates **two deep learning models** for stock market price prediction:

| Model | Architecture | Dataset | Target |
|-------|-------------|---------|--------|
| **Model 1** | Stacked LSTM (4 layers) | GOOGL – Yahoo Finance (2009–2023) | Next-day Close Price |
| **Model 2** | Hybrid LSTM + NLP Sentiment | BSE SENSEX + Times of India Headlines (2001–2020) | Next-day Close Price |

The models demonstrate how sequential deep learning architectures can capture temporal dependencies in financial data — and how fusing structured price data with unstructured news sentiment can improve prediction quality.

---

## 📁 Repository Structure

```
DL-Stock-Prediction/
│
├── DL_Assignment_Complete.ipynb   # Main Colab notebook (all 5 tasks)
├── README.md                      # Project documentation
├── PROJECT_DESCRIPTION.md         # Detailed project description
│
├── outputs/                       # Generated plots & saved models
│   ├── training_validation_loss.png
│   ├── model1_prediction.png
│   ├── model2_prediction.png
│   ├── residuals.png
│   ├── model1_lstm_googl.h5
│   ├── model2_hybrid_lstm_nlp.json
│   └── model2_hybrid_lstm_nlp.h5
│
└── data/
    └── india-news-headlines.csv   # Download separately (see below)
```

---

## 📊 Datasets

### Model 1 — GOOGL Stock Data
- **Source:** Yahoo Finance via `yfinance` Python API
- **Ticker:** `GOOGL` (Alphabet Inc.)
- **Period:** January 2009 – January 2023
- **Samples:** ~3,500 trading days
- **Features:** Open, High, Low, Close, Volume
- **Access:** Loaded automatically in notebook via `yfinance`

### Model 2 — BSE SENSEX + News Headlines
- **Stock Source:** Yahoo Finance — BSE SENSEX (`^BSESN`), 2001–2020
- **News Source:** [Times of India News Headlines — Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DPQMQH)
- **News Size:** ~3.4 million headlines (2001–2020)
- **Combined Samples:** ~4,900 records (after inner join on date)
- **Features:** Close, Open, High, Low, Volume + VADER Sentiment Scores (compound, positive, negative, neutral)

> ⚠️ **Note:** Download `india-news-headlines.csv` from the Harvard Dataverse link above and upload it to your Colab session before running Model 2 cells.

---

## 🏗️ Model Architectures

### Model 1: Stacked LSTM

```
Input (100 time steps × 1 feature)
    ↓
LSTM(50, relu, return_sequences=True) → Dropout(0.2)
    ↓
LSTM(60, relu, return_sequences=True) → Dropout(0.3)
    ↓
LSTM(80, relu, return_sequences=True) → Dropout(0.4)
    ↓
LSTM(120, relu)                       → Dropout(0.5)
    ↓
Dense(1)   ← Output: predicted closing price
```

- **Optimizer:** Adam | **Loss:** MSE | **Epochs:** 50 | **Batch Size:** 32

### Model 2: Hybrid LSTM + Sentiment Analysis

```
Input (7 features × 1 time step)
[Close, Compound, Compound_shifted, Volume, Open, High, Low]
    ↓
LSTM(100, tanh, return_sequences=True) → Dropout(0.1)
    ↓
LSTM(100, tanh, return_sequences=True) → Dropout(0.1)
    ↓
LSTM(100, tanh)                        → Dropout(0.1)
    ↓
Dense(1)   ← Output: next-day closing price
```

- **Optimizer:** Adam | **Loss:** MSE | **Epochs:** 10 | **Batch Size:** 8 | **Val Split:** 20%

---

## 📋 Assignment Tasks Covered

| Task | Description | Marks |
|------|-------------|-------|
| **Task 1** | Dataset identification, justification, problem type | 4 |
| **Task 2** | Data cleaning, normalization, visualization, train/test split | 3 |
| **Task 3** | Two model architectures with layer details, hyperparameters, justification | 10 |
| **Task 4** | RMSE, MAE, R², MAPE, Directional Accuracy, Loss & Prediction plots | 10 |
| **Task 5** | Conclusion, future improvements, GitHub link | 8 |
| **Total** | | **35 + 5** |

---

## 📈 Evaluation Metrics

Since both tasks are **regression** (continuous price prediction), the following metrics are used:

| Metric | Description |
|--------|-------------|
| **RMSE** | Root Mean Square Error — penalizes large errors |
| **MAE** | Mean Absolute Error — average prediction gap |
| **R² Score** | Coefficient of determination — goodness of fit |
| **MAPE** | Mean Absolute Percentage Error — % deviation |
| **Directional Accuracy** | % of correct up/down direction predictions (proxy for classification) |

> Note: ROC/AUC curves are classification metrics and are not applicable to price regression tasks. Directional Accuracy is used as a practical alternative.

---

## 🚀 Getting Started

### Run on Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `DL_Assignment_Complete.ipynb`
3. For Model 2: also upload `india-news-headlines.csv`
4. Run all cells sequentially (`Runtime → Run All`)

### Dependencies

All installed automatically in the notebook. Manual install:

```bash
pip install yfinance pandas_datareader scikit-learn tensorflow keras matplotlib seaborn nltk
```

---

## 🔮 Future Work

1. **Attention Mechanisms** — Focus on the most relevant time steps instead of treating all equally
2. **Temporal Fusion Transformer (TFT)** — State-of-the-art architecture built for time-series
3. **FinBERT** — BERT pre-trained on financial text for superior sentiment extraction vs. VADER
4. **BiLSTM** — Bidirectional LSTM to capture both past and future context
5. **Technical Indicators** — Add RSI, MACD, Bollinger Bands as additional features
6. **Data Augmentation** — Expand training data using synthetic sequences
7. **Ensemble Models** — Stack LSTM + XGBoost + ARIMA predictions

---

## 📚 References

- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
- Hutto, C., & Gilbert, E. (2014). VADER: A Parsimonious Rule-Based Model for Sentiment Analysis. *ICWSM*.
- Times of India News Headlines Dataset — Harvard Dataverse: [doi:10.7910/DVN/DPQMQH](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DPQMQH)
- Yahoo Finance API via `yfinance`: [https://pypi.org/project/yfinance/](https://pypi.org/project/yfinance/)

---

## 👤 Author

> **Name:** Prateeksha Khichi 
> **Subject:** Deep Learning  
> **Submission:** Final Assignment  


