# 📈 Factor Edge Dashboard

A professional-grade financial analytics dashboard built with **Python** and **Streamlit**. This tool provides quantitative insights into factor momentum, relative performance, and statistical mean reversion.

## 🚀 Key Features
* **Performance Monitoring**: Tracks 1D, 5D, and 20D returns across custom factor clusters (AI Tech, SaaS, Energy, etc.).
* **Statistical Pair Analysis**: Utilizes rolling OLS regression to identify "Rich" vs "Cheap" assets using residual analysis and +2/-1 sigma bands.
* **Factor Rotation**: Quadrant visualization to identify market leaders, laggards, and recovering sectors.
* **Momentum Backtester**: A rule-based engine that simulates a rotation strategy and provides **Live Signals** for the next trading session.
* **Macro Overlay**: Correlates equity performance with market volatility (VIX).

## 🛠️ Local Setup
1. Clone this repository or download the files.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. streamlit run app.py

📊 Data Source
Market data is powered by the yfinance API, pulling real-time and historical closing prices from Yahoo Finance.

🧪 Statistical Logic
The Pair Analysis tab uses a rolling hedge ratio (Beta) to remove market beta and isolate the idiosyncratic spread (Residual) between two assets, allowing for precise relative-value trading.
