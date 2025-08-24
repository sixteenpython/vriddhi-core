# 🌟 Vriddhi Alpha Finder — AI Investment Advisor

**Professional-grade investment optimization platform** combining cutting-edge machine learning forecasting with Modern Portfolio Theory for personalized portfolio construction.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501

## Key Features
- 🔬 **ML-Powered Forecasting**: Prophet + LSTM + XGBoost ensemble predictions
- 🎯 **PEG-Based Selection**: Growth at reasonable prices (Round 1: sector diversification, Round 2: PEG < 1.0)
- 📊 **Modern Portfolio Theory**: Professional MPT optimization with risk-adjusted returns
- 🏢 **Sector Diversification**: Automatic balanced exposure across industries
- 📈 **Comprehensive Analysis**: Multi-horizon CAGR forecasts (12M-60M) with detailed visualizations

## Files
- `streamlit_app.py` — Interactive Streamlit UI with educational features
- `vriddhi_core.py` — Core optimization engine with PEG-based selection
- `grand_table_expanded.csv` — ML-generated forecasts for 50+ Indian stocks

## Deploy on Streamlit Cloud (fastest)
1. Push these files to a **GitHub** repository (Streamlit Cloud integrates best with GitHub).
2. Go to https://share.streamlit.io/ → New app → Connect GitHub → select the repo → set **Main file path** to `streamlit_app.py` → Deploy.
3. In app settings, restrict access (invite-only) and add your friends’ emails.

