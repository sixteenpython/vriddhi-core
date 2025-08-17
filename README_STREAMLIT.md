# Vriddhi Alpha Finder — Streamlit MVP

This repository contains a minimal Streamlit app that reads your knowledge asset CSV and runs the Vriddhi optimizer derived from your notebook.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open http://localhost:8501

## Files
- `streamlit_app.py` — Streamlit UI
- `vriddhi_core.py` — code auto-extracted from your notebook
- `grand_table.csv` — bundled knowledge asset

## Deploy on Streamlit Cloud (fastest)
1. Push these files to a **GitHub** repository (Streamlit Cloud integrates best with GitHub).
2. Go to https://share.streamlit.io/ → New app → Connect GitHub → select the repo → set **Main file path** to `streamlit_app.py` → Deploy.
3. In app settings, restrict access (invite-only) and add your friends’ emails.

