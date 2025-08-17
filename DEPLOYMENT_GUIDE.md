# Vriddhi Alpha Finder - Deployment Guide

## Option 1: Streamlit Cloud (Recommended - FREE)

### Prerequisites
- GitLab repository with vriddhi-core files
- Streamlit Cloud account (free)

### Step-by-Step Instructions

1. **Push files to GitLab**
   ```bash
   cd vriddhi-core/
   git add .
   git commit -m "Add deployment files for Streamlit Cloud"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Connect your GitLab account
   - Select your repository
   - Set **Main file path**: `streamlit_app.py`
   - Set **Python version**: `3.9`
   - Click "Deploy"

3. **Configure Secrets (Optional)**
   - In your app dashboard, click "Settings" → "Secrets"
   - Add secrets in TOML format:
   ```toml
   APP_PASSWORD = "your_secret_password"
   API_KEY = "your_api_key_if_needed"
   ```

### App will be live at: `https://your-app-name.streamlit.app`

---

## Option 2: Render (Fallback - FREE tier)

### If Streamlit Cloud fails, use Render:

1. **Create Render account** at [render.com](https://render.com)

2. **Create Web Service**
   - Connect GitLab repository
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0`
   - **Environment**: Python 3

3. **Configure Environment Variables**
   - Add `APP_PASSWORD` if needed
   - Add any other secrets

### Free tier limitations:
- App sleeps after 15 minutes of inactivity
- 750 hours/month limit

---

## Option 3: Railway (Alternative)

1. **Create Railway account** at [railway.app](https://railway.app)
2. **Deploy from GitLab**
3. **Auto-detects** Python and uses Procfile
4. **$5/month** after free trial

---

## Troubleshooting

### Common Issues:
- **Port binding**: Ensure Procfile uses `$PORT` variable
- **Dependencies**: Check requirements.txt versions
- **File paths**: Ensure `grand_table.csv` is in same directory
- **Memory limits**: Free tiers have 512MB-1GB limits

### Performance Tips:
- Use `@st.cache_data` for data loading (already implemented)
- Optimize matplotlib figure sizes if memory issues occur
- Consider data compression for larger datasets
