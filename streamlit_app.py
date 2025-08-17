import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import vriddhi_core
from vriddhi_core import run_vriddhi_backend, plot_enhanced_projection

def display_investment_summary(summary_data):
    """Display the detailed investment summary in Streamlit UI"""
    
    # Main header
    st.markdown("---")
    st.markdown("## 🎯 Investment Analysis Report")
    
    if summary_data["feasible"]:
        st.success("🎉 SUCCESS: Your investment goals are ACHIEVABLE! 🎉")
        
        # Plan Summary Section
        st.markdown("### 📋 Plan Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Investment Period", f"{summary_data['horizon_years']:.1f} years")
            st.metric("Total Investment", f"₹{int(summary_data['total_invested']):,}")
            st.metric("Money Multiplier", f"{summary_data['money_multiplier']:.2f}x")
        
        with col2:
            st.metric("Final Portfolio Value", f"₹{int(summary_data['projected_value']):,}")
            st.metric("Total Gains", f"₹{int(summary_data['gain']):,}")
            st.metric("Monthly Avg Gain", f"₹{int(summary_data['monthly_avg_gain']):,}")
        
        # Success Insights
        st.markdown("### ✨ What This Means For You")
        st.info(f"""
        - Your disciplined investment will grow your wealth by **₹{int(summary_data['gain']):,}**
        - Every ₹1 you invest will become **₹{summary_data['money_multiplier']:.2f}**
        - Your wealth will grow **{summary_data['total_return_pct']:.1f}%** over {summary_data['horizon_years']:.1f} years
        - You're on the path to financial growth! 📈
        """)
        
    else:
        st.warning("⚠️ REALITY CHECK: Your expectations need adjustment")
        
        # Current Scenario
        st.markdown("### 📋 Current Scenario")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Desired CAGR", f"{summary_data['expected_cagr']:.2f}%")
        with col2:
            st.metric("Achievable CAGR", f"{summary_data['achieved_cagr']:.2f}%")
        with col3:
            st.metric("CAGR Gap", f"{summary_data['cagr_gap']:.2f}%", delta=f"{summary_data['cagr_gap']:.2f}%")
        
        # Good News Section
        st.markdown("### 💰 But Here's The Good News")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("You'll Still Gain", f"₹{int(summary_data['gain']):,}")
            st.metric("Total Return", f"{summary_data['total_return_pct']:.1f}%")
        
        with col2:
            st.metric("Final Value", f"₹{int(summary_data['projected_value']):,}")
            st.metric("Monthly Avg Gain", f"₹{int(summary_data['monthly_avg_gain']):,}")
        
        # Smart Recommendations
        st.markdown("### 💡 Smart Recommendations")
        st.info(f"""
        **Option 1:** Accept {summary_data['achieved_cagr']:.2f}% CAGR → Gain ₹{int(summary_data['gain']):,}
        
        **Option 2:** Extend to 60 months for up to {summary_data['best_horizon_60_cagr']:.2f}% CAGR
        
        **Option 3:** Increase monthly investment to reach your target faster
        
        **Option 4:** Adjust expectations - {summary_data['achieved_cagr']:.2f}% is still excellent!
        """)
        
        # Perspective Check
        st.markdown("### 🧠 Perspective Check")
        st.success(f"""
        - **Bank FD gives ~7%** → You're getting **{summary_data['achieved_cagr']:.1f}%**!
        - **Inflation is ~6%** → You're beating it by **{summary_data['inflation_beat']:.1f}%**!
        - This is solid wealth creation, even if not your original target!
        """)
    
    st.markdown("---")

st.set_page_config(page_title="Vriddhi Alpha Finder", layout="wide")

# ---- Optional simple password gate (set APP_PASSWORD in Streamlit secrets) ----
required_pw = st.secrets.get("APP_PASSWORD", None)
if required_pw:
    pw = st.sidebar.text_input("App password", type="password")
    if pw != required_pw:
        st.warning("Enter the app password to continue.")
        st.stop()

# ---- Title ----
st.title("🌟 Vriddhi Alpha Finder")
st.markdown("""
### AI-Powered Personal Investment Advisor

**Vriddhi Alpha Finder** is a sophisticated investment optimization platform that leverages Modern Portfolio Theory (MPT) and advanced analytics to create personalized investment strategies. 

**Key Features:**
- 📊 **Smart Portfolio Optimization**: Uses scientific algorithms to maximize returns while managing risk
- 🎯 **Goal-Based Planning**: Input your target returns and investment horizon for customized recommendations  
- 🏢 **Sector Diversification**: Automatically ensures balanced exposure across different industry sectors
- 📈 **Growth Projections**: Visualizes your wealth accumulation journey with detailed charts and metrics
- 💰 **SIP Modeling**: Optimized for systematic monthly investment plans (SIP)
- 🔍 **50+ Stock Universe**: Curated selection of high-quality Indian stocks with multi-horizon CAGR forecasts

**How It Works:**
1. Set your monthly investment amount and target annual returns (CAGR)
2. Choose your investment horizon (1-5 years)
3. Get AI-powered stock selection and optimal portfolio weights
4. View comprehensive analysis including feasibility assessment and growth projections

*Built with cutting-edge financial algorithms and real-time market data analysis.*
""")

# Load built-in stock data
@st.cache_data
def load_stock_data():
    return pd.read_csv("grand_table.csv")

try:
    df = load_stock_data()
    st.success(f"✅ Loaded {len(df)} stocks from curated universe")
except Exception as e:
    st.error(f"Error loading stock data: {e}")
    st.stop()

# Basic sanity check
if "Ticker" not in df.columns:
    st.error("Stock data must contain a 'Ticker' column. Found columns: {}".format(", ".join(df.columns)))
    st.stop()

# ---- Parameters ----
st.sidebar.header("📊 Investment Parameters")
monthly_investment = st.sidebar.number_input("Monthly Investment (INR)", min_value=1000, step=1000, value=25000, help="Amount you plan to invest every month")

# Discrete horizon selection
horizon_years = st.sidebar.selectbox(
    "Investment Horizon", 
    options=[1, 2, 3, 4, 5],
    index=4,  # Default to 5 years
    help="Choose your investment time horizon"
)
horizon_months = horizon_years * 12

expected_cagr_pct = st.sidebar.slider("Target CAGR (%)", min_value=5, max_value=99, value=18, help="Your expected annual returns")
expected_cagr = expected_cagr_pct / 100.0

# Display investment summary
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Investment Summary")
st.sidebar.info(f"""
**Monthly Investment:** ₹{monthly_investment:,}  
**Investment Period:** {horizon_years} years ({horizon_months} months)  
**Target CAGR:** {expected_cagr_pct}%  
**Total Investment:** ₹{monthly_investment * horizon_months:,}
""")

run = st.sidebar.button("🚀 Generate Investment Plan", type="primary", use_container_width=True)

if run:
    with st.spinner("🚀 Running Vriddhi optimization..."):
        try:
            allocation_df, fig, summary, summary_data = run_vriddhi_backend(
                df, monthly_investment, expected_cagr, horizon_months
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    # ---- Results ----
    st.subheader("Summary")

    c1, c2, c3, c4 = st.columns(4)
    feasible = summary.get("Feasible")
    expected_cagr_display = summary.get("Expected CAGR", expected_cagr_pct)
    achieved_cagr_display = summary.get("Achieved CAGR", 0)  # Already converted to percentage in backend
    
    c1.metric("Feasible", "Yes ✅" if feasible else "No ❌")
    c2.metric("Expected CAGR", f"{expected_cagr_display:.1f}%")
    c3.metric("Achieved CAGR", f"{achieved_cagr_display:.1f}%")
    c4.metric("Final Value", f"₹{summary.get('Final Value', 0):,}")
    
    # Show gain in a separate row
    st.metric("Total Gain", f"₹{summary.get('Gain', 0):,}")

    if feasible:
        st.success("🎉 Your investment goals are achievable at the selected horizon and parameters!")
    else:
        max_achievable_cagr = achieved_cagr_display
        target_cagr = expected_cagr_display
        st.warning(f"⚠️ Target {target_cagr:.1f}% CAGR is not feasible. Best achievable: {max_achievable_cagr:.1f}% CAGR")
        st.info(f"💡 Consider: Lower your target to {max_achievable_cagr:.1f}% or extend your investment horizon for better returns.")

    # Display detailed investment summary
    display_investment_summary(summary_data)
    
    st.subheader("Suggested Allocation")
    st.dataframe(allocation_df, use_container_width=True)

    # Download allocations
    csv_bytes = allocation_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Allocation CSV", data=csv_bytes, file_name="vriddhi_allocation.csv", mime="text/csv")


    # ---- Projection chart from your function ----
    st.subheader("Growth Projection")
    try:
        if fig is not None:
            st.pyplot(fig, use_container_width=True)
            # Download chart
            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.download_button("Download Projection PNG", data=buf.getvalue(), file_name="projection.png", mime="image/png")
        else:
            st.warning("Projection chart could not be generated.")
    except Exception as e:
        st.error(f"Error displaying projection chart: {str(e)}")

else:
    # Welcome message when no optimization has been run
    st.markdown("---")
    st.markdown("### 🚀 Ready to Start Your Investment Journey?")
    st.info("""
    **Getting Started:**
    1. 💰 Set your monthly investment amount in the sidebar
    2. 📅 Choose your investment horizon (1-5 years)  
    3. 🎯 Set your target CAGR percentage
    4. 🚀 Click "Generate Investment Plan" to see your optimized portfolio
    
    The AI will analyze 50+ stocks and create a personalized investment strategy just for you!
    """)
    
    # Display sample of available stocks
    st.markdown("### 📊 Available Stock Universe")
    st.markdown("Here's a preview of the curated stocks available for optimization:")
    
    # Show top 10 stocks by average CAGR
    if 'average_cagr' in df.columns:
        top_stocks = df.nlargest(10, 'average_cagr')[['Ticker', 'Price', 'PE_Ratio', 'average_cagr']]
        top_stocks.columns = ['Stock', 'Price (₹)', 'P/E Ratio', 'Avg CAGR (%)']
        st.dataframe(top_stocks, use_container_width=True)
    else:
        # Fallback to first 10 stocks
        preview_stocks = df.head(10)[['Ticker', 'Price', 'PE_Ratio']]
        preview_stocks.columns = ['Stock', 'Price (₹)', 'P/E Ratio']
        st.dataframe(preview_stocks, use_container_width=True)
