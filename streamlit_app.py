import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import vriddhi_core
from vriddhi_core import run_vriddhi_backend, plot_enhanced_projection

# Educational Disclaimer
def show_disclaimer():
    st.error("""
    ⚠️ **IMPORTANT EDUCATIONAL DISCLAIMER** ⚠️
    
    This application is designed for **EDUCATIONAL PURPOSES ONLY** and is currently in **BETA TESTING**.
    
    **DO NOT** use these recommendations for actual investment decisions. This tool:
    - Uses simulated data and theoretical models
    - Is not reviewed by financial professionals
    - Does not constitute financial advice
    - Should not replace consultation with qualified financial advisors
    
    **For educational learning about portfolio theory and investment concepts only.**
    """)
    st.markdown("---")

def display_stock_selection_rationale(rationale):
    """Display the stock selection rationale"""
    st.markdown("### 🧠 Stock Selection Rationale")
    
    with st.expander("📋 How were these stocks selected?", expanded=False):
        st.write("**📊 Stock Selection Rationale:**")
        st.write(f"- **Selection Method**: Enhanced sector-based diversification")
        st.write(f"- **Universe Size**: {rationale.get('total_universe', 'N/A')} stocks analyzed")
        st.write(f"- **Sectors Available**: {rationale.get('sectors_available', 'N/A')} sectors")
        st.write(f"- **Final Selection**: {rationale.get('stocks_selected', 'N/A')} stocks (minimum 8, maximum unlimited)")
        
        # Display updated selection criteria
        st.write("**🎯 PEG-Based Selection Approach:**")
        st.write("  • **Round 1**: Best stock from each sector with lowest PEG ratio")
        st.write("  • **Round 2**: All remaining stocks with PEG < 1.0")
        st.write("**📈 Selection Criteria:**")
        st.write("  • **PEG Ratio**: PE Ratio ÷ Average Historical CAGR")
        st.write("  • **Lower PEG = Better Value**: Growth at reasonable price")
        st.write("  • **PEG < 1.0**: Premium quality threshold for additional stocks")
        st.write("  • **No Negative CAGR**: Only positive performance stocks selected")
        
        # Display sector breakdown if available
        sector_breakdown = rationale.get('sector_breakdown', {})
        if sector_breakdown:
            st.write("**🏭 Sector-wise Selection:**")
            for sector, details in sector_breakdown.items():
                # Handle both single stock entries and additional stock lists
                if isinstance(details, dict):
                    # Primary sector selection (single stock)
                    st.write(f"  • **{sector}**: {details.get('selected_stock', 'N/A')} "
                            f"(CAGR: {details.get('avg_cagr', 0):.1f}%, "
                            f"PE: {details.get('pe_ratio', 0):.1f}, "
                            f"PEG: {details.get('peg_ratio', 0):.2f})")
                elif isinstance(details, list):
                    # Additional sector selections (multiple stocks)
                    st.write(f"  • **{sector}** (Additional):")
                    for stock in details:
                        st.write(f"    - {stock.get('selected_stock', 'N/A')} "
                                f"(CAGR: {stock.get('avg_cagr', 0):.1f}%, "
                                f"PE: {stock.get('pe_ratio', 0):.1f}, "
                                f"PEG: {stock.get('peg_ratio', 0):.2f})")
        
        # Remove feasibility messaging - app now focuses on best possible recommendations
        
        st.markdown(f"""
        **Portfolio Summary:**
        - **Expected CAGR:** {rationale.get('achieved_cagr', 'N/A')}
        - **Diversification:** Enhanced diversification (minimum 1 per sector + additional quality stocks)
        - **Total Stocks Selected:** {rationale.get('stocks_selected', 'N/A')} stocks
        """)
        
        st.info("""
        **Why This Enhanced Approach?**
        Our two-phase selection maximizes both diversification and CAGR potential:
        - **Phase 1** ensures sector diversification with the best stock from each sector
        - **Phase 2** adds high-quality stocks meeting strict criteria (CAGR ≥10%, PB ≤5.0, PE 5-50)
        This approach balances risk through diversification while maximizing return potential 
        by including all qualifying high-performance stocks.
        """)

def display_investment_summary(summary_data, actual_feasible):
    """Display the detailed investment summary in Streamlit UI"""
    
    # Main header
    st.markdown("---")
    st.markdown("## 📊 Investment Plan Summary")
    
    # Plan Summary Section
    st.markdown("### 📋 Your Investment Journey")
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

st.set_page_config(page_title="Vriddhi Alpha Finder", layout="wide")

# ---- Optional simple password gate (set APP_PASSWORD in Streamlit secrets) ----
required_pw = st.secrets.get("APP_PASSWORD", None)
if required_pw:
    pw = st.sidebar.text_input("App password", type="password")
    if pw != required_pw:
        st.warning("Enter the app password to continue.")
        st.stop()

# Main title and description
st.title("🌟 Vriddhi Alpha Finder")
st.markdown("### Professional AI Investment Advisor")

# Show disclaimer prominently
show_disclaimer()
st.markdown("""
### AI-Powered Personal Investment Advisor

**Vriddhi Alpha Finder** is a sophisticated investment optimization platform that leverages Modern Portfolio Theory (MPT) and advanced analytics to create personalized investment strategies. 

**Key Features:**
- 🔬 **ML-Powered Forecasting**: Advanced Prophet + LSTM + XGBoost ensemble predictions with 20-year lookback
- 🎯 **PEG-Based Selection**: Intelligent growth-at-reasonable-price algorithm (Round 1: sector diversification, Round 2: PEG < 1.0)
- 📊 **Modern Portfolio Theory**: Professional MPT optimization with risk-adjusted returns and sector constraints
- 🏢 **Automatic Diversification**: Balanced exposure across industries with quality filtering
- 📈 **Multi-Horizon Analysis**: Comprehensive 12M/24M/36M/48M/60M CAGR forecasts and growth projections
- 💰 **SIP Optimization**: Systematic monthly investment modeling with both fractional and whole-share allocations

**How It Works:**
1. Set your monthly investment amount (minimum ₹50,000)
2. Choose your investment horizon (1-5 years)
3. Get AI-powered stock selection with optimal portfolio weights
4. View comprehensive analysis and growth projections for maximum CAGR

*Built with cutting-edge financial algorithms and real-time market data analysis.*
""")

# Load built-in stock data
@st.cache_data
def load_stock_data():
    return pd.read_csv("grand_table_expanded.csv")

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
monthly_investment = st.sidebar.number_input("Monthly Investment (INR)", min_value=50000, step=5000, value=50000, help="Amount you plan to invest every month (minimum ₹50,000)")

# Discrete horizon selection
horizon_years = st.sidebar.selectbox(
    "Investment Horizon", 
    options=[1, 2, 3, 4, 5],
    index=4,  # Default to 5 years
    help="Choose your investment time horizon"
)
horizon_months = horizon_years * 12

# Investment Summary in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Investment Summary")
st.sidebar.info(f"""
**Monthly Investment:** ₹{monthly_investment:,}  
**Investment Horizon:** {horizon_years} years  
**Total Investment:** ₹{monthly_investment * horizon_months:,}
""")

# ---- Run Optimization ----
if st.button("🚀 Generate Investment Plan", type="primary"):
    with st.spinner("🔍 Analyzing market data and optimizing your portfolio..."):
        # Run the backend analysis with default expected CAGR (algorithm will find best possible)
        expected_cagr = 0.15  # Default 15% - algorithm will optimize for best possible CAGR
        portfolio_df, fig, frill_output, summary_data, selection_rationale, whole_share_df = run_vriddhi_backend(
            monthly_investment, horizon_months, expected_cagr
        )

    # Quick Summary Metrics
    c1, c2, c3 = st.columns(3)
    achieved_cagr_display = frill_output.get("Achieved CAGR", 0)
    
    c1.metric("Expected CAGR", f"{achieved_cagr_display:.1f}%")
    c2.metric("Final Value", f"₹{frill_output.get('Final Value', 0):,}")
    c3.metric("Total Stocks", f"{len(portfolio_df)} stocks")

    # Display stock selection rationale
    display_stock_selection_rationale(selection_rationale)
    
    # Display detailed investment summary
    display_investment_summary(summary_data, True)
    
    # Display portfolio allocation - Side by side comparison
    st.markdown("### 📊 Portfolio Allocation Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💰 Fractional Share Plan")
        st.markdown(f"**Monthly Investment:** ₹{monthly_investment:,}")
        st.dataframe(portfolio_df, use_container_width=True)
    
    with col2:
        st.markdown("#### 🔢 Whole Share Plan")
        total_investment = whole_share_df['Total_Monthly_Investment'].iloc[0] if len(whole_share_df) > 0 else 0
        st.markdown(f"**Monthly Investment Required:** ₹{total_investment:,.0f}")
        
        # Display whole share allocation with better formatting
        display_df = whole_share_df[['Ticker', 'Current_Price', 'Whole_Shares', 'Share_Cost', 'Actual_Weight']].copy()
        display_df['Current_Price'] = display_df['Current_Price'].apply(lambda x: f"₹{x:,.0f}")
        display_df['Share_Cost'] = display_df['Share_Cost'].apply(lambda x: f"₹{x:,.0f}")
        display_df['Actual_Weight'] = display_df['Actual_Weight'].apply(lambda x: f"{x:.1%}")
        display_df.columns = ['Stock', 'Price/Share', 'Qty', 'Total Cost', 'Weight']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Show investment difference
        difference = total_investment - monthly_investment
        if difference > 0:
            st.info(f"💡 **Additional ₹{difference:,.0f}/month** needed for whole shares")
        else:
            st.success(f"💡 **Save ₹{abs(difference):,.0f}/month** with whole shares")
    
    # Display the comprehensive chart (single instance)
    st.markdown("### 📈 Investment Growth Analysis")
    try:
        if fig:
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            st.download_button("Download Projection PNG", data=buf.getvalue(), file_name="projection.png", mime="image/png")
        else:
            st.warning("Projection chart could not be generated.")
    except Exception as e:
        st.error(f"Error displaying projection chart: {str(e)}")
    
    st.success("✅ Analysis complete! Review your personalized investment strategy above.")

else:
    # Welcome message when no optimization has been run
    st.markdown("---")
    st.markdown("### 🚀 Ready to Start Your Investment Journey?")
    st.info("""
    **Getting Started:**
    1. 💰 Set your monthly investment amount in the sidebar
    2. 📅 Choose your investment horizon (1-5 years)  
    3. 🚀 Click "Generate Investment Plan" to see your optimized portfolio
    
    The AI will analyze 50+ curated stocks and create a personalized investment strategy just for you!
    """)
