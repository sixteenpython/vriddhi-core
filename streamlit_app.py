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
        **Universe Filtering:**
        - Started with **{rationale['total_universe']} stocks** from our curated database
        - Applied quality filters, resulting in **{rationale['after_quality_filters']} eligible stocks**
        
        **Quality Filters Applied:**
        """)
        for filter_desc in rationale['filters_applied']:
            st.markdown(f"- {filter_desc}")
        
        st.markdown(f"""
        **Selection Method:** {rationale['selection_method']}
        """)
        
        if 'scoring_factors' in rationale:
            st.markdown(f"**Scoring System:** {rationale['scoring_factors']}")
            st.markdown("- **Growth (30%):** Horizon-specific forecast performance")
            st.markdown("- **Value (25%):** Risk-adjusted returns from expanded metrics")
            st.markdown("- **Quality (20%):** Momentum score × Trend direction")
            st.markdown("- **Risk (15%):** Historical volatility × Risk level")
            st.markdown("- **Style (10%):** Investment style diversification")
        else:
            st.markdown("- Stocks ranked by PEG-adjusted returns (growth potential vs valuation)")
            st.markdown("- Selected greedily to maximize portfolio CAGR")
        
        if 'diversification_rules' in rationale:
            st.markdown(f"**Diversification Rules:** {rationale['diversification_rules']}")
        else:
            st.markdown(f"**Diversification Rule:** {rationale.get('diversification', 'Max 2 stocks per sector')}")
        
        if rationale.get('fallback_used', False):
            st.warning(f"⚠️ **Fallback Selection Used:** {rationale['fallback_reason']}")
            if 'fallback_diversification' in rationale:
                st.markdown("**Fallback Portfolio Diversification:**")
                sectors = rationale['fallback_diversification']['sectors']
                styles = rationale['fallback_diversification']['styles']
                st.markdown(f"- Sectors: {dict(sectors)}")
                st.markdown(f"- Investment Styles: {dict(styles)}")
        else:
            st.success(f"✅ **Target Achieved:** Selected {rationale.get('stocks_selected', 'N/A')} stocks achieving {rationale.get('achieved_cagr', 'N/A')} CAGR")
            if 'portfolio_diversification' in rationale:
                st.markdown("**Portfolio Diversification Achieved:**")
                div = rationale['portfolio_diversification']
                st.markdown(f"- Sectors: {dict(div['sectors'])}")
                st.markdown(f"- Investment Styles: {dict(div['styles'])}")
                st.markdown(f"- Risk Levels: {dict(div['risk_levels'])}")
        
        st.info("""
        **Why This Enhanced Approach?**
        Our multi-factor composite scoring system balances growth potential, value metrics, 
        quality indicators, risk factors, and style diversification. This ensures optimal 
        stock selection while maintaining sophisticated portfolio diversification across 
        sectors, investment styles, and risk levels.
>>>>>>> 3a031820e97cca0b7d03077b9f46fe0012679329
        """)

def display_investment_summary(summary_data, actual_feasible):
    """Display the detailed investment summary in Streamlit UI"""
    
    # Main header
    st.markdown("---")
<<<<<<< HEAD
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
=======
    st.markdown("## 🎯 Investment Analysis Report")
    
    # Single source of truth comparison - Target vs Best Achievable
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Target CAGR", f"{summary_data['expected_cagr']:.1f}%")
    with col2:
        st.metric("Best Achievable CAGR", f"{summary_data['achieved_cagr']:.1f}%")
    with col3:
        gap = summary_data['cagr_gap']
        st.metric("CAGR Gap", f"{gap:.1f}%", delta=f"{gap:.1f}%" if gap != 0 else "Perfect Match")
    
    if actual_feasible:
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
            st.metric("Desired CAGR", f"{summary_data['expected_cagr']:.1f}%")
        with col2:
            st.metric("Achievable CAGR", f"{summary_data['achieved_cagr']:.1f}%")
        with col3:
            st.metric("CAGR Gap", f"{summary_data['cagr_gap']:.1f}%")
        
        # But Here's The Good News Section
        st.markdown("### 💰 But Here's The Good News")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("You'll Still Gain", f"₹{int(summary_data['gain']):,}")
            st.metric("Total Return", f"{summary_data['total_return_pct']:.1f}%")
        with col2:
            st.metric("Final Value", f"₹{int(summary_data['projected_value']):,}")
            st.metric("Monthly Avg Gain", f"₹{int(summary_data['monthly_avg_gain']):,}")
        
        # Recommendations
        st.markdown("### 💡 Recommendations")
        st.info(f"""
        **Option 1:** Lower your target CAGR to **{summary_data['achieved_cagr']:.1f}%** for this horizon
        
        **Option 2:** Extend your investment horizon for potentially higher returns
        
        **Current Reality:** Even at {summary_data['achieved_cagr']:.1f}% CAGR, you'll still earn ₹{int(summary_data['gain']):,} in gains!
        """)
>>>>>>> 3a031820e97cca0b7d03077b9f46fe0012679329

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
<<<<<<< HEAD
st.markdown("### Professional AI Investment Advisor | 9.5/10 Technical Excellence")
=======
st.markdown("### AI-Powered Personal Investment Advisor")
>>>>>>> 3a031820e97cca0b7d03077b9f46fe0012679329

# Show disclaimer prominently
show_disclaimer()
st.markdown("""
### AI-Powered Personal Investment Advisor

**Vriddhi Alpha Finder** is a sophisticated investment optimization platform that leverages Modern Portfolio Theory (MPT) and advanced analytics to create personalized investment strategies. 

**Key Features:**
<<<<<<< HEAD
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
=======
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
>>>>>>> 3a031820e97cca0b7d03077b9f46fe0012679329

*Built with cutting-edge financial algorithms and real-time market data analysis.*
""")

# Load built-in stock data
@st.cache_data
def load_stock_data():
<<<<<<< HEAD
    return pd.read_csv("grand_table_expanded.csv")
=======
    return pd.read_csv("grand_table.csv")
>>>>>>> 3a031820e97cca0b7d03077b9f46fe0012679329

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
<<<<<<< HEAD
monthly_investment = st.sidebar.number_input("Monthly Investment (INR)", min_value=50000, step=5000, value=50000, help="Amount you plan to invest every month (minimum ₹50,000)")
=======
monthly_investment = st.sidebar.number_input("Monthly Investment (INR)", min_value=1000, step=1000, value=25000, help="Amount you plan to invest every month")
>>>>>>> 3a031820e97cca0b7d03077b9f46fe0012679329

# Discrete horizon selection
horizon_years = st.sidebar.selectbox(
    "Investment Horizon", 
    options=[1, 2, 3, 4, 5],
    index=4,  # Default to 5 years
    help="Choose your investment time horizon"
)
horizon_months = horizon_years * 12

<<<<<<< HEAD
=======
expected_cagr_pct = st.sidebar.slider("Target CAGR (%)", min_value=8, max_value=99, value=35, step=1, help="Your expected annual returns")
expected_cagr = expected_cagr_pct / 100

>>>>>>> 3a031820e97cca0b7d03077b9f46fe0012679329
# Investment Summary in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Investment Summary")
st.sidebar.info(f"""
**Monthly Investment:** ₹{monthly_investment:,}  
**Investment Horizon:** {horizon_years} years  
<<<<<<< HEAD
=======
**Target CAGR:** {expected_cagr_pct}%  
>>>>>>> 3a031820e97cca0b7d03077b9f46fe0012679329
**Total Investment:** ₹{monthly_investment * horizon_months:,}
""")

# ---- Run Optimization ----
if st.button("🚀 Generate Investment Plan", type="primary"):
    with st.spinner("🔍 Analyzing market data and optimizing your portfolio..."):
<<<<<<< HEAD
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
=======
        try:
            portfolio_df, fig, frill_output, summary_data, selection_rationale, whole_share_df = run_vriddhi_backend(
                monthly_investment, horizon_months, expected_cagr
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    # ---- Results ----
    st.subheader("Summary")

    # Quick Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    feasible = frill_output.get("Feasible")
    expected_cagr_display = frill_output.get("Expected CAGR", expected_cagr_pct)
    achieved_cagr_display = frill_output.get("Achieved CAGR", 0)
    
    # Re-check feasibility based on actual CAGR comparison
    actual_feasible = achieved_cagr_display >= expected_cagr_display
    
    # Display feasibility with colored indicator
    if actual_feasible:
        c1.metric("Feasible", "✅ Yes")
    else:
        c1.metric("Feasible", "❌ No")
    
    c2.metric("Target CAGR", f"{expected_cagr_display:.1f}%")
    c3.metric("Best Achievable CAGR", f"{achieved_cagr_display:.1f}%")
    c4.metric("Final Value", f"₹{frill_output.get('Final Value', 0):,}")
>>>>>>> 3a031820e97cca0b7d03077b9f46fe0012679329

    # Display stock selection rationale
    display_stock_selection_rationale(selection_rationale)
    
    # Display detailed investment summary
<<<<<<< HEAD
    display_investment_summary(summary_data, True)
=======
    display_investment_summary(summary_data, actual_feasible)
>>>>>>> 3a031820e97cca0b7d03077b9f46fe0012679329
    
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
