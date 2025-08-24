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
        st.markdown(f"""
        **Simplified Sector-Based Selection:**
        - Started with **{rationale['total_universe']} stocks** from our curated database
        - Covered **{rationale['sectors_covered']} sectors** for maximum diversification
        - Selected **{rationale['stocks_selected']} stocks** (one best stock per sector)
        
        **Selection Method:** {rationale['selection_method']}
        
        **Selection Criteria:**
        """)
        for criteria in rationale['selection_criteria']:
            st.markdown(f"- {criteria}")
        
        st.markdown("### 📊 Sector-wise Selection Details")
        
        # Display sector selections in a nice format
        for sector, details in rationale['sector_selections'].items():
            st.markdown(f"""
            **{sector} Sector:**
            - Selected: **{details['selected_stock']}**
            - Avg CAGR: **{details['avg_cagr']:.1f}%**
            - PE Ratio: **{details['pe_ratio']:.1f}**
            - PB Ratio: **{details['pb_ratio']:.1f}**
            - Sector Score: **{details['sector_score']:.3f}**
            - (Chosen from {details['total_in_sector']} stocks in sector)
            """)
        
        st.markdown(f"""
        **Portfolio Summary:**
        - **Achieved CAGR:** {rationale['achieved_cagr']}
        - **Feasibility:** {'✅ Target Achievable' if rationale['feasible'] else '⚠️ Below Target'}
        - **Diversification:** Perfect sector diversification (1 stock per sector)
        """)
        
        st.info("""
        **Why This Simplified Approach?**
        Our sector-based selection ensures maximum diversification by selecting the best stock 
        from each sector based on fundamentals. Primary focus on CAGR (50%), secondary on 
        low PB ratios (40%), and tertiary on optimal PE ranges (10%). This provides balanced 
        exposure across all market sectors while maintaining quality standards.
        """)

def display_investment_summary(summary_data, actual_feasible):
    """Display the detailed investment summary in Streamlit UI"""
    
    # Main header
    st.markdown("---")
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
st.markdown("### AI-Powered Personal Investment Advisor")

# Show disclaimer prominently
show_disclaimer()
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

expected_cagr_pct = st.sidebar.slider("Target CAGR (%)", min_value=8, max_value=99, value=35, step=1, help="Your expected annual returns")
expected_cagr = expected_cagr_pct / 100

# Investment Summary in Sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Investment Summary")
st.sidebar.info(f"""
**Monthly Investment:** ₹{monthly_investment:,}  
**Investment Horizon:** {horizon_years} years  
**Target CAGR:** {expected_cagr_pct}%  
**Total Investment:** ₹{monthly_investment * horizon_months:,}
""")

# ---- Run Optimization ----
if st.button("🚀 Generate Investment Plan", type="primary"):
    with st.spinner("🔍 Analyzing market data and optimizing your portfolio..."):
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

    # Display stock selection rationale
    display_stock_selection_rationale(selection_rationale)
    
    # Display detailed investment summary
    display_investment_summary(summary_data, actual_feasible)
    
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
    
    # Download allocations
    csv_bytes = portfolio_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Allocation CSV", data=csv_bytes, file_name="allocation.csv", mime="text/csv")
    
    st.success("✅ Analysis complete! Review your personalized investment strategy above.")

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
        st.dataframe(df.head(10), use_container_width=True)
