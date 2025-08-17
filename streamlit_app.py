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


st.title("Vriddhi Alpha Finder — MVP")

with st.expander("About this app", expanded=False):
    st.write(
        "This is a private MVP that reads a knowledge asset CSV and uses the Vriddhi optimizer "
        "to produce a suggested allocation and growth projection."
    )

# ---- Data source selection ----
st.sidebar.header("Inputs")
data_source = st.sidebar.radio("Data Source", ["Use bundled CSV", "Upload CSV"])

@st.cache_data
def load_default_csv():
    return pd.read_csv("grand_table.csv")

if data_source == "Upload CSV":
    f = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if f is not None:
        df = pd.read_csv(f)
    else:
        st.info("Please upload a CSV to proceed, or switch to 'Use bundled CSV'.")
        st.stop()
else:
    df = load_default_csv()

# Basic sanity
if "Ticker" not in df.columns:
    st.error("CSV must contain a 'Ticker' column. Found columns: {}".format(", ".join(df.columns)))
    st.stop()

# ---- Parameters ----
monthly_investment = st.sidebar.number_input("Monthly Investment (INR)", min_value=1000, step=1000, value=25000)
horizon_years = st.sidebar.slider("Horizon (years)", min_value=1, max_value=10, value=5)
horizon_months = horizon_years * 12
expected_cagr_pct = st.sidebar.slider("Target CAGR (%)", min_value=5, max_value=40, value=18)
expected_cagr = expected_cagr_pct / 100.0

run = st.sidebar.button("Run Optimization", type="primary")

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
    c1.metric("Feasible", "Yes ✅" if feasible else "No ❌")
    c2.metric("Achieved CAGR", f"{summary.get('Achieved CAGR', 0):.2f}%")
    c3.metric("Final Value", f"₹{summary.get('Final Value', 0):,}")
    c4.metric("Gain", f"₹{summary.get('Gain', 0):,}")

    if feasible:
        st.success("Your investment goals are achievable at the selected horizon and parameters.")
    else:
        st.warning("Your expectations need adjustment — see chart and allocation for guidance.")

    # Display detailed investment summary
    display_investment_summary(summary_data)
    
    st.subheader("Suggested Allocation")
    st.dataframe(allocation_df, use_container_width=True)

    # Download allocations
    csv_bytes = allocation_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Allocation CSV", data=csv_bytes, file_name="vriddhi_allocation.csv", mime="text/csv")

    # ---- Chart: weights ----
    st.subheader("Allocation Weights")
    fig1, ax1 = plt.subplots()
    if "Weight" in allocation_df.columns and "Ticker" in allocation_df.columns:
        ax1.bar(allocation_df["Ticker"], allocation_df["Weight"])
        ax1.set_ylabel("Weight")
        ax1.set_xlabel("Ticker")
        ax1.set_title("Portfolio Weights by Ticker")
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig1, use_container_width=True)

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
    st.info("Set your parameters in the sidebar and click **Run Optimization** to generate results.")
