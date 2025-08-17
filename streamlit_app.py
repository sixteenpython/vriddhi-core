
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

import vriddhi_core  # generated from your notebook

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
    with st.spinner("Running Vriddhi optimizer..."):
        try:
            optimized_df, summary = vriddhi_core.run_vriddhi_backend(
                df.copy(),
                monthly_investment=monthly_investment,
                expected_cagr=expected_cagr,
                horizon_months=horizon_months
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    # ---- Results ----
    st.subheader("Summary")

    c1, c2, c3, c4 = st.columns(4)
    feasible = summary.get("Feasible")
    c1.metric("Feasible", "Yes ✅" if feasible else "No ❌")
    c2.metric("Achieved CAGR", f"{summary.get('Achieved CAGR', 0)*100:.2f}%")
    c3.metric("Final Value", f"₹{summary.get('Final Value', 0):,}")
    c4.metric("Gain", f"₹{summary.get('Gain', 0):,}")

    if feasible:
        st.success("Your investment goals are achievable at the selected horizon and parameters.")
    else:
        st.warning("Your expectations need adjustment — see chart and allocation for guidance.")

    st.subheader("Suggested Allocation")
    st.dataframe(optimized_df, use_container_width=True)

    # Download allocations
    csv_bytes = optimized_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Allocation CSV", data=csv_bytes, file_name="vriddhi_allocation.csv", mime="text/csv")

    # ---- Chart: weights ----
    st.subheader("Allocation Weights")
    fig1, ax1 = plt.subplots()
    if "Weight" in optimized_df.columns and "Ticker" in optimized_df.columns:
        ax1.bar(optimized_df["Ticker"], optimized_df["Weight"])
        ax1.set_ylabel("Weight")
        ax1.set_xlabel("Ticker")
        ax1.set_title("Portfolio Weights by Ticker")
        plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
        st.pyplot(fig1, use_container_width=True)

    # ---- Projection chart from your function ----
    st.subheader("Growth Projection")
    try:
        fig2 = vriddhi_core.plot_enhanced_projection(
            monthly_investment=monthly_investment,
            horizon_months=horizon_months,
            achieved_cagr=summary.get("Achieved CAGR"),
            optimized_df=optimized_df
        )
        if fig2 is None:
            import matplotlib.pyplot as _plt
            fig2 = _plt.gcf()
        st.pyplot(fig2, use_container_width=True)
        # Download chart
        buf = io.BytesIO()
        fig2.savefig(buf, format="png", bbox_inches="tight")
        st.download_button("Download Projection PNG", data=buf.getvalue(), file_name="projection.png", mime="image/png")
    except Exception as e:
        st.info("Projection plot not available: " + str(e))

else:
    st.info("Set your parameters in the sidebar and click **Run Optimization** to generate results.")
