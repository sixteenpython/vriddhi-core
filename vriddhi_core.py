
# Vriddhi Alpha Finder - Core Investment Optimization Engine
# AI-Powered Personal Investment Advisor
# Built with Modern Portfolio Theory and Advanced Analytics

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

# ===============================
# CONFIGURATION & WARNING SUPPRESSION
# ===============================

# Suppress matplotlib font warnings for emojis in Google Colab
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, module='IPython')
warnings.filterwarnings("ignore", category=UserWarning, message="Glyph .* missing from font")

FORECAST_MAP = {
    12: "12M", 18: "18M", 24: "24M", 36: "36M", 48: "48M", 60: "60M",
}

# Set style for better visuals
plt.style.use('default')
sns.set_palette("husl")

# ===============================
# 1. STOCK SELECTION MODULE (Enhanced with Sector Diversification)
# ===============================

def get_forecast_column(horizon_months):
    return FORECAST_MAP.get(horizon_months, "60M")

def advanced_stock_selector(df, expected_cagr, horizon_months):
    """
    Simplified sector-based stock selection: Best stock from each sector
    Selection criteria: Highest Avg CAGR, Lowest PB, PE between 15-25 preferred
    """
    from collections import defaultdict
    
    # Map horizon to forecast column
    if horizon_months <= 12:
        forecast_col = 'Forecast_12M'
    elif horizon_months <= 18:
        forecast_col = 'Forecast_18M'  
    elif horizon_months <= 24:
        forecast_col = 'Forecast_24M'
    elif horizon_months <= 36:
        forecast_col = 'Forecast_36M'
    elif horizon_months <= 48:
        forecast_col = 'Forecast_48M'
    else:
        forecast_col = 'Forecast_60M'
    
    def calculate_sector_score(row):
        """Calculate score for best stock selection within each sector"""
        
        # Primary: Average Historical CAGR (50% weight)
        avg_cagr_score = row['Avg_Historical_CAGR'] / 50.0  # Normalize to max 50%
        
        # Secondary: PE ratio preference (25% weight) - prefer 15-25 range
        pe_ratio = row['PE_Ratio']
        if 15 <= pe_ratio <= 25:
            pe_score = 1.0  # Perfect score for preferred range
        elif pe_ratio < 15:
            pe_score = 0.8  # Good but not ideal (might be too cheap)
        elif 25 < pe_ratio <= 35:
            pe_score = 0.6  # Acceptable
        else:
            pe_score = 0.2  # Poor (too expensive)
        pe_score *= 0.25
        
        # Tertiary: PB ratio (25% weight) - lower is better
        pb_ratio = row['PB_Ratio']
        pb_score = max(0, (10 - pb_ratio) / 10) * 0.25  # Lower PB gets higher score
        
        return avg_cagr_score + pe_score + pb_score
    
    # Apply sector-based scoring
    df['Sector_Score'] = df.apply(calculate_sector_score, axis=1)
    
    # Get unique sectors
    sectors = df['Sector'].unique()
    
    # Select best stock from each sector
    selected_stocks = []
    sector_selections = {}
    
    for sector in sectors:
        sector_stocks = df[df['Sector'] == sector].copy()
        
        # Sort by sector score (highest first)
        sector_stocks = sector_stocks.sort_values('Sector_Score', ascending=False)
        
        # Select the best stock from this sector
        best_stock = sector_stocks.iloc[0]
        selected_stocks.append(best_stock)
        sector_selections[sector] = {
            'selected_stock': best_stock['Ticker'],
            'avg_cagr': best_stock['Avg_Historical_CAGR'],
            'pe_ratio': best_stock['PE_Ratio'],
            'pb_ratio': best_stock['PB_Ratio'],
            'sector_score': best_stock['Sector_Score'],
            'total_in_sector': len(sector_stocks)
        }
    
    # Convert to DataFrame
    selected_df = pd.DataFrame(selected_stocks)
    
    # Calculate portfolio statistics
    cumulative_cagr = selected_df['Avg_Historical_CAGR'].mean() / 100
    feasible = cumulative_cagr >= expected_cagr
    
    # Create selection rationale for new simplified method
    selection_rationale = {
        "total_universe": len(df),
        "sectors_covered": len(sectors),
        "selection_method": "Sector-based diversification: Best stock from each sector",
        "selection_criteria": [
            "Primary: Highest Average Historical CAGR (50% weight)",
            "Secondary: PE ratio 15-25 preferred (25% weight)", 
            "Tertiary: Lowest PB ratio (25% weight)"
        ],
        "sector_selections": sector_selections,
        "stocks_selected": len(selected_stocks),
        "achieved_cagr": f"{cumulative_cagr*100:.1f}%",
        "feasible": feasible
    }
    
    return selected_df, feasible, cumulative_cagr, selection_rationale

# Legacy wrapper for backward compatibility
def stock_selector(df, expected_cagr, horizon_months):
    """Wrapper function to maintain compatibility with existing code"""
    return advanced_stock_selector(df, expected_cagr, horizon_months)

# ===============================
# 2. OPTIMIZATION MODULE (MPT)
# ===============================

def optimize_portfolio(selected_df, horizon_months):
    # Map horizon to forecast column for expanded database
    forecast_map = {
        6: 'Forecast_6M', 
        12: 'Forecast_12M', 
        18: 'Forecast_18M', 
        24: 'Forecast_24M',
        36: 'Forecast_36M',
        48: 'Forecast_48M', 
        60: 'Forecast_60M'
    }
    forecast_col = forecast_map.get(horizon_months, 'Forecast_24M')
    
    returns = selected_df[forecast_col].values
    # Use PE ratio as risk proxy since Historical_Volatility is not available
    # Higher PE = higher risk, normalize to 0.1-0.5 range
    pe_ratios = selected_df["PE_Ratio"].values
    risks = 0.1 + (pe_ratios / 100) * 0.4  # Scale PE to risk range
    cov_matrix = np.diag(risks ** 2)

    def objective(weights):
        port_return = np.dot(weights, returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -port_return / port_vol

    n = len(returns)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [(0, 1)] * n
    init_guess = np.ones(n) / n

    result = minimize(objective, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        selected_df['Weight'] = result.x
    else:
        selected_df['Weight'] = 1.0 / n

    return selected_df

# ===============================
# 3. FINAL PROJECTION & METRICS
# ===============================

def compute_projection(df, monthly_investment, horizon_months, horizon_cagr):
    # horizon_cagr is already in decimal format (0.30 for 30%)
    monthly_cagr = horizon_cagr / 12
    total_investment = monthly_investment * horizon_months
    if monthly_cagr > 0:
        future_value = monthly_investment * (((1 + monthly_cagr) ** horizon_months - 1) / monthly_cagr)
    else:
        future_value = total_investment
    gain = future_value - total_investment
    return round(total_investment), round(future_value), round(gain)

# ===============================
# 4. ENHANCED VISUALIZATION MODULE
# ===============================

def plot_enhanced_projection(monthly_investment, horizon_months, achieved_cagr, optimized_df=None):
    """
    Creates comprehensive investment visualization with multiple subplots
    """
    fig = plt.figure(figsize=(16, 12))

    # Calculate projections for the specified horizon
    months = np.arange(1, horizon_months + 1)
    monthly_cagr = achieved_cagr / 12  # achieved_cagr is already in decimal format

    # Calculate cumulative investment (linear)
    cumulative_invested = monthly_investment * months

    # Calculate projected value using monthly compounding
    projected_values = []
    for month in months:
        if monthly_cagr > 0:
            fv = monthly_investment * (((1 + monthly_cagr) ** month - 1) / monthly_cagr)
        else:
            fv = monthly_investment * month
        projected_values.append(fv)

    projected_values = np.array(projected_values)
    gains = projected_values - cumulative_invested

    # Subplot 1: Main Investment Growth Chart
    ax1 = plt.subplot(2, 3, (1, 2))
    ax1.fill_between(months, cumulative_invested, projected_values,
                     alpha=0.3, color='green', label='Potential Gains')
    ax1.plot(months, cumulative_invested, '--', linewidth=3, color='#2E86AB',
             label='Total Investment', marker='o', markersize=3, markevery=12)
    ax1.plot(months, projected_values, '-', linewidth=3, color='#A23B72',
             label=f'Portfolio Value ({achieved_cagr*100:.1f}% CAGR)', marker='s', markersize=4, markevery=12)

    key_months = [m for m in [12, 24, 36, 48, 60] if m <= horizon_months]
    for month in key_months:
        idx = month - 1
        ax1.annotate(f'₹{projected_values[idx]/100000:.1f}L',
                    xy=(month, projected_values[idx]),
                    xytext=(month, projected_values[idx] + max(projected_values)*0.08),
                    ha='center', fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    ax1.set_xlabel('Investment Period (Months)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Amount (₹)', fontsize=12, fontweight='bold')
    ax1.set_title(f'💰 Your Investment Journey: ₹{monthly_investment:,}/month for {horizon_months} months',
                  fontsize=14, fontweight='bold', pad=20)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x/100000:.1f}L' if x >= 100000 else f'₹{x/1000:.0f}K'))

    # Subplot 2: Year-on-Year Growth
    ax2 = plt.subplot(2, 3, 3)
    years = list(range(1, int(horizon_months / 12) + 1))
    year_months = [m for m in [12, 24, 36, 48, 60] if m <= horizon_months]
    year_invested = [cumulative_invested[m-1] for m in year_months]
    year_projected = [projected_values[m-1] for m in year_months]
    year_gains = [year_projected[i] - year_invested[i] for i in range(len(years))]

    x_pos = np.arange(len(years))
    width = 0.35
    bars1 = ax2.bar(x_pos - width/2, [v/100000 for v in year_invested], width,
                    label='Invested', color='#2E86AB', alpha=0.8)
    bars2 = ax2.bar(x_pos + width/2, [v/100000 for v in year_gains], width,
                    label='Gains', color='#F18F01', alpha=0.8)

    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()
        ax2.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.1,
                f'₹{height1:.1f}L', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax2.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.1,
                f'₹{height2:.1f}L', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Amount (₹ Lakhs)', fontsize=12, fontweight='bold')
    ax2.set_title('📈 Year-wise Breakdown', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f'Year {y}' for y in years])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Portfolio Allocation (if provided)
    if optimized_df is not None and len(optimized_df) > 0:
        ax3 = plt.subplot(2, 3, 4)
        top_holdings = optimized_df.nlargest(8, 'Weight')
        others_weight = optimized_df[~optimized_df.index.isin(top_holdings.index)]['Weight'].sum()

        if others_weight > 0:
            plot_data = pd.concat([top_holdings, pd.DataFrame({
                'Ticker': ['Others'], 'Weight': [others_weight]
            })], ignore_index=True)
        else:
            plot_data = top_holdings

        colors = plt.cm.Set3(np.linspace(0, 1, len(plot_data)))
        wedges, texts, autotexts = ax3.pie(plot_data['Weight'], labels=plot_data['Ticker'],
                                          autopct='%1.1f%%', colors=colors, startangle=90)

        ax3.set_title('🎯 Portfolio Allocation', fontsize=12, fontweight='bold')

        for autotext in autotexts:
            autotext.set_weight('bold')
            autotext.set_fontsize(9)

    # Subplot 4: Monthly Investment Breakdown
    ax4 = plt.subplot(2, 3, 5)
    if optimized_df is not None and len(optimized_df) > 0:
        monthly_allocations = optimized_df['Monthly Allocation (INR)'].head(8)
        stock_names = optimized_df['Ticker'].head(8)

        bars = ax4.barh(range(len(monthly_allocations)), monthly_allocations,
                       color=plt.cm.magma(np.linspace(0, 0.8, len(monthly_allocations))),
                       edgecolor='gray', alpha=0.9, linewidth=0.5)

        ax4.set_yticks(range(len(monthly_allocations)))
        ax4.set_yticklabels(stock_names, fontsize=10, fontweight='bold', rotation=15)
        ax4.set_xlabel('Monthly Investment (₹)', fontsize=12, fontweight='bold')
        ax4.set_title('💸 Monthly Stock Allocation', fontsize=14, fontweight='bold', pad=10)

        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax4.text(width + max(monthly_allocations)*0.01, bar.get_y() + bar.get_height()/2,
                    f'₹{width:,.0f}', ha='left', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))

        ax4.tick_params(axis='x', labelsize=10, labelcolor='black', width=1.5)
        ax4.grid(axis='x', linestyle='--', alpha=0.2, color='gray')
        ax4.set_xlim(0, max(monthly_allocations) * 1.2)

    # Subplot 5: Key Metrics Summary
    ax5 = plt.subplot(2, 3, 6)  # Define ax5 here
    ax5.axis('off')

    # Calculate final values
    final_invested = cumulative_invested[horizon_months-1]
    final_value = projected_values[horizon_months-1]
    total_gain = final_value - final_invested
    gain_percentage = (total_gain / final_invested) * 100

    # Create summary text
    summary_text = f"""
    📊 INVESTMENT SUMMARY

    🎯 Target Period: {horizon_months} months ({horizon_months/12:.1f} years)
    💰 Monthly Investment: ₹{monthly_investment:,}
    📈 Achievable CAGR: {achieved_cagr * 100:.2f}%

    💵 Total Investment: ₹{final_invested:,.0f}
    🚀 Final Portfolio Value: ₹{final_value:,.0f}
    💎 Total Gains: ₹{total_gain:,.0f}
    """

    ax5.text(0.1, 0.9, summary_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5",
             facecolor="lightblue", alpha=0.8), fontweight='bold')

    plt.tight_layout(pad=3.0)
    plt.suptitle(f'🌟 VRIDDHI INVESTMENT PLAN - Complete Analysis 🌟',
                 fontsize=16, fontweight='bold', y=0.98)
    return fig

# ===============================
# 4. FINAL OUTPUT MODULE (Updated with Frill Integration)
# ===============================

def final_summary_output(feasible: bool, horizon_months: int, expected_cagr: float, achieved_cagr: float, projected_value: float, total_invested: float, best_horizon_60_cagr: float, max_possible_cagr_current_horizon: float, frill_output: dict):
    horizon_years = horizon_months / 12
    gain = projected_value - total_invested

    # Create structured output for both console and Streamlit
    summary_data = {
        "feasible": feasible,
        "horizon_years": horizon_years,
        "horizon_months": horizon_months,
        "gain": gain,
        "expected_cagr": expected_cagr * 100,  # Convert to percentage for UI display
        "achieved_cagr": achieved_cagr * 100,  # Convert to percentage for UI display
        "projected_value": projected_value,
        "total_invested": total_invested,
        "money_multiplier": projected_value/total_invested,
        "monthly_avg_gain": gain/horizon_months,
        "total_return_pct": ((projected_value/total_invested - 1) * 100),
        "cagr_gap": (expected_cagr - max_possible_cagr_current_horizon) * 100,  # Convert to percentage
        "best_horizon_60_cagr": best_horizon_60_cagr * 100,  # Convert to percentage
        "inflation_beat": (achieved_cagr * 100) - 6  # Convert to percentage for comparison
    }

    # Console output (existing functionality)
    print("="*80)
    print("🎯 VRIDDHI INVESTMENT ANALYSIS REPORT")
    print("="*80)

    # Display the technical frill output first
    print("🔢 TECHNICAL SUMMARY:")
    print(f"   Feasible: {frill_output['Feasible']}")
    print(f"   Expected CAGR: {frill_output['Expected CAGR']:.2f}%")
    print(f"   Achieved CAGR: {frill_output['Achieved CAGR']:.2f}%")
    print(f"   Final Value: ₹{frill_output['Final Value']:,}")
    print(f"   Total Gain: ₹{frill_output['Gain']:,}\n")

    if feasible:
        print("🎉 SUCCESS: Your investment goals are ACHIEVABLE! 🎉\n")
        print(f"📋 PLAN SUMMARY:")
        print(f"   • Investment Period: {horizon_years:.1f} years ({horizon_months} months)")
        print(f"   • Total Investment: ₹{int(total_invested):,}")
        print(f"   • Expected CAGR: {expected_cagr*100:.1f}% → Achieved CAGR: {achieved_cagr*100:.1f}%")
        print(f"   • Final Portfolio Value: ₹{int(projected_value):,}")
        print(f"   • Total Gains: ₹{int(gain):,}")
        print(f"   • Money Multiplier: {projected_value/total_invested:.2f}x\n")

        print("✨ WHAT THIS MEANS FOR YOU:")
        print(f"   Your disciplined investment will grow your wealth by ₹{int(gain):,}")
        print(f"   Every ₹1 you invest will become ₹{projected_value/total_invested:.2f}")
        print("   You're on the path to financial growth! 📈\n")

        print("🎊 CELEBRATION METRICS:")
        print(f"   • You'll be ₹{int(gain):,} richer!")
        print(f"   • That's ₹{int(gain/horizon_months):,} average gain per month!")
        print(f"   • Your wealth will grow {((projected_value/total_invested - 1) * 100):.1f}% over {horizon_years:.1f} years!")

    else:
        print("⚠️  REALITY CHECK: Your expectations need adjustment ⚠️\n")
        print(f"📋 CURRENT SCENARIO:")
        print(f"   • Desired CAGR: {expected_cagr*100:.1f}%")
        print(f"   • Best Achievable CAGR ({horizon_months} months): {max_possible_cagr_current_horizon*100:.1f}%")
        print(f"   • Gap: {(expected_cagr - max_possible_cagr_current_horizon)*100:.1f}% short\n")

        print("💰 BUT HERE'S THE GOOD NEWS:")
        print(f"   • Even at {achieved_cagr*100:.1f}% CAGR, you'll still gain ₹{int(gain):,}!")
        print(f"   • Your ₹{int(total_invested):,} will become ₹{int(projected_value):,}")
        print(f"   • That's still a {((projected_value/total_invested - 1) * 100):.1f}% total return!")
        print(f"   • Monthly average gain: ₹{int(gain/horizon_months):,}\n")

        print("💡 SMART RECOMMENDATIONS:")
        print(f"   • Option 1: Accept {max_possible_cagr_current_horizon*100:.1f}% CAGR → Gain ₹{int(gain):,}")
        print(f"   • Option 2: Extend to 60 months for up to {best_horizon_60_cagr*100:.1f}% CAGR")
        print(f"   • Option 3: Increase monthly investment to reach your target faster")
        print(f"   • Option 4: Adjust expectations - {achieved_cagr*100:.1f}% is still excellent!\n")

        print("🧠 PERSPECTIVE CHECK:")
        print(f"   • Bank FD gives ~7% → You're getting {achieved_cagr*100:.1f}%!")
        print(f"   • Inflation is ~6% → You're beating it by {(achieved_cagr*100) - 6:.1f}%!")
        print("   • This is solid wealth creation, even if not your original target!")

    print("="*80)
    
    return summary_data

# ===============================
# 6. WHOLE SHARE ALLOCATION MODULE
# ===============================

def calculate_whole_share_allocation(optimized_df, full_df):
    """
    Calculate whole share allocation based on optimal weights
    
    Args:
        optimized_df: DataFrame with optimal weights and monthly allocations
        full_df: Full dataset with Current_Price information
        
    Returns:
        DataFrame with whole share recommendations
    """
    # Merge to get current prices - handle both Current_Price and Expected_Inc_Price columns
    merged_df = optimized_df.merge(full_df[['Ticker', 'Current_Price']], on='Ticker', how='left', suffixes=('', '_from_full'))
    
    # Handle duplicate column names from merge
    if 'Current_Price_from_full' in merged_df.columns:
        merged_df['Current_Price'] = merged_df['Current_Price_from_full']
        merged_df = merged_df.drop(columns=['Current_Price_from_full'])
    
    # Verify Current_Price column exists and has valid data
    if 'Current_Price' not in merged_df.columns:
        raise KeyError(f"Current_Price column missing after merge. Available columns: {merged_df.columns.tolist()}")
    
    # Check for null values in Current_Price
    null_prices = merged_df['Current_Price'].isnull().sum()
    if null_prices > 0:
        print(f"Warning: {null_prices} stocks have null prices")
        print("Stocks with null prices:", merged_df[merged_df['Current_Price'].isnull()]['Ticker'].tolist())
    
    # Calculate target shares based on optimal weights and current prices
    target_shares = []
    actual_shares = []
    share_costs = []
    
    for _, row in merged_df.iterrows():
        target_allocation = row['Monthly Allocation (INR)']
        current_price = row['Current_Price']
        
        # Calculate ideal number of shares (fractional)
        ideal_shares = target_allocation / current_price
        
        # Round to nearest whole number, but ensure minimum 1 share
        whole_shares = max(1, round(ideal_shares))
        
        target_shares.append(ideal_shares)
        actual_shares.append(whole_shares)
        share_costs.append(whole_shares * current_price)
    
    # Create whole share allocation DataFrame
    whole_share_df = merged_df.copy()
    whole_share_df['Target_Shares'] = target_shares
    whole_share_df['Whole_Shares'] = actual_shares
    whole_share_df['Share_Cost'] = share_costs
    whole_share_df['Actual_Weight'] = whole_share_df['Share_Cost'] / whole_share_df['Share_Cost'].sum()
    
    # Calculate total monthly investment required
    total_monthly_investment = whole_share_df['Share_Cost'].sum()
    
    # Add summary information
    whole_share_df['Total_Monthly_Investment'] = total_monthly_investment
    
    return whole_share_df[['Ticker', 'Current_Price', 'Weight', 'Whole_Shares', 'Share_Cost', 'Actual_Weight', 'Total_Monthly_Investment']]

# ===============================
# 7. WRAPPER FUNCTION (Updated)
# ===============================

def run_vriddhi_backend(monthly_investment, horizon_months, expected_cagr):
    print("🚀 Starting Vriddhi Investment Analysis...\n")

    # Load expanded database
    df = pd.read_csv("grand_table_expanded.csv")
    
    # Convert CAGR from percentage to decimal for calculations
    expected_cagr_decimal = expected_cagr / 100

    selected_df, feasible, achieved_cagr, selection_rationale = stock_selector(df, expected_cagr_decimal, horizon_months)
    
    # Re-check feasibility after stock selection with proper comparison
    feasible = achieved_cagr >= expected_cagr_decimal
    optimized_df = optimize_portfolio(selected_df, horizon_months)
    optimized_df["Monthly Allocation (INR)"] = optimized_df["Weight"] * monthly_investment

    total_invested, final_value, gain = compute_projection(
        optimized_df, monthly_investment, horizon_months, achieved_cagr
    )

    _, _, best_cagr_60, _ = stock_selector(df, expected_cagr_decimal, 60)

    # Create the frill output dictionary
    frill_output = {
        "Feasible": feasible,
        "Expected CAGR": expected_cagr * 100,  # Convert decimal to percentage for display
        "Achieved CAGR": achieved_cagr * 100,  # achieved_cagr is already decimal from stock_selector
        "Final Value": final_value,
        "Gain": gain
    }

    summary_data = final_summary_output(
        feasible=feasible,
        horizon_months=horizon_months,
        expected_cagr=expected_cagr,
        achieved_cagr=achieved_cagr,
        projected_value=final_value,
        total_invested=total_invested,
        best_horizon_60_cagr=best_cagr_60,
        max_possible_cagr_current_horizon=achieved_cagr,
        frill_output=frill_output
    )

    # Enhanced visualization
    print("\n📊 Generating comprehensive investment visualization...\n")
    fig = plot_enhanced_projection(monthly_investment, horizon_months, achieved_cagr, optimized_df)

    # Calculate whole share allocation
    whole_share_df = calculate_whole_share_allocation(optimized_df, df)
    
    return optimized_df[['Ticker', 'Weight', 'Monthly Allocation (INR)']], fig, frill_output, summary_data, selection_rationale, whole_share_df
