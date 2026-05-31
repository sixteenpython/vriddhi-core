
# Vriddhi Alpha Finder - Core Investment Optimization Engine
# AI-Powered Personal Investment Advisor
# Built with Modern Portfolio Theory and Advanced Analytics

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings

# seaborn is purely cosmetic here. Guard it so a broken/incompatible install
# (e.g. numpy ABI mismatch on the deploy host) can never break module import,
# which would otherwise crash the whole app via `from vriddhi_core import ...`.
try:
    import seaborn as sns
except Exception:  # pragma: no cover - optional dependency
    sns = None

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
if sns is not None:
    sns.set_palette("husl")

# ===============================
# 1. STOCK SELECTION MODULE (Enhanced with Sector Diversification)
# ===============================

def get_forecast_column(horizon_months):
    return FORECAST_MAP.get(horizon_months, "60M")

def advanced_stock_selector(df, expected_cagr, horizon_months):
    """
    Simplified PEG-based stock selection for maximum CAGR:
    Round 1: Best stock per sector with lowest PEG ratio (PE / Avg_Historical_CAGR)
    Round 2: All remaining stocks with PEG < 1.0
    """
    from collections import defaultdict
    
    # Map horizon to forecast column - matches grand_table_expanded.csv columns
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
    
    # Calculate PEG ratio for all stocks
    df['PEG_Ratio'] = df['PE_Ratio'] / df['Avg_Historical_CAGR']
    
    # Filter out invalid PEG ratios and negative CAGR stocks
    df = df[(df['PE_Ratio'] > 0) & (df['Avg_Historical_CAGR'] > 0)].copy()
    
    # Get unique sectors
    sectors = df['Sector'].unique()
    
    # Round 1: Select best stock from each sector with lowest PEG ratio
    selected_stocks = []
    sector_selections = {}
    used_tickers = set()
    
    for sector in sectors:
        sector_stocks = df[df['Sector'] == sector].copy()
        
        # Sort by PEG ratio (lowest first - best value)
        sector_stocks = sector_stocks.sort_values('PEG_Ratio', ascending=True)
        
        # Select the stock with lowest PEG ratio from this sector
        best_stock = sector_stocks.iloc[0]
        selected_stocks.append(best_stock)
        used_tickers.add(best_stock['Ticker'])
        sector_selections[sector] = {
            'selected_stock': best_stock['Ticker'],
            'avg_cagr': best_stock['Avg_Historical_CAGR'],
            'pe_ratio': best_stock['PE_Ratio'],
            'pb_ratio': best_stock['PB_Ratio'],
            'peg_ratio': best_stock['PEG_Ratio'],
            'total_in_sector': len(sector_stocks)
        }
    
    # Round 2: Select all remaining stocks with PEG < 1.0
    remaining_stocks = df[~df['Ticker'].isin(used_tickers)].copy()
    
    # Apply PEG filter for additional selection
    quality_stocks = remaining_stocks[remaining_stocks['PEG_Ratio'] < 1.0].copy()
    
    # Sort by PEG ratio and select all qualifying stocks
    if len(quality_stocks) > 0:
        quality_stocks = quality_stocks.sort_values('PEG_Ratio', ascending=True)
        
        # Add all qualifying stocks
        for _, stock in quality_stocks.iterrows():
            selected_stocks.append(stock)
            
            # Update sector selections to track additional picks
            sector = stock['Sector']
            if f"{sector}_additional" not in sector_selections:
                sector_selections[f"{sector}_additional"] = []
            
            sector_selections[f"{sector}_additional"].append({
                'selected_stock': stock['Ticker'],
                'avg_cagr': stock['Avg_Historical_CAGR'],
                'pe_ratio': stock['PE_Ratio'],
                'pb_ratio': stock['PB_Ratio'],
                'peg_ratio': stock['PEG_Ratio']
            })
    
    # Convert to DataFrame
    selected_df = pd.DataFrame(selected_stocks)
    
    # Calculate portfolio statistics using Avg_Historical_CAGR
    if len(selected_df) > 0:
        portfolio_cagr = selected_df[forecast_col].mean() / 100
    else:
        portfolio_cagr = 0
    
    # Always return best possible portfolio (no feasibility check needed)
    feasible = True  # App now focuses on best recommendations rather than target matching
    
    # Create selection rationale
    selection_rationale = {
        "total_universe": len(df),
        "after_quality_filters": len(df),
        "sectors_available": len(sectors),
        "stocks_selected": len(selected_df),
        "selection_method": "PEG-based stock selection for maximum CAGR optimization",
        "selection_criteria": [
            "Round 1: Lowest PEG ratio per sector (PE / Avg_Historical_CAGR)",
            "Round 2: All remaining stocks with PEG < 1.0"
        ],
        "quality_filters": [
            "PE Ratio > 0 (valid valuation)",
            "Average CAGR > 0 (positive performance - no negative CAGR stocks)",
            "Valid PEG ratio calculation"
        ],
        "diversification_approach": "Sector diversification through Round 1 + PEG-filtered growth stocks in Round 2",
        "sector_breakdown": sector_selections,
        "achieved_cagr": f"{portfolio_cagr*100:.1f}%",
        "feasible": feasible,
        "fallback_used": False
    }
    
    if not feasible:
        selection_rationale["feasibility_note"] = f"Target {expected_cagr*100:.1f}% CAGR not achieved with sector diversification approach. Best achievable: {portfolio_cagr*100:.1f}%"
    
    return selected_df, feasible, portfolio_cagr, selection_rationale

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
    # Use PE ratio as risk proxy (normalized); Historical_Volatility not in current dataset
    risks = selected_df["PE_Ratio"].values / 100
    cov_matrix = np.diag(risks ** 2)

    def objective(weights):
        port_return = np.dot(weights, returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -port_return / port_vol

    n = len(returns)
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    bounds = [(0, 1)] * n
    init_guess = np.ones(n) / n

    # Imported lazily so a scipy/numpy ABI mismatch on the host can't break
    # module import for the dashboard, which never calls this optimizer.
    from scipy.optimize import minimize
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

    # Subplot 3: Sector Diversification (if provided)
    if optimized_df is not None and len(optimized_df) > 0:
        ax3 = plt.subplot(2, 3, 4)
        
        # Calculate sector-wise allocation amounts
        sector_allocation = optimized_df.groupby('Sector')['Monthly Allocation (INR)'].sum()
        sector_percentages = (sector_allocation / sector_allocation.sum()) * 100
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(sector_allocation)))
        wedges, texts, autotexts = ax3.pie(sector_percentages, labels=sector_allocation.index,
                                          autopct='%1.1f%%', colors=colors, startangle=90)

        ax3.set_title('🎯 Sector Diversification %', fontsize=12, fontweight='bold')

        for autotext in autotexts:
            autotext.set_weight('bold')
            autotext.set_fontsize(9)

    # Subplot 4: Monthly Investment Breakdown (All Stocks)
    ax4 = plt.subplot(2, 3, 5)
    if optimized_df is not None and len(optimized_df) > 0:
        # Show all selected stocks, not just top 8
        monthly_allocations = optimized_df['Monthly Allocation (INR)']
        stock_names = optimized_df['Ticker']

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

    # Load stock database
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


# ===============================
# 8. RESEARCH BUNDLE LOADERS (consumed by the investor dashboard)
# ===============================

RESEARCH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "research")


def load_portfolio_bundle(horizon_years):
    """Load the precomputed research bundle for a given horizon (3, 4 or 5 yr).

    Bundles are produced offline by build_research_db.py and committed under
    ./research/. Returns the parsed dict, or None if the file is missing so the
    UI can degrade gracefully instead of crashing.
    """
    path = os.path.join(RESEARCH_DIR, f"portfolio_{int(horizon_years)}y.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_benchmark_series():
    """Load the normalized Nifty 50 benchmark series (DataFrame) or None."""
    path = os.path.join(RESEARCH_DIR, "benchmark.csv")
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, parse_dates=["Date"])
    except (OSError, ValueError):
        return None


def scale_allocations(bundle, monthly_investment):
    """Turn a bundle's portfolio weights into a concrete monthly SIP plan.

    Allocations are linear in the monthly contribution, so a single precomputed
    bundle per horizon covers the full 50K-1L input range. Returns a DataFrame
    with per-stock monthly amount, whole-share quantity and share cost.
    """
    if not bundle or not bundle.get("stocks"):
        return pd.DataFrame()

    rows = []
    for s in bundle["stocks"]:
        weight = float(s.get("weight", 0) or 0)
        price = float(s.get("current_price", 0) or 0)
        monthly_amount = weight * monthly_investment
        whole_shares = int(monthly_amount // price) if price > 0 else 0
        rows.append({
            "Ticker": s.get("ticker"),
            "Sector": s.get("sector"),
            "Weight": weight,
            "Monthly Allocation (INR)": round(monthly_amount, 2),
            "Current_Price": price,
            "Whole_Shares": whole_shares,
            "Share_Cost": round(whole_shares * price, 2),
        })
    return pd.DataFrame(rows)
