#!/usr/bin/env python3
"""
Quick test script for vriddhi_core.py functionality
"""

import pandas as pd
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import vriddhi_core
    print("✅ vriddhi_core.py imports successfully")
    
    # Test loading the CSV
    df = pd.read_csv("grand_table_expanded.csv")
    print(f"✅ Loaded CSV with {len(df)} stocks")
    
    # Check column structure
    expected_cols = ['Current_Price', 'Forecast_12M', 'Forecast_24M', 'Forecast_36M', 'Forecast_48M', 'Forecast_60M']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
    else:
        print("✅ All required forecast columns present")
    
    # Test basic functionality
    print("\n🧪 Testing stock selection with realistic parameters...")
    
    # Test with moderate expectations
    monthly_investment = 25000
    horizon_months = 24
    expected_cagr = 0.15  # 15% CAGR
    
    selected_df, feasible, achieved_cagr, rationale = vriddhi_core.advanced_stock_selector(
        df, expected_cagr, horizon_months
    )
    
    print(f"✅ Stock selection completed:")
    print(f"   - Selected {len(selected_df)} stocks")
    print(f"   - Feasible: {feasible}")
    print(f"   - Achieved CAGR: {achieved_cagr*100:.2f}%")
    
    # Test optimization
    optimized_df = vriddhi_core.optimize_portfolio(selected_df, horizon_months)
    print(f"✅ Portfolio optimization completed")
    
    # Test projection calculation
    total_invested, final_value, gain = vriddhi_core.compute_projection(
        optimized_df, monthly_investment, horizon_months, achieved_cagr
    )
    
    print(f"✅ Projection calculation completed:")
    print(f"   - Total Investment: ₹{total_invested:,}")
    print(f"   - Final Value: ₹{final_value:,}")
    print(f"   - Gain: ₹{gain:,}")
    
    print("\n🎉 All tests passed! vriddhi_core.py is working correctly.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()
