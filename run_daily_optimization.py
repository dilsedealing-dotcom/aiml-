#!/usr/bin/env python3
"""
Quick Daily Optimization Runner
Tests every day from recent week to find optimal entry/exit parameters
"""

from daily_backtest_optimizer import DailyBacktestOptimizer
from adaptive_live_trader import AdaptiveLiveTrader

def main():
    print("=" * 60)
    print("DAILY BACKTEST OPTIMIZATION")
    print("=" * 60)
    
    symbol = "XAUUSD"
    
    # Step 1: Run optimization
    print(f"Optimizing {symbol} for recent week...")
    optimizer = DailyBacktestOptimizer(symbol)
    
    try:
        results = optimizer.run_weekly_optimization()
        
        if results:
            patterns = results['patterns']
            
            print("\n" + "=" * 40)
            print("OPTIMIZATION COMPLETE")
            print("=" * 40)
            print(f"Optimal Entry Threshold: {patterns['best_entry_threshold']}")
            print(f"Optimal Exit Threshold: {patterns['best_exit_threshold']}")
            print(f"Expected Win Rate: {patterns['avg_win_rate']:.2%}")
            print(f"Total P&L: ${patterns['total_pnl']:.2f}")
            print(f"Best Day: {patterns['best_day']}")
            
            # Step 2: Test adaptive trader
            print(f"\nTesting adaptive trader with optimized parameters...")
            trader = AdaptiveLiveTrader(symbol)
            
            print(f"Loaded config:")
            print(f"  Entry: {trader.adaptive_config['optimal_entry_threshold']}")
            print(f"  Exit: {trader.adaptive_config['optimal_exit_threshold']}")
            
            print(f"\nOptimization successful! Ready for live trading.")
            
        else:
            print("Optimization failed - no results")
            
    except Exception as e:
        print(f"Error during optimization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()