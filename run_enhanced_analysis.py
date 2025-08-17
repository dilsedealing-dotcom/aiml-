#!/usr/bin/env python3
"""
Enhanced BB Trading System with Correlation Analysis
Analyzes feature correlations and trains ML model to improve BB signal quality
"""

import pandas as pd
import numpy as np
from enhanced_backtester import EnhancedBacktester
from data_processor import DataProcessor
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample data with your specified features"""
    
    # Create sample data matching your feature format
    np.random.seed(42)
    n_samples = 10000
    
    # Base price data
    base_price = 2000
    price_changes = np.random.normal(0, 0.001, n_samples)
    prices = [base_price]
    
    for change in price_changes:
        prices.append(prices[-1] * (1 + change))
    
    prices = np.array(prices[1:])
    
    # Create DataFrame with normalized features (matching your input)
    data = {
        'time': pd.date_range('2025-01-01', periods=n_samples, freq='5min'),
        'close': prices / prices[0],  # Normalized to ~1.0
        'low': (prices * 0.999) / prices[0],
        'high': (prices * 1.001) / prices[0], 
        'open': (prices * np.random.uniform(0.9995, 1.0005, n_samples)) / prices[0],
        'tick_volume': np.random.uniform(0.1, 0.2, n_samples),
        'atr': np.random.uniform(0.3, 0.4, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate Bollinger Bands
    window = 20
    df['bb_middle'] = df['close'].rolling(window).mean()
    bb_std = df['close'].rolling(window).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    
    # Normalize BB bands
    df['bb_middle'] = df['bb_middle'] / df['close'].iloc[0]
    df['bb_upper'] = df['bb_upper'] / df['close'].iloc[0]
    df['bb_lower'] = df['bb_lower'] / df['close'].iloc[0]
    
    # Add other technical indicators
    df['adx'] = np.random.uniform(0.02, 0.08, n_samples)
    df['adx_strong'] = (df['adx'] > 0.025).astype(float) * np.random.uniform(0.02, 0.03, n_samples)
    df['rsi_overbought'] = np.random.uniform(0.01, 0.05, n_samples)
    df['macd_signal'] = np.random.uniform(0.005, 0.015, n_samples)
    
    # Remove NaN values
    df = df.dropna().reset_index(drop=True)
    
    print(f"Created sample data with {len(df)} bars")
    print("Feature ranges:")
    for col in df.columns:
        if col != 'time':
            print(f"{col:15}: {df[col].min():.6f} - {df[col].max():.6f}")
    
    return df

def main():
    """Main execution function"""
    
    print("="*80)
    print("ENHANCED BOLLINGER BAND TRADING SYSTEM")
    print("Correlation Analysis & ML Enhancement")
    print("="*80)
    
    # Try to load real data first, fallback to sample data
    try:
        data_processor = DataProcessor()
        print("\nAttempting to download real MT5 data...")
        df = data_processor.download_and_process_data(symbol="XAUUSD", count=20000)
        
        if df is None or len(df) < 1000:
            raise Exception("Insufficient real data")
            
        print(f"[OK] Loaded real data: {len(df)} bars")
        
    except Exception as e:
        print(f"[ERROR] Real data unavailable: {e}")
        print("Using sample data for demonstration...")
        df = create_sample_data()
    
    # Initialize enhanced backtester
    backtester = EnhancedBacktester(initial_balance=10000, risk_percentage=0.02)
    
    # Run enhanced backtest with correlation analysis
    print(f"\nRunning enhanced backtest...")
    print(f"Data period: {df['time'].min()} to {df['time'].max()}")
    
    try:
        metrics, correlations = backtester.run_enhanced_backtest(df, start_date='2025-01-01')
        
        # Display results
        print("\n" + "="*80)
        print("FINAL BACKTEST RESULTS")
        print("="*80)
        
        if 'error' in metrics:
            print(f"Error: {metrics['error']}")
            return
        
        # Performance metrics
        print(f"{'Metric':<25} {'Value':<15}")
        print("-" * 40)
        print(f"{'Total Trades':<25} {metrics['total_trades']:<15}")
        print(f"{'Win Rate':<25} {metrics['win_rate']:<15.2%}")
        print(f"{'Total Return':<25} {metrics['total_return_pct']:<15.2f}%")
        print(f"{'Profit Factor':<25} {metrics['profit_factor']:<15.2f}")
        print(f"{'Sharpe Ratio':<25} {metrics['sharpe_ratio']:<15.2f}")
        print(f"{'Max Drawdown':<25} {metrics['max_drawdown_pct']:<15.2f}%")
        print(f"{'Final Balance':<25} ${metrics['final_balance']:<15.2f}")
        print(f"{'Avg Confidence':<25} {metrics['avg_confidence']:<15.2f}")
        print(f"{'High Conf Win Rate':<25} {metrics['high_conf_win_rate']:<15.2%}")
        
        # Top correlations
        print(f"\n" + "="*50)
        print("TOP FEATURE CORRELATIONS WITH BB SIGNALS")
        print("="*50)
        
        if correlations is not None:
            top_corr = correlations.abs().sort_values(ascending=False).head(8)
            for feature, corr in top_corr.items():
                print(f"{feature:<20}: {corr:>8.4f}")
        
        # Plot comprehensive results
        print(f"\nGenerating visualization...")
        backtester.plot_comprehensive_results(df)
        
        print(f"\n[OK] Analysis complete!")
        print(f"[OK] Charts saved to 'data/enhanced_backtest_results.png'")
        print(f"[OK] Correlation analysis saved to 'data/correlation_analysis.png'")
        
        # Feature importance summary
        if hasattr(backtester.correlation_analyzer, 'feature_importance'):
            print(f"\n" + "="*50)
            print("FEATURE IMPORTANCE FOR BB ENHANCEMENT")
            print("="*50)
            
            importance = backtester.correlation_analyzer.feature_importance
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            
            for feature, imp in sorted_importance[:8]:
                print(f"{feature:<20}: {imp:>8.4f}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()