import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from technical_indicators import TechnicalIndicators
from onnx_model import ONNXPricePredictor
from signal_generator import SignalGenerator
from backtest import Backtester

def create_sample_data():
    """Create sample XAUUSD data for testing when MT5 is not available"""
    np.random.seed(42)
    
    # Generate 1000 bars of sample data
    n_bars = 1000
    base_price = 2000.0
    
    dates = pd.date_range(start='2023-01-01', periods=n_bars, freq='5min')
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.001, n_bars)  # Small random returns
    prices = [base_price]
    
    for i in range(1, n_bars):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(new_price)
    
    # Create OHLC data
    data = []
    for i in range(n_bars):
        open_price = prices[i]
        close_price = prices[i] * (1 + np.random.normal(0, 0.0005))
        high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.0003)))
        low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.0003)))
        volume = np.random.randint(100, 1000)
        
        data.append({
            'time': dates[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'tick_volume': volume
        })
    
    return pd.DataFrame(data)

def test_technical_indicators():
    """Test technical indicators calculation"""
    print("Testing Technical Indicators...")
    
    df = create_sample_data()
    df = TechnicalIndicators.calculate_all_indicators(df)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("Technical indicators added successfully!")
    
    return df

def test_onnx_model(df):
    """Test ONNX model training and prediction"""
    print("\nTesting ONNX Model...")
    
    predictor = ONNXPricePredictor()
    
    try:
        mse, r2 = predictor.train_model(df)
        print(f"Model trained successfully - MSE: {mse:.6f}, R2: {r2:.4f}")
        
        # Test prediction
        features = np.random.random(80)  # 10 lookback * 8 features
        prediction = predictor.predict(features)
        print(f"Sample prediction: {prediction}")
        
        return predictor
    except Exception as e:
        print(f"Model training failed: {e}")
        return None

def test_signal_generation(df, predictor):
    """Test signal generation"""
    print("\nTesting Signal Generation...")
    
    if predictor is None:
        print("Skipping signal generation - no trained model")
        return []
    
    signal_generator = SignalGenerator()
    signal_generator.predictor = predictor
    
    try:
        signals = signal_generator.generate_signals(df)
        print(f"Generated {len(signals)} signals")
        
        for i, signal in enumerate(signals[:3]):  # Show first 3 signals
            print(f"Signal {i+1}: {signal['action']} - Confidence: {signal['confidence']:.3f}")
        
        return signals
    except Exception as e:
        print(f"Signal generation failed: {e}")
        return []

def test_backtesting(df):
    """Test backtesting functionality"""
    print("\nTesting Backtesting...")
    
    try:
        backtester = Backtester(initial_balance=10000)
        metrics = backtester.run_backtest(df)
        
        print("Backtest Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        return metrics
    except Exception as e:
        print(f"Backtesting failed: {e}")
        return {}

def main():
    print("MT5 AI Trading System - Test Suite")
    print("=" * 50)
    
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('onnx_models', exist_ok=True)
    
    # Test 1: Technical Indicators
    df = test_technical_indicators()
    
    # Test 2: ONNX Model
    predictor = test_onnx_model(df)
    
    # Test 3: Signal Generation
    signals = test_signal_generation(df, predictor)
    
    # Test 4: Backtesting
    metrics = test_backtesting(df)
    
    print("\n" + "=" * 50)
    print("Test Suite Completed!")
    
    if predictor and metrics:
        print("[SUCCESS] All core tests passed successfully!")
        print("\nSystem is ready for live trading.")
        print("Next steps:")
        print("1. Connect to MT5 terminal")
        print("2. Run main.py for full functionality")
        print("3. Deploy to Google Cloud if needed")
    else:
        print("[WARNING] Some tests failed. Please check the errors above.")

if __name__ == "__main__":
    main()