#!/usr/bin/env python3
"""
Quick Start Script for Enhanced MT5 AI Trading System
Automatically sets up and runs the complete workflow
"""

import os
import sys
from main import TradingSystem

def quick_setup_and_run():
    """Complete automated setup and demo"""
    
    print("=" * 70)
    print("ENHANCED MT5 AI TRADING SYSTEM - QUICK START")
    print("=" * 70)
    
    # Initialize system
    trading_system = TradingSystem()
    trading_system.setup_directories()
    
    symbol = "XAUUSD"
    
    print(f"\nRunning complete workflow for {symbol}...")
    
    # Step 1: Download real data
    print("\n[1/5] Downloading real market data...")
    df = trading_system.download_and_store_data(symbol, count=30000)
    
    if df is None:
        print("Failed to download data. Please check MT5 connection.")
        return
    
    # Step 2: Run correlation analysis
    print("\n[2/5] Running correlation analysis...")
    try:
        results = trading_system.run_correlation_analysis(symbol)
        print("Correlation analysis completed successfully")
    except Exception as e:
        print(f"Correlation analysis failed: {e}")
    
    # Step 3: Pre-train multi-timeframe models
    print("\n[3/5] Pre-training multi-timeframe ML models...")
    try:
        # Collect multi-timeframe data
        multi_tf_data = trading_system.ml_trainer.collect_multi_timeframe_data(symbol, bars_per_tf=10000)
        
        if multi_tf_data:
            # Create enhanced features
            combined_data = trading_system.ml_trainer.create_enhanced_features(multi_tf_data)
            
            # Train ensemble models
            models = trading_system.ml_trainer.train_ensemble_models(combined_data)
            
            # Save models
            model_file = trading_system.ml_trainer.save_pretrained_models(symbol)
            
            print(f"Pre-training complete! Models saved.")
        else:
            print("Failed to collect multi-timeframe data")
    except Exception as e:
        print(f"Pre-training failed: {e}")
    
    # Step 4: Run enhanced backtest
    print("\n[4/5] Running enhanced backtest...")
    try:
        df = trading_system.load_latest_data(symbol)
        if df is not None:
            metrics, _ = trading_system.enhanced_backtester.run_enhanced_backtest(df)
            
            print("\nBacktest Results Summary:")
            print(f"Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"Total Return: {metrics.get('total_return_pct', 0):.1f}%")
            print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {metrics.get('max_drawdown_pct', 0):.1f}%")
        else:
            print("No data available for backtesting")
    except Exception as e:
        print(f"Backtesting failed: {e}")
    
    # Step 5: Demo memory-enhanced trading
    print("\n[5/5] Demonstrating memory-enhanced trading...")
    try:
        from memory_enhanced_trader import MemoryEnhancedTrader
        
        # Initialize memory trader
        memory_trader = MemoryEnhancedTrader(symbol)
        
        # Load models and memory
        if memory_trader.load_pretrained_models():
            memory_trader.load_trade_memory()
            
            # Get latest data and generate signals
            df = trading_system.data_processor.download_and_process_data(symbol, count=1000)
            
            if df is not None:
                signals = memory_trader.generate_memory_enhanced_signals(df)
                
                print(f"\nGenerated {len(signals)} memory-enhanced signals:")
                for signal in signals:
                    print(f"  {signal['action']} @ {signal['price']:.5f} (Confidence: {signal['confidence']:.3f})")
            else:
                print("Failed to get latest data for signal generation")
        else:
            print("No pre-trained models available for memory trading")
    except Exception as e:
        print(f"Memory trading demo failed: {e}")
    
    print("\n" + "=" * 70)
    print("QUICK START COMPLETE!")
    print("=" * 70)
    print("\nSystem is now ready for:")
    print("• Real-time memory-enhanced trading")
    print("• Continuous learning from trade results")
    print("• Multi-timeframe analysis")
    print("• Advanced correlation-based signals")
    
    print(f"\nTo start live trading, run: python main.py (Option 6)")
    print(f"All data and models are saved in respective directories.")

if __name__ == "__main__":
    quick_setup_and_run()