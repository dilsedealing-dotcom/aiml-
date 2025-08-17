#!/usr/bin/env python3
"""
Auto Start Sequence - Complete Automated Training & Trading Pipeline
Runs full sequence: Data → Training → Optimization → Live Trading
"""

import os
import sys
import time
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import TradingSystem
from daily_backtest_optimizer import DailyBacktestOptimizer
from adaptive_live_trader import AdaptiveLiveTrader

class AutoStartSequence:
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol
        self.trading_system = TradingSystem()
        self.success_log = []
        
    def log_step(self, step, status, message=""):
        """Log each step result"""
        entry = {
            'step': step,
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.success_log.append(entry)
        
        status_icon = "✓" if status == "SUCCESS" else "✗" if status == "FAILED" else "⚠"
        print(f"[{status_icon}] Step {step}: {message}")
    
    def run_complete_sequence(self):
        """Run complete automated sequence"""
        print("=" * 70)
        print("AUTO START SEQUENCE - COMPLETE AI TRADING PIPELINE")
        print("=" * 70)
        print(f"Symbol: {self.symbol}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Step 1: Setup directories
        try:
            self.trading_system.setup_directories()
            self.log_step(1, "SUCCESS", "Directories created")
        except Exception as e:
            self.log_step(1, "FAILED", f"Directory setup failed: {e}")
            return False
        
        # Step 2: Download fresh data
        try:
            print(f"\n[2/8] Downloading latest market data...")
            df = self.trading_system.download_and_store_data(self.symbol, count=50000)
            if df is not None:
                self.log_step(2, "SUCCESS", f"Downloaded {len(df)} bars")
            else:
                self.log_step(2, "FAILED", "Data download failed")
                return False
        except Exception as e:
            self.log_step(2, "FAILED", f"Data download error: {e}")
            return False
        
        # Step 3: Run correlation analysis
        try:
            print(f"\n[3/8] Running correlation analysis...")
            results = self.trading_system.run_correlation_analysis(self.symbol)
            if results:
                self.log_step(3, "SUCCESS", "Correlation analysis completed")
            else:
                self.log_step(3, "WARNING", "Correlation analysis had issues")
        except Exception as e:
            self.log_step(3, "WARNING", f"Correlation analysis error: {e}")
        
        # Step 4: Train enhanced model
        try:
            print(f"\n[4/8] Training enhanced ML model...")
            model = self.trading_system.train_enhanced_model(self.symbol)
            if model:
                self.log_step(4, "SUCCESS", "Enhanced model trained")
            else:
                self.log_step(4, "WARNING", "Enhanced model training had issues")
        except Exception as e:
            self.log_step(4, "WARNING", f"Enhanced model error: {e}")
        
        # Step 5: Pre-train multi-timeframe models
        try:
            print(f"\n[5/8] Pre-training multi-timeframe models...")
            multi_tf_data = self.trading_system.ml_trainer.collect_multi_timeframe_data(self.symbol, bars_per_tf=15000)
            
            if multi_tf_data:
                combined_data = self.trading_system.ml_trainer.create_enhanced_features(multi_tf_data)
                models = self.trading_system.ml_trainer.train_ensemble_models(combined_data)
                model_file = self.trading_system.ml_trainer.save_pretrained_models(self.symbol)
                self.log_step(5, "SUCCESS", f"Multi-timeframe models trained: {len(multi_tf_data)} timeframes")
            else:
                self.log_step(5, "FAILED", "Multi-timeframe training failed")
                return False
        except Exception as e:
            self.log_step(5, "FAILED", f"Multi-timeframe training error: {e}")
            return False
        
        # Step 6: Run daily optimization
        try:
            print(f"\n[6/8] Running daily backtest optimization...")
            optimizer = DailyBacktestOptimizer(self.symbol)
            results = optimizer.run_weekly_optimization()
            
            if results:
                patterns = results['patterns']
                self.log_step(6, "SUCCESS", f"Optimization complete - Win rate: {patterns['avg_win_rate']:.1%}")
            else:
                self.log_step(6, "WARNING", "Optimization completed with limited results")
        except Exception as e:
            self.log_step(6, "WARNING", f"Optimization error: {e}")
        
        # Step 7: Run enhanced backtest
        try:
            print(f"\n[7/8] Running enhanced backtest validation...")
            df = self.trading_system.load_latest_data(self.symbol)
            if df is not None:
                metrics, _ = self.trading_system.enhanced_backtester.run_enhanced_backtest(df)
                win_rate = metrics.get('win_rate', 0)
                total_return = metrics.get('total_return_pct', 0)
                self.log_step(7, "SUCCESS", f"Backtest complete - Win rate: {win_rate:.1%}, Return: {total_return:.1f}%")
            else:
                self.log_step(7, "WARNING", "Backtest validation skipped - no data")
        except Exception as e:
            self.log_step(7, "WARNING", f"Backtest error: {e}")
        
        # Step 8: Start adaptive live trading
        try:
            print(f"\n[8/8] Starting adaptive live trading...")
            self.log_step(8, "SUCCESS", "Ready for live trading")
            
            # Print summary
            self.print_sequence_summary()
            
            # Ask user if they want to start live trading
            start_trading = input("\nStart adaptive live trading now? (y/n): ").lower().strip()
            
            if start_trading == 'y':
                print(f"\nStarting adaptive live trading for {self.symbol}...")
                trader = AdaptiveLiveTrader(self.symbol)
                trader.run_adaptive_trading()
            else:
                print("Auto sequence complete. Use main.py Option 11 to start trading later.")
                
        except Exception as e:
            self.log_step(8, "FAILED", f"Live trading setup error: {e}")
            return False
        
        return True
    
    def print_sequence_summary(self):
        """Print summary of sequence results"""
        print("\n" + "=" * 70)
        print("AUTO SEQUENCE SUMMARY")
        print("=" * 70)
        
        success_count = len([log for log in self.success_log if log['status'] == 'SUCCESS'])
        warning_count = len([log for log in self.success_log if log['status'] == 'WARNING'])
        failed_count = len([log for log in self.success_log if log['status'] == 'FAILED'])
        
        print(f"Total Steps: {len(self.success_log)}")
        print(f"Successful: {success_count}")
        print(f"Warnings: {warning_count}")
        print(f"Failed: {failed_count}")
        
        print(f"\nStep Details:")
        for log in self.success_log:
            status_icon = "✓" if log['status'] == "SUCCESS" else "✗" if log['status'] == "FAILED" else "⚠"
            print(f"  {status_icon} Step {log['step']}: {log['message']}")
        
        print(f"\nSystem Status:")
        print(f"Symbol: {self.symbol}")
        print(f"Data files: {len([f for f in os.listdir('data') if f.endswith('.csv')])}")
        print(f"Models: {len([f for f in os.listdir('models') if f.endswith('.pkl')])}")
        
        # Check for key files
        key_files = [
            f'models/{self.symbol}_pretrained_ensemble_',
            f'models/{self.symbol}_adaptive_config.json',
            f'data/{self.symbol}_processed_'
        ]
        
        for file_pattern in key_files:
            matching_files = [f for f in os.listdir('models' if 'models' in file_pattern else 'data') 
                            if f.startswith(file_pattern.split('/')[-1])]
            if matching_files:
                print(f"✓ {file_pattern.split('/')[-1]}* found")
            else:
                print(f"✗ {file_pattern.split('/')[-1]}* missing")
        
        if success_count >= 6:  # At least 6 successful steps
            print(f"\n🚀 SYSTEM READY FOR LIVE TRADING!")
        elif success_count >= 4:
            print(f"\n⚠ SYSTEM PARTIALLY READY - Some features may be limited")
        else:
            print(f"\n❌ SYSTEM NOT READY - Please check errors and retry")

def main():
    """Main auto start function"""
    print("Enhanced MT5 AI Trading System - Auto Start Sequence")
    print("This will automatically run the complete training pipeline")
    
    symbol = input("\nEnter symbol (default XAUUSD): ").strip() or "XAUUSD"
    
    confirm = input(f"\nThis will:\n"
                   f"1. Download 50,000 bars of {symbol} data\n"
                   f"2. Run correlation analysis\n"
                   f"3. Train enhanced ML models\n"
                   f"4. Pre-train multi-timeframe models\n"
                   f"5. Run daily optimization\n"
                   f"6. Validate with backtest\n"
                   f"7. Setup adaptive trading\n"
                   f"8. Start live trading (optional)\n"
                   f"\nProceed? (y/n): ").lower().strip()
    
    if confirm != 'y':
        print("Auto start cancelled.")
        return
    
    # Run auto sequence
    auto_start = AutoStartSequence(symbol)
    success = auto_start.run_complete_sequence()
    
    if success:
        print(f"\n✓ Auto start sequence completed successfully!")
    else:
        print(f"\n✗ Auto start sequence completed with errors. Check logs above.")

if __name__ == "__main__":
    main()