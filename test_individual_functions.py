#!/usr/bin/env python3
"""
Individual Function Tester
Tests each main menu function separately to ensure they work correctly
"""

import os
import sys
import traceback
from datetime import datetime

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import TradingSystem

class FunctionTester:
    def __init__(self):
        self.trading_system = TradingSystem()
        self.test_results = []
        self.symbol = "XAUUSD"
        
    def log_test(self, function_name, status, message="", error=None):
        """Log test result"""
        result = {
            'function': function_name,
            'status': status,
            'message': message,
            'error': str(error) if error else None,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status_icon = "✓" if status == "PASS" else "✗" if status == "FAIL" else "⚠"
        print(f"[{status_icon}] {function_name}: {message}")
        if error:
            print(f"    Error: {error}")
    
    def test_setup_directories(self):
        """Test Option 1: Setup directories"""
        try:
            self.trading_system.setup_directories()
            
            # Check if directories exist
            required_dirs = ['data', 'onnx_models', 'ea', 'models', 'logs', 'config']
            missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
            
            if not missing_dirs:
                self.log_test("setup_directories", "PASS", "All directories created successfully")
                return True
            else:
                self.log_test("setup_directories", "FAIL", f"Missing directories: {missing_dirs}")
                return False
                
        except Exception as e:
            self.log_test("setup_directories", "FAIL", "Directory setup failed", e)
            return False
    
    def test_download_data(self):
        """Test Option 1: Download & Store Real Data"""
        try:
            print(f"\nTesting data download for {self.symbol}...")
            df = self.trading_system.download_and_store_data(self.symbol, count=1000)  # Small sample
            
            if df is not None and len(df) > 0:
                self.log_test("download_data", "PASS", f"Downloaded {len(df)} bars successfully")
                return True
            else:
                self.log_test("download_data", "FAIL", "No data downloaded")
                return False
                
        except Exception as e:
            self.log_test("download_data", "FAIL", "Data download failed", e)
            return False
    
    def test_correlation_analysis(self):
        """Test Option 2: Run Correlation Analysis"""
        try:
            print(f"\nTesting correlation analysis...")
            
            # Check if data exists first
            if not self.trading_system.load_latest_data(self.symbol):
                self.log_test("correlation_analysis", "SKIP", "No data available for analysis")
                return False
            
            results = self.trading_system.run_correlation_analysis(self.symbol)
            
            if results:
                self.log_test("correlation_analysis", "PASS", "Correlation analysis completed")
                return True
            else:
                self.log_test("correlation_analysis", "FAIL", "Correlation analysis returned no results")
                return False
                
        except Exception as e:
            self.log_test("correlation_analysis", "FAIL", "Correlation analysis failed", e)
            return False
    
    def test_enhanced_model_training(self):
        """Test Option 3: Train Enhanced ML Model"""
        try:
            print(f"\nTesting enhanced model training...")
            
            # Check if data exists first
            if not self.trading_system.load_latest_data(self.symbol):
                self.log_test("enhanced_model_training", "SKIP", "No data available for training")
                return False
            
            model = self.trading_system.train_enhanced_model(self.symbol)
            
            if model:
                self.log_test("enhanced_model_training", "PASS", "Enhanced model trained successfully")
                return True
            else:
                self.log_test("enhanced_model_training", "FAIL", "Enhanced model training failed")
                return False
                
        except Exception as e:
            self.log_test("enhanced_model_training", "FAIL", "Enhanced model training error", e)
            return False
    
    def test_multi_timeframe_training(self):
        """Test Option 4: Pre-Train Multi-Timeframe Models"""
        try:
            print(f"\nTesting multi-timeframe training...")
            
            # Collect multi-timeframe data (smaller sample for testing)
            multi_tf_data = self.trading_system.ml_trainer.collect_multi_timeframe_data(self.symbol, bars_per_tf=500)
            
            if multi_tf_data:
                # Create enhanced features
                combined_data = self.trading_system.ml_trainer.create_enhanced_features(multi_tf_data)
                
                # Train ensemble models
                models = self.trading_system.ml_trainer.train_ensemble_models(combined_data)
                
                # Save models
                model_file = self.trading_system.ml_trainer.save_pretrained_models(self.symbol)
                
                self.log_test("multi_timeframe_training", "PASS", f"Multi-timeframe models trained: {len(multi_tf_data)} timeframes")
                return True
            else:
                self.log_test("multi_timeframe_training", "FAIL", "Failed to collect multi-timeframe data")
                return False
                
        except Exception as e:
            self.log_test("multi_timeframe_training", "FAIL", "Multi-timeframe training error", e)
            return False
    
    def test_enhanced_backtest(self):
        """Test Option 5: Run Enhanced Backtest"""
        try:
            print(f"\nTesting enhanced backtest...")
            
            df = self.trading_system.load_latest_data(self.symbol)
            if df is not None:
                metrics, _ = self.trading_system.enhanced_backtester.run_enhanced_backtest(df)
                
                if metrics and 'win_rate' in metrics:
                    win_rate = metrics.get('win_rate', 0)
                    self.log_test("enhanced_backtest", "PASS", f"Backtest completed - Win rate: {win_rate:.1%}")
                    return True
                else:
                    self.log_test("enhanced_backtest", "FAIL", "Backtest returned invalid metrics")
                    return False
            else:
                self.log_test("enhanced_backtest", "SKIP", "No data available for backtesting")
                return False
                
        except Exception as e:
            self.log_test("enhanced_backtest", "FAIL", "Enhanced backtest error", e)
            return False
    
    def test_system_status(self):
        """Test Option 9: System Status"""
        try:
            print(f"\nTesting system status check...")
            
            # Count files
            data_files = len([f for f in os.listdir('data') if f.endswith('.csv')]) if os.path.exists('data') else 0
            model_files = len([f for f in os.listdir('models') if f.endswith('.pkl')]) if os.path.exists('models') else 0
            log_files = len([f for f in os.listdir('logs') if f.endswith('.json')]) if os.path.exists('logs') else 0
            
            # Test MT5 connection
            mt5_status = "UNKNOWN"
            try:
                self.trading_system.data_processor.mt5.connect()
                mt5_status = "OK"
            except:
                mt5_status = "FAILED"
            
            status_msg = f"Data: {data_files}, Models: {model_files}, Logs: {log_files}, MT5: {mt5_status}"
            self.log_test("system_status", "PASS", status_msg)
            return True
            
        except Exception as e:
            self.log_test("system_status", "FAIL", "System status check failed", e)
            return False
    
    def test_daily_optimization(self):
        """Test Option 10: Daily Backtest Optimization"""
        try:
            print(f"\nTesting daily optimization (limited test)...")
            
            from daily_backtest_optimizer import DailyBacktestOptimizer
            
            # Check if we have pre-trained models first
            model_files = [f for f in os.listdir('models') if f.startswith(f'{self.symbol}_pretrained_ensemble_')] if os.path.exists('models') else []
            
            if not model_files:
                self.log_test("daily_optimization", "SKIP", "No pre-trained models available")
                return False
            
            optimizer = DailyBacktestOptimizer(self.symbol)
            
            # Test just the data collection part
            daily_data = optimizer.get_recent_week_data()
            
            if daily_data and len(daily_data) > 0:
                self.log_test("daily_optimization", "PASS", f"Daily optimization setup works - {len(daily_data)} days available")
                return True
            else:
                self.log_test("daily_optimization", "FAIL", "No recent data available for optimization")
                return False
                
        except Exception as e:
            self.log_test("daily_optimization", "FAIL", "Daily optimization test failed", e)
            return False
    
    def test_adaptive_trader_setup(self):
        """Test Option 11: Adaptive Live Trading Setup"""
        try:
            print(f"\nTesting adaptive trader setup...")
            
            from adaptive_live_trader import AdaptiveLiveTrader
            
            trader = AdaptiveLiveTrader(self.symbol)
            
            # Test loading configuration
            config_loaded = trader.load_adaptive_config()
            
            if config_loaded:
                self.log_test("adaptive_trader_setup", "PASS", "Adaptive trader setup successful")
                return True
            else:
                self.log_test("adaptive_trader_setup", "FAIL", "Adaptive trader setup failed")
                return False
                
        except Exception as e:
            self.log_test("adaptive_trader_setup", "FAIL", "Adaptive trader setup error", e)
            return False
    
    def run_all_tests(self):
        """Run all individual function tests"""
        print("=" * 70)
        print("INDIVIDUAL FUNCTION TESTING")
        print("=" * 70)
        print(f"Testing symbol: {self.symbol}")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        # Test functions in order
        tests = [
            ("Setup Directories", self.test_setup_directories),
            ("Download Data", self.test_download_data),
            ("Correlation Analysis", self.test_correlation_analysis),
            ("Enhanced Model Training", self.test_enhanced_model_training),
            ("Multi-Timeframe Training", self.test_multi_timeframe_training),
            ("Enhanced Backtest", self.test_enhanced_backtest),
            ("System Status", self.test_system_status),
            ("Daily Optimization", self.test_daily_optimization),
            ("Adaptive Trader Setup", self.test_adaptive_trader_setup)
        ]
        
        for test_name, test_func in tests:
            print(f"\n--- Testing {test_name} ---")
            try:
                test_func()
            except Exception as e:
                self.log_test(test_name, "FAIL", "Unexpected error", e)
        
        # Print summary
        self.print_test_summary()
    
    def print_test_summary(self):
        """Print test summary"""
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        passed = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed = len([r for r in self.test_results if r['status'] == 'FAIL'])
        skipped = len([r for r in self.test_results if r['status'] == 'SKIP'])
        
        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        
        print(f"\nDetailed Results:")
        for result in self.test_results:
            status_icon = "✓" if result['status'] == "PASS" else "✗" if result['status'] == "FAIL" else "⚠"
            print(f"  {status_icon} {result['function']}: {result['message']}")
            if result['error']:
                print(f"    Error: {result['error']}")
        
        # Overall status
        if failed == 0:
            print(f"\n🚀 ALL TESTS PASSED! System is ready for use.")
        elif passed > failed:
            print(f"\n⚠ MOSTLY WORKING - {failed} issues need attention.")
        else:
            print(f"\n❌ SYSTEM NEEDS FIXES - {failed} critical issues found.")

def main():
    """Main testing function"""
    print("Individual Function Tester for Enhanced MT5 AI Trading System")
    
    symbol = input("Enter symbol to test (default XAUUSD): ").strip() or "XAUUSD"
    
    tester = FunctionTester()
    tester.symbol = symbol
    
    print(f"\nStarting individual function tests for {symbol}...")
    tester.run_all_tests()

if __name__ == "__main__":
    main()