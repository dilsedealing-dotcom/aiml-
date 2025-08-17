#!/usr/bin/env python3
"""
Quick Function Test - Fast validation of core functions
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    
    try:
        from main import TradingSystem
        print("[OK] Main system imported")
        
        from data_processor import DataProcessor
        print("[OK] Data processor imported")
        
        from enhanced_backtester import EnhancedBacktester
        print("[OK] Enhanced backtester imported")
        
        from correlation_analyzer import CorrelationAnalyzer
        print("[OK] Correlation analyzer imported")
        
        from advanced_ml_trainer import AdvancedMLTrainer
        print("[OK] Advanced ML trainer imported")
        
        from memory_enhanced_trader import MemoryEnhancedTrader
        print("[OK] Memory enhanced trader imported")
        
        from daily_backtest_optimizer import DailyBacktestOptimizer
        print("[OK] Daily optimizer imported")
        
        from adaptive_live_trader import AdaptiveLiveTrader
        print("[OK] Adaptive trader imported")
        
        from auto_start_sequence import AutoStartSequence
        print("[OK] Auto start sequence imported")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return False

def test_system_initialization():
    """Test system initialization"""
    print("\nTesting system initialization...")
    
    try:
        from main import TradingSystem
        
        trading_system = TradingSystem()
        print("[OK] Trading system initialized")
        
        trading_system.setup_directories()
        print("[OK] Directories setup completed")
        
        # Check if directories exist
        required_dirs = ['data', 'models', 'logs', 'config']
        for dir_name in required_dirs:
            if os.path.exists(dir_name):
                print(f"[OK] {dir_name} directory exists")
            else:
                print(f"[ERROR] {dir_name} directory missing")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] System initialization failed: {e}")
        return False

def test_mt5_connection():
    """Test MT5 connection"""
    print("\nTesting MT5 connection...")
    
    try:
        from data_processor import DataProcessor
        
        data_processor = DataProcessor()
        data_processor.mt5.connect()
        print("[OK] MT5 connection successful")
        return True
        
    except Exception as e:
        print(f"[ERROR] MT5 connection failed: {e}")
        print("  Make sure MT5 terminal is running")
        return False

def test_basic_functionality():
    """Test basic functionality without heavy operations"""
    print("\nTesting basic functionality...")
    
    try:
        from main import TradingSystem
        
        trading_system = TradingSystem()
        
        # Test configuration loading
        config = trading_system.load_config()
        if config:
            print("[OK] Configuration loaded")
        else:
            print("[ERROR] Configuration loading failed")
        
        # Test data loading (if exists)
        try:
            df = trading_system.load_latest_data("XAUUSD")
            if df is not None:
                print(f"[OK] Found existing data: {len(df)} bars")
            else:
                print("[WARNING] No existing data found (normal for first run)")
        except:
            print("[WARNING] Data loading test skipped")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Basic functionality test failed: {e}")
        return False

def main():
    """Run quick tests"""
    print("=" * 50)
    print("QUICK FUNCTION TEST")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("System Initialization", test_system_initialization),
        ("MT5 Connection", test_mt5_connection),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"[ERROR] {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("QUICK TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("[SUCCESS] All quick tests passed! System ready for full testing.")
    elif passed >= total * 0.75:
        print("[WARNING] Most tests passed. Some issues may need attention.")
    else:
        print("[ERROR] Multiple issues found. Check errors above.")

if __name__ == "__main__":
    main()