import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_processor import DataProcessor
    from onnx_model import ONNXPricePredictor
    from signal_generator import SignalGenerator
    from backtest import Backtester
    from technical_indicators import TechnicalIndicators
    from enhanced_backtester import EnhancedBacktester
    from correlation_analyzer import CorrelationAnalyzer
    from advanced_ml_trainer import AdvancedMLTrainer
    from memory_enhanced_trader import MemoryEnhancedTrader
    from daily_backtest_optimizer import DailyBacktestOptimizer
    from adaptive_live_trader import AdaptiveLiveTrader
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all module files are in the same directory")
    sys.exit(1)

import schedule
import time
import threading

class TradingSystem:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.predictor = ONNXPricePredictor()
        self.signal_generator = SignalGenerator(self.predictor)
        self.backtester = Backtester()
        self.enhanced_backtester = EnhancedBacktester()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.ml_trainer = AdvancedMLTrainer()
        self.memory_trader = None
        self.config = self.load_config()
        
    def setup_directories(self):
        """Ensure all required directories exist"""
        directories = ['data', 'onnx_models', 'ea', 'models', 'logs', 'config']
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")
    
    def load_config(self):
        """Load or create configuration"""
        config_path = 'config/trading_config.json'
        default_config = {
            'symbols': ['XAUUSD', 'EURUSD', 'GBPUSD'],
            'timeframes': ['M5', 'M15', 'H1'],
            'data_count': 50000,
            'retrain_interval': 24,  # hours
            'risk_percentage': 0.02,
            'initial_balance': 10000
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Ensure config directory exists
            os.makedirs('config', exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=4)
            return default_config
    
    def download_and_store_data(self, symbol="XAUUSD", count=50000):
        """Download and store real data for training"""
        print(f"Downloading {symbol} data from MT5...")
        
        df = self.data_processor.download_and_process_data(symbol, count=count)
        
        if df is None:
            print("Failed to download data. Please check MT5 connection.")
            return None
        
        # Store raw and processed data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        processed_file = f'data/{symbol}_processed_{timestamp}.csv'
        
        df.to_csv(processed_file, index=False)
        print(f"Stored {len(df)} bars to {processed_file}")
        
        return df
    
    def run_correlation_analysis(self, symbol="XAUUSD"):
        """Run comprehensive correlation analysis"""
        print("Running correlation analysis...")
        
        # Load latest data
        df = self.load_latest_data(symbol)
        if df is None:
            print("No data found. Please download data first.")
            return
        
        # Run enhanced analysis
        metrics, correlations = self.enhanced_backtester.run_enhanced_backtest(df, start_date='2025-01-01')
        
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'metrics': metrics,
            'correlations': correlations.to_dict() if correlations is not None else None
        }
        
        results_file = f'data/{symbol}_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4, default=str)
        
        print(f"Analysis results saved to {results_file}")
        return results
    
    def train_enhanced_model(self, symbol="XAUUSD"):
        """Train enhanced ML model with correlation features"""
        print(f"Training enhanced model for {symbol}...")
        
        df = self.load_latest_data(symbol)
        if df is None:
            print("No data found. Please download data first.")
            return None
        
        # Train correlation analyzer
        correlations = self.correlation_analyzer.analyze_bb_correlations(df)
        model = self.correlation_analyzer.train_bb_enhancement_model(df)
        
        # Save model
        model_file = f'models/{symbol}_enhanced_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl'
        import pickle
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Enhanced model saved to {model_file}")
        return model
    
    def load_latest_data(self, symbol):
        """Load latest processed data for symbol"""
        data_files = [f for f in os.listdir('data') if f.startswith(f'{symbol}_processed_')]
        if not data_files:
            return None
        
        latest_file = sorted(data_files)[-1]
        df = pd.read_csv(f'data/{latest_file}')
        df['time'] = pd.to_datetime(df['time'])
        return df
    
    def run_memory_enhanced_trading(self, symbol="XAUUSD"):
        """Run live trading with memory-enhanced ML models"""
        print(f"Starting memory-enhanced trading for {symbol}...")
        
        # Initialize memory trader
        self.memory_trader = MemoryEnhancedTrader(symbol)
        
        # Load pre-trained models and memory
        if not self.memory_trader.load_pretrained_models():
            print("No pre-trained models found. Please run pre-training first.")
            return
        
        self.memory_trader.load_trade_memory()
        
        def trading_loop():
            try:
                # Get latest data
                df = self.data_processor.download_and_process_data(symbol, count=2000)
                if df is None:
                    return
                
                # Generate memory-enhanced signals
                signals = self.memory_trader.generate_memory_enhanced_signals(df)
                
                # Process signals
                for signal in signals:
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'signal': signal,
                        'symbol': symbol,
                        'memory_enhanced': True
                    }
                    
                    # Log to file
                    log_file = f'logs/memory_signals_{datetime.now().strftime("%Y%m%d")}.json'
                    with open(log_file, 'a') as f:
                        f.write(json.dumps(log_entry, default=str) + '\n')
                    
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] MEMORY-ENHANCED: {signal['action']} {symbol} @ {signal['price']:.5f} (Conf: {signal['confidence']:.3f})")
                    
                    # Simulate trade execution and record result
                    if signal['action'] != 'HOLD':
                        # Simulate exit after some time (for demo)
                        exit_price = signal['price'] * (1 + np.random.normal(0, 0.001))
                        exit_time = datetime.now()
                        
                        self.memory_trader.record_trade_result(signal, exit_price, exit_time)
                
            except Exception as e:
                print(f"Error in memory trading loop: {e}")
        
        # Schedule trading loop every 5 minutes
        schedule.every(5).minutes.do(trading_loop)
        
        print("Memory-enhanced trading system is running. Press Ctrl+C to stop.")
        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            print("Memory-enhanced trading stopped.")

def main():
    print("Enhanced MT5 AI Trading System with Daily Optimization")
    print("=" * 60)
    
    trading_system = TradingSystem()
    trading_system.setup_directories()
    
    while True:
        print("\n" + "=" * 60)
        print("MAIN MENU")
        print("=" * 60)
        print("1. Download & Store Real Data")
        print("2. Run Correlation Analysis")
        print("3. Train Enhanced ML Model")
        print("4. Pre-Train Multi-Timeframe Models")
        print("5. Run Enhanced Backtest")
        print("6. Start Memory-Enhanced Live Trading")
        print("7. Multi-Symbol Analysis")
        print("8. View Trading Logs")
        print("9. System Status")
        print("10. Daily Backtest Optimization")
        print("11. Start Adaptive Live Trading")
        print("12. AUTO START - Complete Training & Trading")
        print("13. Exit")
        
        choice = input("\nEnter your choice (1-13): ")
        
        if choice == '1':
            symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
            count = int(input("Enter data count (default 50000): ") or 50000)
            trading_system.download_and_store_data(symbol, count)
        
        elif choice == '2':
            symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
            trading_system.run_correlation_analysis(symbol)
        
        elif choice == '3':
            symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
            trading_system.train_enhanced_model(symbol)
        
        elif choice == '4':
            symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
            print(f"Starting pre-training for {symbol}...")
            
            # Collect multi-timeframe data
            multi_tf_data = trading_system.ml_trainer.collect_multi_timeframe_data(symbol, bars_per_tf=20000)
            
            if multi_tf_data:
                # Create enhanced features
                combined_data = trading_system.ml_trainer.create_enhanced_features(multi_tf_data)
                
                # Train ensemble models
                models = trading_system.ml_trainer.train_ensemble_models(combined_data)
                
                # Save models
                model_file = trading_system.ml_trainer.save_pretrained_models(symbol)
                
                print(f"Pre-training complete! Models saved to: {model_file}")
            else:
                print("Failed to collect multi-timeframe data")
        
        elif choice == '5':
            symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
            df = trading_system.load_latest_data(symbol)
            if df is not None:
                metrics, _ = trading_system.enhanced_backtester.run_enhanced_backtest(df)
                print("\nBacktest Results:")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print("No data found. Please download data first.")
        
        elif choice == '6':
            symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
            trading_system.run_memory_enhanced_trading(symbol)
        
        elif choice == '7':
            symbols = input("Enter symbols (comma-separated, default XAUUSD,EURUSD): ") or "XAUUSD,EURUSD"
            for symbol in symbols.split(','):
                symbol = symbol.strip()
                print(f"\nProcessing {symbol}...")
                trading_system.download_and_store_data(symbol, 20000)
                trading_system.run_correlation_analysis(symbol)
        
        elif choice == '8':
            log_files = [f for f in os.listdir('logs') if f.startswith('signals_') or f.startswith('memory_signals_')]
            if log_files:
                latest_log = sorted(log_files)[-1]
                print(f"\nLatest signals from {latest_log}:")
                with open(f'logs/{latest_log}', 'r') as f:
                    lines = f.readlines()[-10:]  # Last 10 signals
                    for line in lines:
                        try:
                            signal = json.loads(line)
                            action = signal.get('signal', {}).get('action', 'N/A')
                            symbol = signal.get('symbol', 'N/A')
                            enhanced = '[MEMORY]' if signal.get('memory_enhanced') else '[REGULAR]'
                            print(f"  {signal['timestamp']}: {enhanced} {action} {symbol}")
                        except:
                            continue
            else:
                print("No trading logs found.")
        
        elif choice == '9':
            print("\nSystem Status:")
            print(f"Data files: {len([f for f in os.listdir('data') if f.endswith('.csv')])}")
            print(f"Models: {len([f for f in os.listdir('models') if f.endswith('.pkl')])}")
            print(f"Log files: {len([f for f in os.listdir('logs') if f.endswith('.json')])}")
            
            # Check for pre-trained models
            pretrained = [f for f in os.listdir('models') if 'pretrained_ensemble' in f]
            print(f"Pre-trained models: {len(pretrained)}")
            
            # Check trade memory
            if os.path.exists('models/trade_memory.json'):
                with open('models/trade_memory.json', 'r') as f:
                    memory = json.load(f)
                print(f"Trade memory: {len(memory)} trades")
            else:
                print("Trade memory: None")
            
            # Check MT5 connection
            try:
                trading_system.data_processor.mt5.connect()
                print("MT5 Connection: OK")
            except:
                print("MT5 Connection: FAILED")
        
        elif choice == '10':
            symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
            print(f"Running daily backtest optimization for {symbol}...")
            
            optimizer = DailyBacktestOptimizer(symbol)
            results = optimizer.run_weekly_optimization()
            
            if results:
                print("\nOptimization Results:")
                patterns = results['patterns']
                print(f"Optimal entry threshold: {patterns['best_entry_threshold']}")
                print(f"Optimal exit threshold: {patterns['best_exit_threshold']}")
                print(f"Expected win rate: {patterns['avg_win_rate']:.2%}")
                print(f"Total P&L: ${patterns['total_pnl']:.2f}")
            else:
                print("Optimization failed")
        
        elif choice == '11':
            symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
            
            trader = AdaptiveLiveTrader(symbol)
            trader.run_adaptive_trading()
        
        elif choice == '12':
            from auto_start_sequence import AutoStartSequence
            
            symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
            
            print(f"\nStarting AUTO SEQUENCE for {symbol}...")
            print("This will run complete pipeline: Data → Training → Optimization → Trading")
            
            confirm = input("Continue? (y/n): ").lower().strip()
            if confirm == 'y':
                auto_start = AutoStartSequence(symbol)
                auto_start.run_complete_sequence()
            else:
                print("Auto sequence cancelled")
        
        elif choice == '13':
            print("Exiting Enhanced Trading System...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()