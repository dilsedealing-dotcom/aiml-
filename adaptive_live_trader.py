import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from memory_enhanced_trader import MemoryEnhancedTrader
import os

class AdaptiveLiveTrader(MemoryEnhancedTrader):
    def __init__(self, symbol="XAUUSD"):
        super().__init__(symbol)
        self.adaptive_config = self.load_adaptive_config()
        self.daily_performance = []
        
    def load_adaptive_config(self):
        """Load adaptive configuration from optimization"""
        config_file = f'models/{self.symbol}_adaptive_config.json'
        
        # Default config
        default_config = {
            'optimal_entry_threshold': 0.7,
            'optimal_exit_threshold': 0.3,
            'confidence_adjustment': 1.0
        }
        
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    content = f.read().strip()
                    if content:
                        config = json.loads(content)
                        # Validate required keys
                        if all(key in config for key in ['optimal_entry_threshold', 'optimal_exit_threshold']):
                            print(f"Loaded adaptive config: Entry={config['optimal_entry_threshold']}, Exit={config['optimal_exit_threshold']}")
                            return config
                        else:
                            print("Invalid config format, using defaults")
                    else:
                        print("Empty config file, using defaults")
            except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
                print(f"Error loading config: {e}, using defaults")
                # Remove corrupted file
                try:
                    os.remove(config_file)
                except:
                    pass
        
        # Save default config
        try:
            os.makedirs('models', exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(default_config, f, indent=4)
        except Exception as e:
            print(f"Error saving default config: {e}")
        
        return default_config
    
    def generate_adaptive_signals(self, df):
        """Generate signals with adaptive thresholds"""
        # Get base signals from memory-enhanced trader
        base_signals = self.generate_memory_enhanced_signals(df)
        
        adaptive_signals = []
        
        for signal in base_signals:
            # Apply adaptive thresholds
            entry_threshold = self.adaptive_config['optimal_entry_threshold']
            confidence_adjustment = self.adaptive_config.get('confidence_adjustment', 1.0)
            
            # Adjust confidence based on recent performance
            adjusted_confidence = signal['confidence'] * confidence_adjustment
            
            # Only keep signals above adaptive threshold
            if adjusted_confidence >= entry_threshold:
                signal['confidence'] = adjusted_confidence
                signal['adaptive_threshold'] = entry_threshold
                signal['original_confidence'] = signal['confidence'] / confidence_adjustment
                adaptive_signals.append(signal)
        
        return adaptive_signals
    
    def should_exit_position(self, position, current_price, current_confidence):
        """Determine if position should be exited using adaptive logic"""
        exit_threshold = self.adaptive_config['optimal_exit_threshold']
        
        # Exit conditions
        exit_reasons = []
        
        # 1. Confidence-based exit
        if current_confidence < exit_threshold:
            exit_reasons.append("Low Confidence")
        
        # 2. Adaptive profit taking
        entry_price = position['entry_price']
        
        if position['action'] == 'BUY':
            profit_pct = (current_price - entry_price) / entry_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
        
        # Dynamic profit target based on confidence
        profit_target = position['confidence'] * 0.01  # 1% per confidence point
        
        if profit_pct >= profit_target:
            exit_reasons.append("Adaptive Profit Target")
        
        # 3. Adaptive stop loss
        stop_loss_pct = (1 - position['confidence']) * 0.005  # Tighter SL for low confidence
        
        if profit_pct <= -stop_loss_pct:
            exit_reasons.append("Adaptive Stop Loss")
        
        return len(exit_reasons) > 0, exit_reasons
    
    def track_daily_performance(self):
        """Track daily performance for continuous adaptation"""
        try:
            today = datetime.now().date()
            
            # Get today's trades
            today_trades = [t for t in self.trade_history 
                           if pd.to_datetime(t['timestamp']).date() == today]
            
            if today_trades:
                df = pd.DataFrame(today_trades)
                
                daily_perf = {
                    'date': str(today),
                    'trades': len(df),
                    'win_rate': (df['pnl'] > 0).mean() if 'pnl' in df.columns else 0,
                    'total_pnl': df['pnl'].sum() if 'pnl' in df.columns else 0,
                    'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else 0,
                    'config_used': self.adaptive_config.copy()
                }
                
                # Update daily performance
                self.daily_performance.append(daily_perf)
                
                # Save daily performance
                try:
                    os.makedirs('logs', exist_ok=True)
                    perf_file = f'logs/daily_performance_{self.symbol}.json'
                    with open(perf_file, 'w') as f:
                        json.dump(self.daily_performance, f, indent=2)
                except Exception as e:
                    print(f"Error saving daily performance: {e}")
        except Exception as e:
            print(f"Error tracking daily performance: {e}")
    
    def adapt_parameters(self):
        """Adapt parameters based on recent performance"""
        if len(self.daily_performance) < 3:
            return  # Need at least 3 days of data
        
        recent_perf = self.daily_performance[-3:]  # Last 3 days
        
        avg_win_rate = np.mean([p['win_rate'] for p in recent_perf])
        avg_pnl = np.mean([p['total_pnl'] for p in recent_perf])
        
        # Adapt entry threshold
        if avg_win_rate < 0.5:  # Poor performance
            self.adaptive_config['optimal_entry_threshold'] += 0.05  # Be more selective
        elif avg_win_rate > 0.7:  # Good performance
            self.adaptive_config['optimal_entry_threshold'] -= 0.02  # Be less selective
        
        # Adapt exit threshold
        if avg_pnl < 0:  # Losing money
            self.adaptive_config['optimal_exit_threshold'] += 0.05  # Exit faster
        elif avg_pnl > 100:  # Making good money
            self.adaptive_config['optimal_exit_threshold'] -= 0.02  # Hold longer
        
        # Keep thresholds in reasonable range
        self.adaptive_config['optimal_entry_threshold'] = np.clip(
            self.adaptive_config['optimal_entry_threshold'], 0.5, 0.9)
        self.adaptive_config['optimal_exit_threshold'] = np.clip(
            self.adaptive_config['optimal_exit_threshold'], 0.1, 0.5)
        
        # Save updated config
        try:
            os.makedirs('models', exist_ok=True)
            config_file = f'models/{self.symbol}_adaptive_config.json'
            self.adaptive_config['last_updated'] = datetime.now().isoformat()
            
            with open(config_file, 'w') as f:
                json.dump(self.adaptive_config, f, indent=4)
        except Exception as e:
            print(f"Error saving adaptive config: {e}")
        
        print(f"Parameters adapted: Entry={self.adaptive_config['optimal_entry_threshold']:.2f}, Exit={self.adaptive_config['optimal_exit_threshold']:.2f}")
    
    def run_adaptive_trading(self, check_interval=300):
        """Run adaptive live trading"""
        print(f"Starting adaptive trading for {self.symbol}")
        print(f"Entry threshold: {self.adaptive_config['optimal_entry_threshold']}")
        print(f"Exit threshold: {self.adaptive_config['optimal_exit_threshold']}")
        print("=" * 50)
        
        # Load models and memory
        if not self.load_pretrained_models():
            print("No pre-trained models found")
            return
        
        self.load_trade_memory()
        
        last_adaptation = datetime.now()
        
        try:
            while True:
                # Get latest data
                df = self.get_live_data(count=1000)
                if df is None:
                    continue
                
                current_price = df['close'].iloc[-1]
                
                # Generate adaptive signals
                signals = self.generate_adaptive_signals(df)
                
                # Process signals
                for signal in signals:
                    # Check position limits
                    open_positions = [p for p in self.positions if p['status'] == 'open']
                    
                    if len(open_positions) < 3:  # Max 3 positions
                        position = self.execute_signal(signal)
                        
                        # Log adaptive signal
                        try:
                            log_entry = {
                                'timestamp': datetime.now().isoformat(),
                                'signal': signal,
                                'adaptive_config': self.adaptive_config,
                                'type': 'adaptive_entry'
                            }
                            
                            os.makedirs('logs', exist_ok=True)
                            log_file = f'logs/adaptive_trading_{datetime.now().strftime("%Y%m%d")}.json'
                            with open(log_file, 'a') as f:
                                f.write(json.dumps(log_entry, default=str) + '\n')
                        except Exception as e:
                            print(f"Error logging signal: {e}")
                
                # Check exits with adaptive logic
                for position in [p for p in self.positions if p['status'] == 'open']:
                    # Get current confidence (simplified)
                    current_confidence = position['confidence']
                    
                    should_exit, reasons = self.should_exit_position(
                        position, current_price, current_confidence)
                    
                    if should_exit:
                        self.close_position(position, current_price, ", ".join(reasons))
                
                # Track daily performance
                self.track_daily_performance()
                
                # Adapt parameters daily
                if (datetime.now() - last_adaptation).days >= 1:
                    self.adapt_parameters()
                    last_adaptation = datetime.now()
                
                # Status update
                open_pos = len([p for p in self.positions if p['status'] == 'open'])
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Price: {current_price:.5f} | Balance: ${self.balance:.2f} | Open: {open_pos} | Entry TH: {self.adaptive_config['optimal_entry_threshold']:.2f}")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nAdaptive trading stopped")
            self.print_adaptive_summary()
    
    def print_adaptive_summary(self):
        """Print adaptive trading summary"""
        print("\n" + "=" * 60)
        print("ADAPTIVE TRADING SUMMARY")
        print("=" * 60)
        
        # Basic summary
        self.print_summary()
        
        # Adaptive metrics
        if self.daily_performance:
            df = pd.DataFrame(self.daily_performance)
            
            print(f"\nAdaptive Performance:")
            print(f"Days traded: {len(df)}")
            print(f"Average daily win rate: {df['win_rate'].mean():.2%}")
            print(f"Average daily P&L: ${df['total_pnl'].mean():.2f}")
            print(f"Best day P&L: ${df['total_pnl'].max():.2f}")
            print(f"Worst day P&L: ${df['total_pnl'].min():.2f}")
        
        print(f"\nFinal Adaptive Config:")
        print(f"Entry threshold: {self.adaptive_config['optimal_entry_threshold']:.3f}")
        print(f"Exit threshold: {self.adaptive_config['optimal_exit_threshold']:.3f}")

def main():
    """Run adaptive live trading"""
    symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
    
    trader = AdaptiveLiveTrader(symbol)
    trader.run_adaptive_trading()

if __name__ == "__main__":
    main()