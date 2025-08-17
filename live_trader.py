import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
from correlation_analyzer import CorrelationAnalyzer
from data_processor import DataProcessor
import pickle
import os

class LiveTrader:
    def __init__(self, symbol="XAUUSD", risk_pct=0.02):
        self.symbol = symbol
        self.risk_pct = risk_pct
        self.data_processor = DataProcessor()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.balance = 10000
        self.positions = []
        self.signals_log = []
        
    def load_trained_model(self):
        """Load latest trained model"""
        model_files = [f for f in os.listdir('models') if f.startswith(f'{self.symbol}_enhanced_')]
        if not model_files:
            print("No trained model found. Training new model...")
            return self.train_model()
        
        latest_model = sorted(model_files)[-1]
        print(f"Loading model: {latest_model}")
        
        with open(f'models/{latest_model}', 'rb') as f:
            self.correlation_analyzer.bb_model = pickle.load(f)
        
        return True
    
    def train_model(self):
        """Train model with latest data"""
        # Load training data
        data_files = [f for f in os.listdir('data') if f.startswith(f'{self.symbol}_') and f.endswith('.csv')]
        if not data_files:
            print("No training data found. Please collect data first.")
            return False
        
        latest_data = sorted(data_files)[-1]
        df = pd.read_csv(f'data/{latest_data}')
        df['time'] = pd.to_datetime(df['time'])
        
        print(f"Training model with {len(df)} bars...")
        
        # Train correlation analyzer
        self.correlation_analyzer.analyze_bb_correlations(df)
        model = self.correlation_analyzer.train_bb_enhancement_model(df)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_file = f'models/{self.symbol}_enhanced_model_{timestamp}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"Model trained and saved to {model_file}")
        return True
    
    def get_live_data(self, count=1000):
        """Get latest live data"""
        return self.data_processor.download_and_process_data(self.symbol, count=count)
    
    def generate_signals(self, df):
        """Generate trading signals"""
        if self.correlation_analyzer.bb_model is None:
            print("No model loaded")
            return []
        
        return self.correlation_analyzer.generate_enhanced_trading_signals(df, threshold=0.7)
    
    def execute_signal(self, signal):
        """Execute trading signal (simulation)"""
        entry_price = signal['price']
        lot_size = signal['lot_size']
        action = signal['action']
        confidence = signal['confidence']
        
        # Calculate position size based on risk
        risk_amount = self.balance * self.risk_pct
        
        # Simulate position
        position = {
            'id': len(self.positions) + 1,
            'symbol': self.symbol,
            'action': action,
            'entry_price': entry_price,
            'lot_size': lot_size,
            'entry_time': datetime.now(),
            'confidence': confidence,
            'tp': signal.get('tp'),
            'sl': signal.get('sl'),
            'status': 'open'
        }
        
        self.positions.append(position)
        
        # Log signal
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'signal': signal,
            'position_id': position['id'],
            'balance': self.balance
        }
        
        self.signals_log.append(log_entry)
        
        # Save to file
        log_file = f'logs/live_trading_{datetime.now().strftime("%Y%m%d")}.json'
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] EXECUTED: {action} {self.symbol} @ {entry_price:.5f} (Conf: {confidence:.3f})")
        
        return position
    
    def check_positions(self, current_price):
        """Check and close positions based on TP/SL"""
        for position in self.positions:
            if position['status'] != 'open':
                continue
            
            entry_price = position['entry_price']
            action = position['action']
            tp = position.get('tp')
            sl = position.get('sl')
            
            should_close = False
            close_reason = ""
            
            if action == 'BUY':
                if tp and current_price >= tp:
                    should_close = True
                    close_reason = "TP"
                elif sl and current_price <= sl:
                    should_close = True
                    close_reason = "SL"
            else:  # SELL
                if tp and current_price <= tp:
                    should_close = True
                    close_reason = "TP"
                elif sl and current_price >= sl:
                    should_close = True
                    close_reason = "SL"
            
            if should_close:
                self.close_position(position, current_price, close_reason)
    
    def close_position(self, position, close_price, reason):
        """Close position and calculate P&L"""
        entry_price = position['entry_price']
        lot_size = position['lot_size']
        action = position['action']
        
        # Calculate P&L
        if action == 'BUY':
            pnl = (close_price - entry_price) * lot_size * 100000
        else:
            pnl = (entry_price - close_price) * lot_size * 100000
        
        self.balance += pnl
        
        position['status'] = 'closed'
        position['close_price'] = close_price
        position['close_time'] = datetime.now()
        position['pnl'] = pnl
        position['close_reason'] = reason
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] CLOSED: {action} {self.symbol} @ {close_price:.5f} | P&L: {pnl:.2f} | Reason: {reason}")
    
    def run_live_trading(self, check_interval=300):  # 5 minutes
        """Run live trading loop"""
        print(f"Starting live trading for {self.symbol}")
        print(f"Initial balance: ${self.balance}")
        print(f"Risk per trade: {self.risk_pct*100}%")
        print("=" * 50)
        
        # Load model
        if not self.load_trained_model():
            print("Failed to load model. Exiting.")
            return
        
        last_signal_time = None
        
        try:
            while True:
                # Get latest data
                df = self.get_live_data(count=1000)
                if df is None:
                    print("Failed to get live data")
                    time.sleep(60)
                    continue
                
                current_price = df['close'].iloc[-1]
                current_time = df['time'].iloc[-1]
                
                # Check existing positions
                self.check_positions(current_price)
                
                # Generate new signals (avoid duplicate signals)
                if last_signal_time is None or current_time > last_signal_time:
                    signals = self.generate_signals(df)
                    
                    # Process latest signals
                    for signal in signals[-3:]:  # Last 3 signals
                        signal_time = pd.to_datetime(signal['timestamp'])
                        
                        if last_signal_time is None or signal_time > last_signal_time:
                            if signal['action'] != 'HOLD':
                                # Check if we already have open positions
                                open_positions = [p for p in self.positions if p['status'] == 'open']
                                
                                if len(open_positions) < 3:  # Max 3 open positions
                                    self.execute_signal(signal)
                                    last_signal_time = signal_time
                
                # Print status
                open_positions = len([p for p in self.positions if p['status'] == 'open'])
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Price: {current_price:.5f} | Balance: ${self.balance:.2f} | Open: {open_positions}")
                
                # Wait for next check
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nLive trading stopped by user")
            self.print_summary()
    
    def print_summary(self):
        """Print trading summary"""
        print("\n" + "=" * 50)
        print("LIVE TRADING SUMMARY")
        print("=" * 50)
        
        total_trades = len(self.positions)
        closed_trades = [p for p in self.positions if p['status'] == 'closed']
        winning_trades = [p for p in closed_trades if p.get('pnl', 0) > 0]
        
        print(f"Total Trades: {total_trades}")
        print(f"Closed Trades: {len(closed_trades)}")
        print(f"Winning Trades: {len(winning_trades)}")
        print(f"Win Rate: {len(winning_trades)/len(closed_trades)*100:.1f}%" if closed_trades else "N/A")
        print(f"Final Balance: ${self.balance:.2f}")
        print(f"Total P&L: ${self.balance - 10000:.2f}")
        
        if closed_trades:
            total_pnl = sum(p.get('pnl', 0) for p in closed_trades)
            print(f"Realized P&L: ${total_pnl:.2f}")

def main():
    """Main function for live trading"""
    print("Live Trading System")
    print("=" * 30)
    
    symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
    risk_pct = float(input("Enter risk percentage (default 0.02): ") or 0.02)
    
    trader = LiveTrader(symbol=symbol, risk_pct=risk_pct)
    trader.run_live_trading()

if __name__ == "__main__":
    main()