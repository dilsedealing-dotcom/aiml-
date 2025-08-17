import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from data_processor import DataProcessor
from memory_enhanced_trader import MemoryEnhancedTrader
import os

class DailyBacktestOptimizer:
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol
        self.data_processor = DataProcessor()
        self.daily_results = []
        self.optimal_params = {}
        
    def get_recent_week_data(self):
        """Get last 7 days of 5-minute data"""
        # Calculate bars needed (7 days * 288 bars per day)
        bars_needed = 7 * 288
        
        df = self.data_processor.download_and_process_data(self.symbol, count=bars_needed)
        
        if df is None:
            return None
            
        # Split into daily chunks
        df['date'] = pd.to_datetime(df['time']).dt.date
        daily_data = {}
        
        for date in df['date'].unique()[-7:]:  # Last 7 days
            daily_df = df[df['date'] == date].copy()
            if len(daily_df) > 50:  # Minimum bars per day
                daily_data[str(date)] = daily_df
                
        return daily_data
    
    def backtest_single_day(self, daily_df, entry_threshold=0.7, exit_threshold=0.3):
        """Backtest single day with specific parameters"""
        trader = MemoryEnhancedTrader(self.symbol)
        
        # Load models if available
        if not trader.load_pretrained_models():
            return None
            
        trader.load_trade_memory()
        
        trades = []
        balance = 10000
        positions = []
        
        for i in range(len(daily_df)):
            current_bar = daily_df.iloc[i:i+50]  # Use 50 bars for context
            
            if len(current_bar) < 50:
                continue
                
            # Generate signals
            signals = trader.generate_memory_enhanced_signals(current_bar)
            
            # Process new signals
            for signal in signals:
                if signal['confidence'] >= entry_threshold:
                    position = {
                        'entry_time': signal['timestamp'],
                        'entry_price': signal['price'],
                        'action': signal['action'],
                        'confidence': signal['confidence'],
                        'tp': signal.get('tp'),
                        'sl': signal.get('sl'),
                        'lot_size': signal['lot_size']
                    }
                    positions.append(position)
            
            # Check exits
            current_price = daily_df.iloc[i]['close']
            
            for pos in positions[:]:
                should_exit = False
                exit_reason = ""
                
                # Confidence-based exit
                if pos['confidence'] < exit_threshold:
                    should_exit = True
                    exit_reason = "Low Confidence"
                
                # TP/SL exit
                elif pos['action'] == 'BUY':
                    if pos['tp'] and current_price >= pos['tp']:
                        should_exit = True
                        exit_reason = "TP"
                    elif pos['sl'] and current_price <= pos['sl']:
                        should_exit = True
                        exit_reason = "SL"
                elif pos['action'] == 'SELL':
                    if pos['tp'] and current_price <= pos['tp']:
                        should_exit = True
                        exit_reason = "TP"
                    elif pos['sl'] and current_price >= pos['sl']:
                        should_exit = True
                        exit_reason = "SL"
                
                if should_exit:
                    # Calculate P&L
                    if pos['action'] == 'BUY':
                        pnl = (current_price - pos['entry_price']) * pos['lot_size'] * 100000
                    else:
                        pnl = (pos['entry_price'] - current_price) * pos['lot_size'] * 100000
                    
                    balance += pnl
                    
                    trade = {
                        'entry_time': pos['entry_time'],
                        'exit_time': daily_df.iloc[i]['time'],
                        'action': pos['action'],
                        'entry_price': pos['entry_price'],
                        'exit_price': current_price,
                        'pnl': pnl,
                        'confidence': pos['confidence'],
                        'exit_reason': exit_reason
                    }
                    trades.append(trade)
                    positions.remove(pos)
        
        # Calculate daily metrics
        if trades:
            trades_df = pd.DataFrame(trades)
            win_rate = (trades_df['pnl'] > 0).mean()
            total_pnl = trades_df['pnl'].sum()
            avg_confidence = trades_df['confidence'].mean()
            
            return {
                'date': str(daily_df['date'].iloc[0]),
                'trades': len(trades),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'final_balance': balance,
                'avg_confidence': avg_confidence,
                'entry_threshold': entry_threshold,
                'exit_threshold': exit_threshold,
                'trades_data': trades
            }
        
        return None
    
    def optimize_daily_parameters(self, daily_data):
        """Optimize entry/exit parameters for each day"""
        entry_thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
        exit_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
        
        optimization_results = []
        
        for date, daily_df in daily_data.items():
            print(f"Optimizing parameters for {date}...")
            
            best_result = None
            best_score = -float('inf')
            
            for entry_th in entry_thresholds:
                for exit_th in exit_thresholds:
                    result = self.backtest_single_day(daily_df, entry_th, exit_th)
                    
                    if result:
                        # Score = win_rate * total_pnl * avg_confidence
                        score = result['win_rate'] * result['total_pnl'] * result['avg_confidence']
                        
                        if score > best_score:
                            best_score = score
                            best_result = result
                            best_result['score'] = score
            
            if best_result:
                optimization_results.append(best_result)
                print(f"  Best: Entry={best_result['entry_threshold']}, Exit={best_result['exit_threshold']}, Score={best_score:.2f}")
        
        return optimization_results
    
    def analyze_patterns(self, results):
        """Analyze patterns in optimal parameters"""
        if not results:
            return {}
        
        df = pd.DataFrame(results)
        
        patterns = {
            'avg_entry_threshold': df['entry_threshold'].mean(),
            'avg_exit_threshold': df['exit_threshold'].mean(),
            'best_entry_threshold': df.loc[df['score'].idxmax(), 'entry_threshold'],
            'best_exit_threshold': df.loc[df['score'].idxmax(), 'exit_threshold'],
            'total_trades': df['trades'].sum(),
            'avg_win_rate': df['win_rate'].mean(),
            'total_pnl': df['total_pnl'].sum(),
            'best_day': df.loc[df['score'].idxmax(), 'date'],
            'worst_day': df.loc[df['score'].idxmin(), 'date']
        }
        
        return patterns
    
    def update_adaptive_model(self, patterns):
        """Update model with learned patterns"""
        adaptive_config = {
            'timestamp': datetime.now().isoformat(),
            'symbol': self.symbol,
            'optimal_entry_threshold': patterns.get('best_entry_threshold', 0.7),
            'optimal_exit_threshold': patterns.get('best_exit_threshold', 0.3),
            'confidence_adjustment': patterns.get('avg_win_rate', 0.5),
            'performance_score': patterns.get('total_pnl', 0),
            'learning_data': patterns
        }
        
        # Save adaptive configuration
        config_file = f'models/{self.symbol}_adaptive_config.json'
        with open(config_file, 'w') as f:
            json.dump(adaptive_config, f, indent=4)
        
        print(f"Adaptive model updated: {config_file}")
        return adaptive_config
    
    def run_weekly_optimization(self):
        """Run complete weekly optimization"""
        print(f"Starting weekly optimization for {self.symbol}...")
        
        # Get recent week data
        daily_data = self.get_recent_week_data()
        
        if not daily_data:
            print("No recent data available")
            return None
        
        print(f"Analyzing {len(daily_data)} days of data...")
        
        # Optimize parameters for each day
        results = self.optimize_daily_parameters(daily_data)
        
        if not results:
            print("No optimization results")
            return None
        
        # Analyze patterns
        patterns = self.analyze_patterns(results)
        
        # Update adaptive model
        adaptive_config = self.update_adaptive_model(patterns)
        
        # Save detailed results
        results_file = f'data/{self.symbol}_weekly_optimization_{datetime.now().strftime("%Y%m%d")}.json'
        with open(results_file, 'w') as f:
            json.dump({
                'optimization_results': results,
                'patterns': patterns,
                'adaptive_config': adaptive_config
            }, f, indent=4, default=str)
        
        # Print summary
        print("\n" + "="*60)
        print("WEEKLY OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Days analyzed: {len(results)}")
        print(f"Total trades: {patterns['total_trades']}")
        print(f"Average win rate: {patterns['avg_win_rate']:.2%}")
        print(f"Total P&L: ${patterns['total_pnl']:.2f}")
        print(f"Optimal entry threshold: {patterns['best_entry_threshold']}")
        print(f"Optimal exit threshold: {patterns['best_exit_threshold']}")
        print(f"Best performing day: {patterns['best_day']}")
        
        return {
            'results': results,
            'patterns': patterns,
            'config': adaptive_config
        }

def main():
    """Run daily backtest optimization"""
    symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
    
    optimizer = DailyBacktestOptimizer(symbol)
    results = optimizer.run_weekly_optimization()
    
    if results:
        print(f"\nOptimization complete! Results saved.")
        print(f"Use the adaptive config for improved live trading.")

if __name__ == "__main__":
    main()