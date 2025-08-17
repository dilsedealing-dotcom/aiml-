import pandas as pd
import numpy as np
from data_processor import DataProcessor
from signal_generator import SignalGenerator
from onnx_model import ONNXPricePredictor
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, initial_balance=100, risk_percentage=0.01):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_percentage = risk_percentage
        self.trades = []
        self.equity_curve = []
        self.df = None

    def run_backtest(self, df, signal_generator, start_date=None, end_date=None):
        if start_date:
            df = df[df['time'] >= start_date]
        if end_date:
            df = df[df['time'] <= end_date]
        
        self.df = df

        signals = signal_generator.generate_signals(df)

        for i in range(len(df)):
            current_bar = df.iloc[i]
            for signal in signals:
                if signal['timestamp'] == current_bar['time']:
                    if signal['action'] != 'HOLD':
                        self.execute_trade(signal, current_bar)
            
            self.equity_curve.append(self.balance)

        return self.calculate_performance_metrics()

    def execute_trade(self, signal, current_bar):
        entry_price = float(current_bar['close'])
        lot_size = float(signal['lot_size'])
        
        # Simulate trade execution
        if signal['action'] == 'BUY':
            exit_price = float(signal['tp']) if signal['tp'] else entry_price * 1.01
            sl_price = float(signal['sl']) if signal['sl'] else entry_price * 0.99
        else:  # SELL
            exit_price = float(signal['tp']) if signal['tp'] else entry_price * 0.99
            sl_price = float(signal['sl']) if signal['sl'] else entry_price * 1.01
        
        # Calculate P&L
        if signal['action'] == 'BUY':
            pnl = (exit_price - entry_price) * lot_size * 100000
        else:
            pnl = (entry_price - exit_price) * lot_size * 100000
        
        self.balance += pnl
        
        trade_record = {
            'timestamp': signal['timestamp'],
            'action': signal['action'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'lot_size': lot_size,
            'pnl': pnl,
            'balance': self.balance,
            'confidence': float(signal['confidence'])
        }
        
        self.trades.append(trade_record)

    def calculate_performance_metrics(self):
        if not self.trades:
            return {}
        
        trades_df = pd.DataFrame(self.trades)
        
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        
        returns = trades_df['pnl'] / self.initial_balance
        sharpe_ratio = returns.mean() / returns.std() if returns.std() > 0 else 0.0
        
        max_drawdown = self.calculate_max_drawdown()
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'final_balance': self.balance
        }
        
        return metrics

    def calculate_max_drawdown(self):
        if not self.equity_curve:
            return 0
        
        peak = self.equity_curve[0]
        max_dd = 0
        
        for value in self.equity_curve:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak * 100
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd

    def plot_results(self):
        if not self.trades or self.df is None:
            print("No trades to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price and Bollinger Bands
        ax1.plot(self.df['time'], self.df['close'], label='Close Price')
        ax1.plot(self.df['time'], self.df['bb_upper'], label='Upper BB', linestyle='--')
        ax1.plot(self.df['time'], self.df['bb_middle'], label='Middle BB', linestyle='--')
        ax1.plot(self.df['time'], self.df['bb_lower'], label='Lower BB', linestyle='--')
        
        # Plot trades
        trades_df = pd.DataFrame(self.trades)
        buy_signals = trades_df[trades_df['action'] == 'BUY']
        sell_signals = trades_df[trades_df['action'] == 'SELL']
        
        ax1.plot(buy_signals['timestamp'], buy_signals['entry_price'], '^g', markersize=10, label='Buy Signal')
        ax1.plot(sell_signals['timestamp'], sell_signals['entry_price'], 'vr', markersize=10, label='Sell Signal')
        
        ax1.set_title('Price Chart with Trades and Bollinger Bands')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # Equity curve
        ax2.plot(self.equity_curve)
        ax2.set_title('Equity Curve')
        ax2.set_ylabel('Balance')
        ax2.set_xlabel('Trade Number')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('data/backtest_results_with_trades.png')
        plt.show()

if __name__ == "__main__":
    data_processor = DataProcessor()
    df = data_processor.download_and_process_data(symbol="XAUUSD", count=50000)
    
    if df is not None:
        # Train the model
        predictor = ONNXPricePredictor()
        predictor.train_model(df)

        # Initialize Signal Generator with the trained predictor
        signal_generator = SignalGenerator(predictor)

        backtester = Backtester(initial_balance=100, risk_percentage=0.01)
        metrics = backtester.run_backtest(df, signal_generator, start_date='2025-01-01', end_date='2025-08-16')
        
        print("Backtest Results:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        backtester.plot_results()

        # Correlation Analysis
        print("\nFeature Correlation Analysis:")
        data_processor.analyze_correlations(df)
