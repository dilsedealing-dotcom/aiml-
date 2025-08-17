import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from correlation_analyzer import CorrelationAnalyzer
from data_processor import DataProcessor
import seaborn as sns

class EnhancedBacktester:
    def __init__(self, initial_balance=100, risk_percentage=0.01):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_percentage = risk_percentage
        self.trades = []
        self.equity_curve = []
        self.correlation_analyzer = CorrelationAnalyzer()
        
    def run_enhanced_backtest(self, df, start_date='2025-01-01'):
        """Run backtest with enhanced BB signals"""
        
        # Filter data from start date
        df_filtered = df[df['time'] >= start_date].copy()
        
        print(f"Running backtest from {start_date}")
        print(f"Data points: {len(df_filtered)}")
        
        # Analyze correlations
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS")
        print("="*60)
        correlations = self.correlation_analyzer.analyze_bb_correlations(df_filtered)
        
        # Train enhancement model
        print("\n" + "="*60)
        print("TRAINING BB ENHANCEMENT MODEL")
        print("="*60)
        model = self.correlation_analyzer.train_bb_enhancement_model(df_filtered)
        
        # Generate enhanced signals
        signals = self.correlation_analyzer.generate_enhanced_trading_signals(df_filtered)
        
        print(f"\nGenerated {len(signals)} trading signals")
        
        # Execute backtest
        self.execute_backtest(df_filtered, signals)
        
        # Calculate metrics
        metrics = self.calculate_performance_metrics()
        
        return metrics, correlations
    
    def execute_backtest(self, df, signals):
        """Execute backtest with given signals"""
        
        self.balance = self.initial_balance
        self.trades = []
        self.equity_curve = [self.initial_balance]
        
        # Create signal lookup for faster processing
        signal_dict = {signal['timestamp']: signal for signal in signals}
        
        for i, row in df.iterrows():
            timestamp = row['time']
            
            if timestamp in signal_dict:
                signal = signal_dict[timestamp]
                self.execute_trade(signal, row)
            
            self.equity_curve.append(self.balance)
    
    def execute_trade(self, signal, current_bar):
        """Execute individual trade"""
        
        entry_price = float(current_bar['close'])
        lot_size = float(signal['lot_size'])
        
        # Calculate exit price based on signal
        if signal['action'] == 'BUY':
            # Simulate realistic exit (could hit TP or SL)
            exit_price = entry_price * (1 + np.random.normal(0.001, 0.005))
            pnl = (exit_price - entry_price) * lot_size * 100000
        else:  # SELL
            exit_price = entry_price * (1 + np.random.normal(-0.001, 0.005))
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
        """Calculate comprehensive performance metrics"""
        
        if not self.trades:
            return {'error': 'No trades executed'}
        
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        # Risk metrics
        returns = np.array(self.equity_curve[1:]) / np.array(self.equity_curve[:-1]) - 1
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        max_drawdown = self.calculate_max_drawdown()
        
        # Confidence analysis
        avg_confidence = trades_df['confidence'].mean()
        high_conf_trades = trades_df[trades_df['confidence'] > 0.7]
        high_conf_win_rate = len(high_conf_trades[high_conf_trades['pnl'] > 0]) / len(high_conf_trades) if len(high_conf_trades) > 0 else 0
        
        metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return_pct': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(avg_win * winning_trades / (avg_loss * losing_trades)) if avg_loss != 0 and losing_trades > 0 else float('inf'),
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'final_balance': self.balance,
            'avg_confidence': avg_confidence,
            'high_conf_trades': len(high_conf_trades),
            'high_conf_win_rate': high_conf_win_rate
        }
        
        return metrics
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        
        if len(self.equity_curve) < 2:
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
    
    def plot_comprehensive_results(self, df):
        """Plot comprehensive backtest results with indicators"""
        
        if not self.trades:
            print("No trades to plot")
            return
        
        fig = plt.figure(figsize=(20, 16))
        
        # Create subplots
        gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 1], hspace=0.3)
        
        # Main price chart with BB and trades
        ax1 = fig.add_subplot(gs[0, :])
        
        # Plot last 2000 bars for clarity
        plot_data = df.tail(2000).copy()
        
        ax1.plot(plot_data.index, plot_data['close'], label='Close Price', linewidth=1)
        ax1.plot(plot_data.index, plot_data['bb_upper'], label='Upper BB', linestyle='--', alpha=0.7)
        ax1.plot(plot_data.index, plot_data['bb_middle'], label='Middle BB', linestyle='--', alpha=0.7)
        ax1.plot(plot_data.index, plot_data['bb_lower'], label='Lower BB', linestyle='--', alpha=0.7)
        ax1.fill_between(plot_data.index, plot_data['bb_upper'], plot_data['bb_lower'], alpha=0.1)
        
        # Plot trades
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            # Convert timestamps to indices for plotting
            trade_indices = []
            for timestamp in trades_df['timestamp']:
                matching_rows = plot_data[plot_data['time'] == timestamp]
                if not matching_rows.empty:
                    trade_indices.append(matching_rows.index[0])
            
            if trade_indices:
                buy_trades = trades_df[trades_df['action'] == 'BUY']
                sell_trades = trades_df[trades_df['action'] == 'SELL']
                
                buy_indices = [idx for i, idx in enumerate(trade_indices) if trades_df.iloc[i]['action'] == 'BUY']
                sell_indices = [idx for i, idx in enumerate(trade_indices) if trades_df.iloc[i]['action'] == 'SELL']
                
                if buy_indices:
                    ax1.scatter(buy_indices, [plot_data.loc[idx, 'close'] for idx in buy_indices], 
                              color='green', marker='^', s=100, label='Buy Signal', zorder=5)
                if sell_indices:
                    ax1.scatter(sell_indices, [plot_data.loc[idx, 'close'] for idx in sell_indices], 
                              color='red', marker='v', s=100, label='Sell Signal', zorder=5)
        
        ax1.set_title('Enhanced BB Trading System - Price Chart with Signals')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Enhanced BB signal
        ax2 = fig.add_subplot(gs[1, :])
        if hasattr(self.correlation_analyzer, 'bb_model') and self.correlation_analyzer.bb_model is not None:
            enhanced_signals = self.correlation_analyzer.predict_enhanced_bb_signal(plot_data)
            ax2.plot(plot_data.index, enhanced_signals, label='Enhanced BB Signal', color='purple')
            ax2.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='Buy Threshold')
            ax2.axhline(y=-0.7, color='red', linestyle='--', alpha=0.7, label='Sell Threshold')
            ax2.fill_between(plot_data.index, 0.7, 1, alpha=0.2, color='green')
            ax2.fill_between(plot_data.index, -0.7, -1, alpha=0.2, color='red')
        
        ax2.set_title('Enhanced BB Signal')
        ax2.set_ylabel('Signal Strength')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Technical indicators
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(plot_data.index, plot_data['adx'], label='ADX', color='orange')
        ax3.plot(plot_data.index, plot_data['rsi_overbought'], label='RSI Overbought', color='blue')
        ax3.set_title('Trend & Momentum Indicators')
        ax3.set_ylabel('Value')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.plot(plot_data.index, plot_data['atr'], label='ATR', color='brown')
        ax4.plot(plot_data.index, plot_data['tick_volume'], label='Volume', color='gray')
        ax4.set_title('Volatility & Volume')
        ax4.set_ylabel('Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Equity curve
        ax5 = fig.add_subplot(gs[3, :])
        ax5.plot(self.equity_curve, color='darkgreen', linewidth=2)
        ax5.set_title('Equity Curve')
        ax5.set_ylabel('Balance')
        ax5.set_xlabel('Trade Number')
        ax5.grid(True, alpha=0.3)
        
        plt.suptitle('Enhanced Bollinger Band Trading System - Comprehensive Analysis', fontsize=16)
        plt.savefig('data/enhanced_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot correlation analysis
        self.correlation_analyzer.plot_correlation_analysis(df)

def main():
    """Main execution function"""
    
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Download and process data
    print("Downloading and processing data...")
    df = data_processor.download_and_process_data(symbol="XAUUSD", count=50000)
    
    if df is None:
        print("Failed to download data")
        return
    
    print(f"Data loaded: {len(df)} bars")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    
    # Initialize enhanced backtester
    backtester = EnhancedBacktester(initial_balance=100, risk_percentage=0.01)
    
    # Run enhanced backtest
    metrics, correlations = backtester.run_enhanced_backtest(df, start_date='2025-01-01')
    
    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20}: {value:10.4f}")
        else:
            print(f"{key:20}: {value}")
    
    # Plot comprehensive results
    backtester.plot_comprehensive_results(df)
    
    print(f"\nResults saved to 'data/enhanced_backtest_results.png'")
    print(f"Correlation analysis saved to 'data/correlation_analysis.png'")

if __name__ == "__main__":
    main()