import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from data_processor import DataProcessor
import os

class MemoryEnhancedTrader:
    def __init__(self, symbol="XAUUSD"):
        self.symbol = symbol
        self.data_processor = DataProcessor()
        self.models = {}
        self.scalers = {}
        self.trade_history = []
        self.performance_memory = {}
        self.market_memory = {}
        
    def load_pretrained_models(self):
        """Load latest pre-trained ensemble models"""
        model_files = [f for f in os.listdir('models') if f.startswith(f'{self.symbol}_pretrained_ensemble_')]
        
        if not model_files:
            print("No pre-trained models found")
            return False
        
        latest_model = sorted(model_files)[-1]
        
        # Load models
        with open(f'models/{latest_model}', 'rb') as f:
            self.models = pickle.load(f)
        
        # Load scalers
        scaler_file = latest_model.replace('pretrained_ensemble', 'scalers')
        with open(f'models/{scaler_file}', 'rb') as f:
            self.scalers = pickle.load(f)
        
        print(f"Loaded pre-trained models: {latest_model}")
        return True
    
    def load_trade_memory(self):
        """Load historical trade memory"""
        memory_file = 'models/trade_memory.json'
        
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                self.trade_history = json.load(f)
            
            # Process memory into performance metrics
            self.update_performance_memory()
            print(f"Loaded {len(self.trade_history)} historical trades")
        else:
            print("No trade memory found, starting fresh")
    
    def update_performance_memory(self):
        """Update performance memory from trade history"""
        if not self.trade_history:
            return
        
        df = pd.DataFrame(self.trade_history)
        
        # Calculate performance metrics
        self.performance_memory = {
            'total_trades': len(df),
            'win_rate': (df['pnl'] > 0).mean() if 'pnl' in df.columns else 0,
            'avg_profit': df[df['pnl'] > 0]['pnl'].mean() if 'pnl' in df.columns else 0,
            'avg_loss': df[df['pnl'] < 0]['pnl'].mean() if 'pnl' in df.columns else 0,
            'best_trade': df['pnl'].max() if 'pnl' in df.columns else 0,
            'worst_trade': df['pnl'].min() if 'pnl' in df.columns else 0,
            'avg_confidence': df['confidence'].mean() if 'confidence' in df.columns else 0
        }
        
        # Market condition performance
        if 'market_conditions' in df.columns:
            self.analyze_market_performance(df)
    
    def analyze_market_performance(self, df):
        """Analyze performance by market conditions"""
        self.market_memory = {}
        
        # Group by market conditions and calculate performance
        for _, trade in df.iterrows():
            conditions = trade.get('market_conditions', {})
            
            for condition, value in conditions.items():
                if condition not in self.market_memory:
                    self.market_memory[condition] = {'trades': [], 'performance': 0}
                
                self.market_memory[condition]['trades'].append({
                    'pnl': trade.get('pnl', 0),
                    'confidence': trade.get('confidence', 0)
                })
        
        # Calculate performance for each condition
        for condition in self.market_memory:
            trades = self.market_memory[condition]['trades']
            if trades:
                avg_pnl = np.mean([t['pnl'] for t in trades])
                self.market_memory[condition]['performance'] = avg_pnl
    
    def get_enhanced_features(self, df):
        """Get features enhanced with memory"""
        # Add performance memory features
        for key, value in self.performance_memory.items():
            if isinstance(value, (int, float)):
                df[f'memory_{key}'] = value
        
        # Add market memory features
        current_conditions = self.get_current_market_conditions(df)
        
        for condition, value in current_conditions.items():
            if condition in self.market_memory:
                perf = self.market_memory[condition]['performance']
                if isinstance(perf, (int, float)):
                    df[f'market_memory_{condition}'] = perf
            else:
                df[f'market_memory_{condition}'] = 0
        
        return df
    
    def get_current_market_conditions(self, df):
        """Identify current market conditions"""
        latest = df.iloc[-1]
        
        conditions = {
            'high_volatility': latest.get('atr', 0) > df['atr'].quantile(0.8),
            'trending': latest.get('adx', 0) > 0.25,
            'overbought': latest.get('rsi_overbought', 0) > 0.7,
            'high_volume': latest.get('tick_volume', 0) > df['tick_volume'].quantile(0.8)
        }
        
        return conditions
    
    def predict_with_ensemble(self, df):
        """Make predictions using ensemble models"""
        if not self.models:
            print("No models loaded")
            return None
        
        # Enhance features with memory
        enhanced_df = self.get_enhanced_features(df)
        
        predictions = {}
        
        for target_name, target_models in self.models.items():
            target_predictions = {}
            
            # Prepare features - only numeric columns
            exclude_cols = ['time', 'open', 'high', 'low', 'close']
            feature_cols = []
            
            for col in enhanced_df.columns:
                if col not in exclude_cols:
                    if enhanced_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        feature_cols.append(col)
            
            X = enhanced_df[feature_cols].fillna(0).iloc[-1:]  # Latest row
            
            for model_name, model in target_models.items():
                try:
                    if model_name == 'nn' and target_name in self.scalers:
                        # Scale for neural network
                        X_scaled = self.scalers[target_name].transform(X)
                        pred = model.predict(X_scaled)[0]
                    else:
                        pred = model.predict(X)[0]
                    
                    target_predictions[model_name] = pred
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
                    target_predictions[model_name] = 0
            
            # Ensemble prediction (weighted average)
            weights = {'rf': 0.4, 'gb': 0.4, 'nn': 0.2}
            ensemble_pred = sum(weights.get(name, 0.33) * pred 
                              for name, pred in target_predictions.items())
            
            predictions[target_name] = {
                'ensemble': ensemble_pred,
                'individual': target_predictions
            }
        
        return predictions
    
    def generate_memory_enhanced_signals(self, df):
        """Generate signals enhanced with memory and learning"""
        predictions = self.predict_with_ensemble(df)
        
        if not predictions:
            return []
        
        signals = []
        current_price = df['close'].iloc[-1]
        current_time = df['time'].iloc[-1]
        
        # Extract predictions
        direction_pred = predictions.get('price_direction', {}).get('ensemble', 0.5)
        change_pred = predictions.get('price_change', {}).get('ensemble', 0)
        volatility_pred = predictions.get('volatility', {}).get('ensemble', 0)
        
        # Calculate confidence based on memory
        base_confidence = abs(direction_pred - 0.5) * 2  # 0 to 1
        
        # Adjust confidence based on historical performance
        memory_adjustment = self.get_memory_confidence_adjustment()
        final_confidence = min(1.0, base_confidence * memory_adjustment)
        
        # Generate signal
        if direction_pred > 0.6 and final_confidence > 0.7:
            action = 'BUY'
            tp = current_price * (1 + abs(change_pred) * 2)
            sl = current_price * (1 - volatility_pred * 2)
        elif direction_pred < 0.4 and final_confidence > 0.7:
            action = 'SELL'
            tp = current_price * (1 - abs(change_pred) * 2)
            sl = current_price * (1 + volatility_pred * 2)
        else:
            action = 'HOLD'
            tp = None
            sl = None
        
        if action != 'HOLD':
            signal = {
                'timestamp': current_time,
                'action': action,
                'price': current_price,
                'confidence': final_confidence,
                'tp': tp,
                'sl': sl,
                'lot_size': self.calculate_position_size(final_confidence),
                'predictions': predictions,
                'memory_adjustment': memory_adjustment
            }
            signals.append(signal)
        
        return signals
    
    def get_memory_confidence_adjustment(self):
        """Adjust confidence based on historical performance"""
        if not self.performance_memory:
            return 1.0
        
        win_rate = self.performance_memory.get('win_rate', 0.5)
        avg_confidence = self.performance_memory.get('avg_confidence', 0.5)
        
        # Boost confidence if historical performance is good
        if win_rate > 0.6 and avg_confidence > 0.7:
            return 1.2
        elif win_rate < 0.4:
            return 0.8
        else:
            return 1.0
    
    def calculate_position_size(self, confidence):
        """Calculate position size based on confidence and memory"""
        base_size = 0.01
        
        # Adjust based on confidence
        confidence_multiplier = confidence
        
        # Adjust based on recent performance
        if self.performance_memory.get('win_rate', 0) > 0.6:
            performance_multiplier = 1.2
        else:
            performance_multiplier = 0.8
        
        return min(0.1, base_size * confidence_multiplier * performance_multiplier)
    
    def record_trade_result(self, signal, exit_price, exit_time):
        """Record trade result for memory learning"""
        entry_price = signal['price']
        action = signal['action']
        
        # Calculate P&L
        if action == 'BUY':
            pnl = (exit_price - entry_price) * signal['lot_size'] * 100000
        else:
            pnl = (entry_price - exit_price) * signal['lot_size'] * 100000
        
        # Calculate duration
        duration = (exit_time - pd.to_datetime(signal['timestamp'])).total_seconds() / 60
        
        # Get market conditions at trade time
        market_conditions = self.get_current_market_conditions(
            self.data_processor.download_and_process_data(self.symbol, count=100)
        )
        
        trade_result = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'pnl': pnl,
            'duration': duration,
            'confidence': signal['confidence'],
            'market_conditions': market_conditions,
            'predictions': signal.get('predictions', {}),
            'memory_adjustment': signal.get('memory_adjustment', 1.0)
        }
        
        # Add to memory
        self.trade_history.append(trade_result)
        
        # Update performance memory
        self.update_performance_memory()
        
        # Save to file
        memory_file = 'models/trade_memory.json'
        with open(memory_file, 'w') as f:
            json.dump(self.trade_history, f, indent=2)
        
        print(f"Trade recorded: {action} P&L: {pnl:.2f}")
        
        return trade_result

def main():
    """Test memory enhanced trader"""
    trader = MemoryEnhancedTrader("XAUUSD")
    
    # Load models and memory
    trader.load_pretrained_models()
    trader.load_trade_memory()
    
    # Get latest data
    df = trader.data_processor.download_and_process_data("XAUUSD", count=1000)
    
    if df is not None:
        # Generate signals
        signals = trader.generate_memory_enhanced_signals(df)
        
        print(f"Generated {len(signals)} signals")
        for signal in signals:
            print(f"Signal: {signal['action']} @ {signal['price']:.5f} (Conf: {signal['confidence']:.3f})")

if __name__ == "__main__":
    main()