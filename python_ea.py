import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import os
import glob
import pickle
from datetime import datetime, timedelta
from data_processor import DataProcessor
from onnx_model import ONNXPricePredictor
from signal_generator import SignalGenerator
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

class PythonEA:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.predictor = ONNXPricePredictor()
        self.signal_generator = SignalGenerator()
        
        # Trading parameters
        self.symbol = "XAUUSD"
        self.base_risk_percent = 1.0
        self.max_spread = 50.0
        self.min_confidence = 0.75
        self.max_positions = 3
        
        # Advanced parameters
        self.atr_period = 14
        self.atr_sl_multiplier = 2.0
        self.atr_tp_multiplier = 2.0
        self.breakeven_trigger = 1.5
        self.trailing_start = 0.5
        self.trailing_step = 0.1
        
        # Learning and adaptation
        self.trade_history = []
        self.consecutive_losses = 0
        self.max_consecutive_losses = 3
        self.recovery_multiplier = 1.5
        self.pattern_memory = {}
        self.feature_correlations = {}
        
        # State variables
        self.is_running = False
        self.last_retrain_time = 0
        self.last_pattern_analysis = 0
        self.model_accuracy = 0.0
        
    def initialize(self):
        """Initialize MT5 connection and advanced learning system"""
        if not mt5.initialize():
            print("MT5 initialization failed")
            return False
        
        print("Advanced AI Trading EA initialized")
        print(f"Account: {mt5.account_info().login}")
        print(f"Balance: ${mt5.account_info().balance}")
        
        # Load trade history
        self.load_trade_history()
        
        # Learn from all data files
        self.learn_from_data_files()
        
        # Load or train advanced model
        self.initialize_advanced_model()
        
        # Analyze patterns and correlations
        self.analyze_patterns_and_correlations()
        
        return True
    
    def learn_from_data_files(self):
        """Learn from all CSV data files in directory"""
        data_files = glob.glob('data/*.csv')
        all_data = []
        
        for file_path in data_files:
            try:
                df = pd.read_csv(file_path)
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                all_data.append(df)
                print(f"Loaded data from {file_path}: {len(df)} rows")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        if all_data:
            try:
                combined_data = pd.concat(all_data, ignore_index=True)
                if 'time' in combined_data.columns:
                    combined_data = combined_data.drop_duplicates().sort_values('time')
                else:
                    combined_data = combined_data.drop_duplicates()
                self.master_dataset = combined_data
                print(f"Combined dataset: {len(combined_data)} total rows")
            except Exception as e:
                print(f"Error combining data files: {e}")
                self.master_dataset = None
        else:
            print("No data files found, downloading fresh data...")
            try:
                self.master_dataset = self.data_processor.download_and_process_data(self.symbol, count=10000)
            except Exception as e:
                print(f"Error downloading data: {e}")
                self.master_dataset = None
    
    def initialize_advanced_model(self):
        """Initialize advanced prediction model"""
        try:
            self.predictor.load_onnx_model()
            print("Existing model loaded")
            self.model_accuracy = 0.7  # Default assumption for loaded model
        except:
            print("Training advanced model with all available data...")
            if hasattr(self, 'master_dataset') and self.master_dataset is not None:
                # Clean the master dataset
                clean_data = self.clean_data_for_training(self.master_dataset)
                
                if clean_data is not None and len(clean_data) > 100:
                    try:
                        mse, r2 = self.predictor.train_model(clean_data)
                        self.model_accuracy = r2 if r2 > 0 else 0.5
                        print(f"Model trained - Accuracy: {r2:.4f}")
                    except Exception as e:
                        print(f"Error training model: {e}")
                        self.model_accuracy = 0.5
                else:
                    print("Insufficient clean data for model training")
                    # Try to get fresh data
                    try:
                        fresh_data = self.data_processor.download_and_process_data(self.symbol, count=5000)
                        if fresh_data is not None:
                            clean_fresh = self.clean_data_for_training(fresh_data)
                            if clean_fresh is not None:
                                mse, r2 = self.predictor.train_model(clean_fresh)
                                self.model_accuracy = r2 if r2 > 0 else 0.5
                                print(f"Model trained with fresh data - Accuracy: {r2:.4f}")
                            else:
                                self.model_accuracy = 0.5
                        else:
                            self.model_accuracy = 0.5
                    except Exception as e:
                        print(f"Error getting fresh data: {e}")
                        self.model_accuracy = 0.5
            else:
                print("No master dataset available, using default model accuracy")
                self.model_accuracy = 0.5
    
    def analyze_patterns_and_correlations(self):
        """Analyze patterns and feature correlations"""
        if not hasattr(self, 'master_dataset') or self.master_dataset is None:
            print("No master dataset available for pattern analysis")
            return
        
        try:
            df = self.master_dataset.copy()
            
            # Calculate feature correlations with price movement
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if 'close' in numeric_cols and len(df) > 1:
                df['price_change'] = df['close'].pct_change()
                df = df.dropna()  # Remove NaN values
                
                if len(df) > 0 and 'price_change' in df.columns:
                    # Recalculate numeric columns after adding price_change
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    correlations = df[numeric_cols].corr()['price_change'].abs().sort_values(ascending=False)
                    self.feature_correlations = correlations.to_dict()
                    print(f"Top correlated features: {list(correlations.head(5).index)}")
                else:
                    print("Insufficient data for correlation analysis")
            else:
                print("No 'close' column found for correlation analysis")
            
            # Identify winning patterns
            self.identify_winning_patterns(df)
            
        except Exception as e:
            print(f"Error in pattern analysis: {e}")
            self.feature_correlations = {}
    
    def identify_winning_patterns(self, df):
        """Identify patterns that lead to profitable trades"""
        if len(self.trade_history) < 10:
            return
        
        # Analyze successful vs failed trades
        winning_trades = [t for t in self.trade_history if t.get('profit', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('profit', 0) < 0]
        
        if winning_trades and losing_trades:
            # Pattern analysis logic here
            win_rate = len(winning_trades) / len(self.trade_history)
            self.pattern_memory['win_rate'] = win_rate
            print(f"Historical win rate: {win_rate:.2%}")
    
    def shutdown(self):
        """Shutdown MT5 connection"""
        mt5.shutdown()
        print("MT5 Python EA stopped")
    
    def check_spread(self):
        """Check if spread is acceptable"""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False
        
        spread = (tick.ask - tick.bid) / tick.ask * 100000  # Points
        return spread <= self.max_spread
    
    def get_positions_count(self):
        """Get number of open positions for this symbol"""
        positions = mt5.positions_get(symbol=self.symbol)
        return len(positions) if positions else 0
    
    def calculate_dynamic_lot_size(self, confidence, signal_strength=1.0):
        """Calculate dynamic lot size based on multiple factors"""
        account_info = mt5.account_info()
        balance = account_info.balance
        
        # Base risk calculation
        base_risk = self.base_risk_percent / 100
        
        # Adjust for consecutive losses (recovery strategy)
        if self.consecutive_losses >= 2:
            recovery_factor = min(self.recovery_multiplier ** self.consecutive_losses, 3.0)
            print(f"Recovery mode: {self.consecutive_losses} losses, factor: {recovery_factor:.2f}")
        else:
            recovery_factor = 1.0
        
        # Adjust for model accuracy
        accuracy_factor = max(self.model_accuracy, 0.5) if self.model_accuracy > 0 else 0.7
        
        # Adjust for confidence and signal strength
        confidence_factor = min(confidence * 1.2, 1.5)
        strength_factor = min(signal_strength, 1.3)
        
        # Calculate final lot size
        risk_amount = balance * base_risk * recovery_factor * accuracy_factor * confidence_factor * strength_factor
        base_lot = risk_amount / 1000  # Simplified for XAUUSD
        
        # Ensure within broker limits
        symbol_info = mt5.symbol_info(self.symbol)
        min_lot = symbol_info.volume_min
        max_lot = min(symbol_info.volume_max, balance / 10000)  # Max 10% of balance
        
        lot_size = max(min_lot, min(base_lot, max_lot))
        return round(lot_size, 2)
    
    def get_advanced_atr(self, period=None):
        """Get advanced ATR with multiple timeframe analysis"""
        if period is None:
            period = self.atr_period
        
        # Get data from multiple timeframes
        timeframes = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_H1]
        atr_values = []
        
        for tf in timeframes:
            rates = mt5.copy_rates_from_pos(self.symbol, tf, 0, period + 1)
            if rates is not None and len(rates) >= period:
                df = pd.DataFrame(rates)
                high_low = df['high'] - df['low']
                high_close = abs(df['high'] - df['close'].shift(1))
                low_close = abs(df['low'] - df['close'].shift(1))
                
                true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                atr = true_range.rolling(window=period).mean().iloc[-1]
                
                if not pd.isna(atr):
                    atr_values.append(atr)
        
        # Use weighted average of ATR values
        if atr_values:
            weights = [0.5, 0.3, 0.2][:len(atr_values)]
            weighted_atr = sum(atr * w for atr, w in zip(atr_values, weights)) / sum(weights)
            return weighted_atr
        
        return 0.002  # Fallback value
    
    def execute_advanced_buy(self, lot_size, confidence, signal_data):
        """Execute advanced buy order with ATR-based SL/TP"""
        tick = mt5.symbol_info_tick(self.symbol)
        symbol_info = mt5.symbol_info(self.symbol)
        if tick is None or symbol_info is None:
            return False
        
        price = tick.ask
        atr = self.get_advanced_atr()
        
        # Calculate ATR-based SL/TP
        sl = price - (atr * self.atr_sl_multiplier)
        tp = price + (atr * self.atr_tp_multiplier)
        
        # Round to symbol digits
        sl = round(sl, symbol_info.digits)
        tp = round(tp, symbol_info.digits)
        
        # Execute market order first
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY,
            "price": price,
            "comment": f"AI_BUY_{confidence:.3f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            # Set SL/TP after execution
            ticket = result.order
            self.set_sl_tp_after_execution(ticket, sl, tp)
            
            # Record trade
            trade_record = {
                'ticket': ticket,
                'type': 'BUY',
                'entry_price': price,
                'sl': sl,
                'tp': tp,
                'lot_size': lot_size,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'atr': atr,
                'signal_data': signal_data
            }
            self.trade_history.append(trade_record)
            
            print(f"✓ BUY executed: {lot_size} lots at {price}, SL: {sl}, TP: {tp}")
            return True
        else:
            print(f"✗ BUY failed: {result.retcode}")
            return False
    
    def execute_advanced_sell(self, lot_size, confidence, signal_data):
        """Execute advanced sell order with ATR-based SL/TP"""
        tick = mt5.symbol_info_tick(self.symbol)
        symbol_info = mt5.symbol_info(self.symbol)
        if tick is None or symbol_info is None:
            return False
        
        price = tick.bid
        atr = self.get_advanced_atr()
        
        # Calculate ATR-based SL/TP
        sl = price + (atr * self.atr_sl_multiplier)
        tp = price - (atr * self.atr_tp_multiplier)
        
        # Round to symbol digits
        sl = round(sl, symbol_info.digits)
        tp = round(tp, symbol_info.digits)
        
        # Execute market order first
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_SELL,
            "price": price,
            "comment": f"AI_SELL_{confidence:.3f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            # Set SL/TP after execution
            ticket = result.order
            self.set_sl_tp_after_execution(ticket, sl, tp)
            
            # Record trade
            trade_record = {
                'ticket': ticket,
                'type': 'SELL',
                'entry_price': price,
                'sl': sl,
                'tp': tp,
                'lot_size': lot_size,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'atr': atr,
                'signal_data': signal_data
            }
            self.trade_history.append(trade_record)
            
            print(f"✓ SELL executed: {lot_size} lots at {price}, SL: {sl}, TP: {tp}")
            return True
        else:
            print(f"✗ SELL failed: {result.retcode}")
            return False
    
    def set_sl_tp_after_execution(self, ticket, sl, tp):
        """Set SL/TP after order execution"""
        time.sleep(0.1)  # Small delay
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        result = mt5.order_send(request)
        return result.retcode == mt5.TRADE_RETCODE_DONE
    
    def handle_advanced_position_management(self):
        """Advanced position management with breakeven and trailing"""
        positions = mt5.positions_get(symbol=self.symbol)
        if not positions:
            return
        
        for position in positions:
            if not position.comment.startswith("AI_"):
                continue
            
            ticket = position.ticket
            pos_type = position.type
            open_price = position.price_open
            current_sl = position.sl
            current_tp = position.tp
            
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                continue
            
            atr = self.get_advanced_atr()
            
            if pos_type == mt5.POSITION_TYPE_BUY:
                current_price = tick.bid
                profit_pips = (current_price - open_price) / atr
                
                # Breakeven management
                if profit_pips >= self.breakeven_trigger and current_sl < open_price:
                    new_sl = open_price + (atr * 0.1)  # Small profit lock
                    self.modify_position(ticket, new_sl, current_tp)
                    print(f"Breakeven set for BUY position {ticket}")
                
                # Trailing stop
                elif profit_pips >= self.trailing_start:
                    new_sl = current_price - (atr * self.trailing_step)
                    if new_sl > current_sl:
                        self.modify_position(ticket, new_sl, current_tp)
            
            elif pos_type == mt5.POSITION_TYPE_SELL:
                current_price = tick.ask
                profit_pips = (open_price - current_price) / atr
                
                # Breakeven management
                if profit_pips >= self.breakeven_trigger and (current_sl > open_price or current_sl == 0):
                    new_sl = open_price - (atr * 0.1)  # Small profit lock
                    self.modify_position(ticket, new_sl, current_tp)
                    print(f"Breakeven set for SELL position {ticket}")
                
                # Trailing stop
                elif profit_pips >= self.trailing_start:
                    new_sl = current_price + (atr * self.trailing_step)
                    if new_sl < current_sl or current_sl == 0:
                        self.modify_position(ticket, new_sl, current_tp)
    
    def modify_position(self, ticket, sl, tp):
        """Modify position SL/TP"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"Trailing stop updated for position {ticket}")
        return result.retcode == mt5.TRADE_RETCODE_DONE
    
    def clean_data_for_training(self, df):
        """Clean data to remove NaN values and ensure quality"""
        if df is None or df.empty:
            return None
        
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        # Ensure minimum data size
        if len(df_clean) < 100:
            print(f"Insufficient clean data: {len(df_clean)} rows")
            return None
        
        # Remove infinite values
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
        
        return df_clean
    
    def generate_advanced_signals(self):
        """Generate advanced multi-indicator signals"""
        try:
            # Get latest data
            df = self.data_processor.download_and_process_data(self.symbol, count=1000)
            if df is None:
                return
            
            # Clean data
            df_clean = self.clean_data_for_training(df)
            if df_clean is None:
                print("No clean data available for signal generation")
                return
            
            # Periodic retraining with all available data
            current_time = time.time()
            if current_time - self.last_retrain_time > 1800:  # Every 30 minutes
                print("Retraining model with latest data...")
                if hasattr(self, 'master_dataset') and self.master_dataset is not None:
                    # Combine with latest data
                    combined_df = pd.concat([self.master_dataset, df_clean]).drop_duplicates()
                    combined_clean = self.clean_data_for_training(combined_df)
                    
                    if combined_clean is not None and len(combined_clean) > 200:
                        try:
                            mse, r2 = self.predictor.train_model(combined_clean)
                            self.model_accuracy = r2 if r2 > 0 else 0.5
                            print(f"Model retrained - Accuracy: {r2:.4f}")
                        except Exception as e:
                            print(f"Model training failed: {e}")
                    else:
                        print("Insufficient clean data for retraining")
                else:
                    # Train with current data only
                    try:
                        mse, r2 = self.predictor.train_model(df_clean)
                        self.model_accuracy = r2 if r2 > 0 else 0.5
                        print(f"Model trained with current data - Accuracy: {r2:.4f}")
                    except Exception as e:
                        print(f"Model training failed: {e}")
                
                self.last_retrain_time = current_time
            
            # Pattern analysis
            if current_time - self.last_pattern_analysis > 3600:  # Every hour
                self.analyze_patterns_and_correlations()
                self.last_pattern_analysis = current_time
            
            # Generate multiple signal types
            primary_signals = self.signal_generator.generate_signals(df_clean)
            
            # Add custom high-confidence signals
            enhanced_signals = self.enhance_signals_with_patterns(primary_signals, df_clean)
            
            for signal in enhanced_signals:
                if signal['action'] == 'HOLD':
                    continue
                
                # Advanced signal validation
                if not self.validate_advanced_signal(signal, df_clean):
                    continue
                
                # Calculate dynamic lot size
                lot_size = self.calculate_dynamic_lot_size(
                    signal['confidence'], 
                    signal.get('strength', 1.0)
                )
                
                # Execute trade with advanced features
                if signal['action'] == 'BUY':
                    success = self.execute_advanced_buy(lot_size, signal['confidence'], signal)
                elif signal['action'] == 'SELL':
                    success = self.execute_advanced_sell(lot_size, signal['confidence'], signal)
                
                if success:
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                
                print(f"Signal: {signal['action']} | Conf: {signal['confidence']:.3f} | Strength: {signal.get('strength', 1.0):.2f}")
        
        except Exception as e:
            print(f"Error in advanced signal generation: {e}")
    
    def enhance_signals_with_patterns(self, signals, df):
        """Enhance signals using learned patterns"""
        enhanced = []
        
        for signal in signals:
            # Apply pattern-based enhancements
            if self.feature_correlations:
                # Boost confidence based on high-correlation features
                feature_boost = self.calculate_feature_boost(df)
                signal['confidence'] *= feature_boost
                signal['strength'] = feature_boost
            
            # Apply historical pattern matching
            pattern_score = self.match_historical_patterns(signal, df)
            signal['confidence'] *= pattern_score
            
            enhanced.append(signal)
        
        return enhanced
    
    def calculate_feature_boost(self, df):
        """Calculate confidence boost based on feature correlations"""
        if not self.feature_correlations or df.empty:
            return 1.0
        
        boost = 1.0
        latest_row = df.iloc[-1]
        
        # Check top correlated features
        for feature, correlation in list(self.feature_correlations.items())[:5]:
            if feature in latest_row and correlation > 0.3:
                boost *= (1 + correlation * 0.2)
        
        return min(boost, 1.5)  # Cap at 1.5x
    
    def match_historical_patterns(self, signal, df):
        """Match current conditions with historical winning patterns"""
        if len(self.trade_history) < 10:
            return 1.0
        
        # Simple pattern matching based on recent performance
        recent_trades = self.trade_history[-10:]
        similar_trades = [t for t in recent_trades if t['type'] == signal['action']]
        
        if similar_trades:
            win_rate = len([t for t in similar_trades if t.get('profit', 0) > 0]) / len(similar_trades)
            return max(0.7, min(1.3, win_rate * 1.2))
        
        return 1.0
    
    def validate_advanced_signal(self, signal, df):
        """Advanced signal validation with multiple criteria"""
        # Check confidence threshold
        if signal['confidence'] < self.min_confidence:
            print(f"Low confidence: {signal['confidence']:.3f}")
            return False
        
        # Check spread
        if not self.check_spread():
            print("High spread - signal rejected")
            return False
        
        # Check position limit
        if self.get_positions_count() >= self.max_positions:
            print("Max positions reached")
            return False
        
        # Check consecutive losses (risk management)
        if self.consecutive_losses >= self.max_consecutive_losses:
            if signal['confidence'] < 0.9:  # Only very high confidence trades
                print(f"Too many consecutive losses ({self.consecutive_losses}), requiring higher confidence")
                return False
        
        # Market condition check
        if not self.check_market_conditions(df):
            print("Unfavorable market conditions")
            return False
        
        # Model accuracy check
        if self.model_accuracy > 0 and self.model_accuracy < 0.6:
            print(f"Model accuracy too low: {self.model_accuracy:.3f}")
            return False
        
        return True
    
    def check_market_conditions(self, df):
        """Check if market conditions are favorable for trading"""
        if df.empty or len(df) < 20:
            return False
        
        # Check volatility
        recent_volatility = df['close'].pct_change().tail(20).std()
        if recent_volatility > 0.05:  # Too volatile
            return False
        
        # Check if market is trending or ranging
        sma_20 = df['close'].rolling(20).mean().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Avoid trading in extreme conditions
        price_deviation = abs(current_price - sma_20) / sma_20
        if price_deviation > 0.02:  # 2% deviation
            return False
        
        return True
    
    def load_trade_history(self):
        """Load trade history from file"""
        try:
            with open('trade_history.pkl', 'rb') as f:
                self.trade_history = pickle.load(f)
            print(f"Loaded {len(self.trade_history)} historical trades")
        except FileNotFoundError:
            self.trade_history = []
            print("No trade history found, starting fresh")
    
    def save_trade_history(self):
        """Save trade history to file"""
        try:
            with open('trade_history.pkl', 'wb') as f:
                pickle.dump(self.trade_history, f)
        except Exception as e:
            print(f"Error saving trade history: {e}")
    
    def update_trade_results(self):
        """Update trade results for completed positions"""
        for trade in self.trade_history:
            if 'profit' not in trade:
                # Check if position is still open
                positions = mt5.positions_get(symbol=self.symbol)
                position_exists = any(p.ticket == trade['ticket'] for p in positions or [])
                
                if not position_exists:
                    # Position closed, get history
                    deals = mt5.history_deals_get(position=trade['ticket'])
                    if deals and len(deals) >= 2:
                        entry_deal = deals[0]
                        exit_deal = deals[-1]
                        profit = exit_deal.profit
                        trade['profit'] = profit
                        trade['exit_time'] = datetime.fromtimestamp(exit_deal.time)
                        
                        if profit < 0:
                            self.consecutive_losses += 1
                        else:
                            self.consecutive_losses = 0
    
    def run(self):
        """Advanced main trading loop"""
        if not self.initialize():
            return
        
        self.is_running = True
        print("Advanced AI Trading EA started - Press Ctrl+C to stop")
        print(f"Model Accuracy: {self.model_accuracy:.3f}")
        print(f"Historical Trades: {len(self.trade_history)}")
        print(f"Consecutive Losses: {self.consecutive_losses}")
        
        try:
            while self.is_running:
                # Update trade results
                self.update_trade_results()
                
                # Generate and execute advanced signals
                self.generate_advanced_signals()
                
                # Advanced position management
                self.handle_advanced_position_management()
                
                # Save trade history periodically
                if len(self.trade_history) % 10 == 0:
                    self.save_trade_history()
                
                # Wait 30 seconds
                time.sleep(30)
                
        except KeyboardInterrupt:
            print("Stopping Advanced AI Trading EA...")
            self.is_running = False
        
        finally:
            self.save_trade_history()
            self.shutdown()

if __name__ == "__main__":
    ea = PythonEA()
    ea.run()