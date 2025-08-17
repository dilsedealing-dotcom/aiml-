import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_processor import DataProcessor
import MetaTrader5 as mt5

class AdvancedMLTrainer:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.models = {}
        self.scalers = {}
        self.performance_history = []
        self.trade_memory = []
        
    def collect_multi_timeframe_data(self, symbol="XAUUSD", bars_per_tf=20000):
        """Collect data from multiple timeframes"""
        timeframes = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4,
            'D1': mt5.TIMEFRAME_D1
        }
        
        all_data = {}
        
        for tf_name, tf_code in timeframes.items():
            print(f"Collecting {tf_name} data...")
            
            self.data_processor.mt5.connect()
            df = self.data_processor.mt5.get_data(symbol, tf_code, bars_per_tf)
            
            if df is not None:
                # Process indicators
                from technical_indicators import TechnicalIndicators
                df = TechnicalIndicators.calculate_all_indicators(df)
                df = self.data_processor.add_sl_tp_patterns(df)
                df = self.data_processor.clean_data(df)
                
                # Add timeframe weight as numeric feature
                df['tf_weight'] = self.get_timeframe_weight(tf_name)
                
                all_data[tf_name] = df
                
                # Save individual timeframe data
                df.to_csv(f'data/{symbol}_{tf_name}_pretrained.csv', index=False)
                print(f"Saved {len(df)} bars for {tf_name}")
        
        return all_data
    
    def get_timeframe_weight(self, tf_name):
        """Assign weights to different timeframes"""
        weights = {'M5': 1.0, 'M15': 1.2, 'H1': 1.5, 'H4': 2.0, 'D1': 2.5}
        return weights.get(tf_name, 1.0)
    
    def create_enhanced_features(self, df_dict):
        """Create enhanced features from multi-timeframe data"""
        combined_features = []
        
        for tf_name, df in df_dict.items():
            # Remove timeframe column to avoid string conversion error
            if 'timeframe' in df.columns:
                df = df.drop('timeframe', axis=1)
            
            # Add cross-timeframe features
            df[f'{tf_name}_momentum'] = df['close'].pct_change(5)
            df[f'{tf_name}_volatility'] = df['close'].rolling(20).std()
            df[f'{tf_name}_volume_ma'] = df['tick_volume'].rolling(10).mean()
            
            # Add profit/loss memory features
            df = self.add_pnl_memory_features(df, tf_name)
            
            combined_features.append(df)
        
        # Combine all timeframes
        combined_df = pd.concat(combined_features, ignore_index=True)
        combined_df = combined_df.sort_values('time').reset_index(drop=True)
        
        return combined_df
    
    def add_pnl_memory_features(self, df, tf_name):
        """Add P&L memory and learning features"""
        # Simulate historical P&L patterns
        df['historical_win_rate'] = np.random.uniform(0.4, 0.7, len(df))
        df['avg_profit'] = np.random.uniform(10, 50, len(df))
        df['avg_loss'] = np.random.uniform(-30, -10, len(df))
        df['consecutive_wins'] = np.random.randint(0, 5, len(df))
        df['consecutive_losses'] = np.random.randint(0, 3, len(df))
        
        # Market condition memory
        df['market_regime'] = self.classify_market_regime(df)
        df['regime_performance'] = np.random.uniform(0.3, 0.8, len(df))
        
        return df
    
    def classify_market_regime(self, df):
        """Classify market regime (trending/ranging/volatile)"""
        if 'adx' in df.columns:
            conditions = [
                df['adx'] > 0.25,  # Trending
                df['adx'] <= 0.25,  # Ranging
            ]
            choices = [1, 0]  # 1=trending, 0=ranging
            return np.select(conditions, choices, default=0)
        return np.zeros(len(df))
    
    def train_ensemble_models(self, combined_df):
        """Train ensemble of ML models"""
        print("Training ensemble models...")
        
        # Prepare features - exclude non-numeric columns
        exclude_cols = ['time', 'open', 'high', 'low', 'close']
        feature_cols = []
        
        for col in combined_df.columns:
            if col not in exclude_cols:
                # Check if column is numeric
                if combined_df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    feature_cols.append(col)
        
        X = combined_df[feature_cols].fillna(0)
        
        # Create multiple targets
        targets = {
            'price_direction': (combined_df['close'].shift(-1) > combined_df['close']).astype(int),
            'price_change': combined_df['close'].pct_change().shift(-1),
            'volatility': combined_df['close'].rolling(5).std().shift(-1)
        }
        
        models = {}
        
        for target_name, y in targets.items():
            y = y.fillna(0)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train multiple models
            model_configs = {
                'rf': RandomForestRegressor(n_estimators=200, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'nn': MLPRegressor(hidden_layer_sizes=(100, 50), random_state=42, max_iter=500)
            }
            
            target_models = {}
            
            for model_name, model in model_configs.items():
                print(f"Training {model_name} for {target_name}...")
                
                if model_name == 'nn':
                    model.fit(X_train_scaled, y_train)
                    score = model.score(X_test_scaled, y_test)
                else:
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                
                target_models[model_name] = model
                print(f"{model_name} R² score: {score:.4f}")
            
            models[target_name] = target_models
            self.scalers[target_name] = scaler
        
        self.models = models
        return models
    
    def save_pretrained_models(self, symbol="XAUUSD"):
        """Save all trained models"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        model_file = f'models/{symbol}_pretrained_ensemble_{timestamp}.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(self.models, f)
        
        # Save scalers
        scaler_file = f'models/{symbol}_scalers_{timestamp}.pkl'
        with open(scaler_file, 'wb') as f:
            pickle.dump(self.scalers, f)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'symbol': symbol,
            'model_file': model_file,
            'scaler_file': scaler_file,
            'feature_count': len(self.scalers),
            'model_types': list(self.models.keys())
        }
        
        metadata_file = f'models/{symbol}_metadata_{timestamp}.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Models saved: {model_file}")
        return model_file
    
    def update_trade_memory(self, trade_result):
        """Update trade memory for continuous learning"""
        memory_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': trade_result.get('action'),
            'entry_price': trade_result.get('entry_price'),
            'exit_price': trade_result.get('exit_price'),
            'pnl': trade_result.get('pnl'),
            'duration': trade_result.get('duration'),
            'market_conditions': trade_result.get('market_conditions', {}),
            'confidence': trade_result.get('confidence')
        }
        
        self.trade_memory.append(memory_entry)
        
        # Keep only last 1000 trades
        if len(self.trade_memory) > 1000:
            self.trade_memory = self.trade_memory[-1000:]
        
        # Save memory
        memory_file = 'models/trade_memory.json'
        with open(memory_file, 'w') as f:
            json.dump(self.trade_memory, f, indent=2)
    
    def retrain_with_memory(self, symbol="XAUUSD"):
        """Retrain models with trade memory"""
        if not self.trade_memory:
            print("No trade memory available for retraining")
            return
        
        print("Retraining with trade memory...")
        
        # Convert trade memory to features
        memory_df = pd.DataFrame(self.trade_memory)
        
        # Create learning features from memory
        memory_features = self.extract_memory_features(memory_df)
        
        # Get latest market data
        latest_data = self.data_processor.download_and_process_data(symbol, count=5000)
        
        if latest_data is not None:
            # Combine with memory features
            enhanced_data = self.combine_data_with_memory(latest_data, memory_features)
            
            # Retrain models
            self.train_ensemble_models(enhanced_data)
            
            # Save updated models
            self.save_pretrained_models(symbol)
            
            print("Models retrained with trade memory")
    
    def extract_memory_features(self, memory_df):
        """Extract learning features from trade memory"""
        features = {}
        
        if len(memory_df) > 0:
            # Performance metrics
            features['recent_win_rate'] = (memory_df['pnl'] > 0).mean()
            features['avg_profit'] = memory_df[memory_df['pnl'] > 0]['pnl'].mean()
            features['avg_loss'] = memory_df[memory_df['pnl'] < 0]['pnl'].mean()
            
            # Behavioral patterns
            features['avg_confidence'] = memory_df['confidence'].mean()
            features['trade_frequency'] = len(memory_df)
            
        return features
    
    def combine_data_with_memory(self, market_data, memory_features):
        """Combine market data with memory features"""
        for feature, value in memory_features.items():
            market_data[f'memory_{feature}'] = value
        
        return market_data

def main():
    """Main training function"""
    print("Advanced ML Pre-Training System")
    print("=" * 40)
    
    trainer = AdvancedMLTrainer()
    
    symbol = input("Enter symbol (default XAUUSD): ") or "XAUUSD"
    
    print(f"\nStarting pre-training for {symbol}...")
    
    # Step 1: Collect multi-timeframe data
    print("Step 1: Collecting multi-timeframe data...")
    multi_tf_data = trainer.collect_multi_timeframe_data(symbol, bars_per_tf=20000)
    
    if not multi_tf_data:
        print("Failed to collect data")
        return
    
    # Step 2: Create enhanced features
    print("Step 2: Creating enhanced features...")
    combined_data = trainer.create_enhanced_features(multi_tf_data)
    
    # Step 3: Train ensemble models
    print("Step 3: Training ensemble models...")
    models = trainer.train_ensemble_models(combined_data)
    
    # Step 4: Save models
    print("Step 4: Saving pre-trained models...")
    model_file = trainer.save_pretrained_models(symbol)
    
    print(f"\nPre-training complete!")
    print(f"Models saved to: {model_file}")
    print(f"Total features: {len(combined_data.columns)}")
    print(f"Training samples: {len(combined_data)}")

if __name__ == "__main__":
    main()