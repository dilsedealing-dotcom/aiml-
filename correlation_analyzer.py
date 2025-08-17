import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    def __init__(self):
        self.feature_importance = {}
        self.correlation_matrix = None
        self.bb_model = None
        
    def analyze_bb_correlations(self, df):
        """Analyze correlations between features and BB signals"""
        
        # Create BB signal strength indicator
        df['bb_signal_strength'] = self.calculate_bb_signal_strength(df)
        
        # Define features for analysis
        features = ['close', 'low', 'high', 'open', 'bb_middle', 'bb_lower', 
                   'bb_upper', 'atr', 'tick_volume', 'adx', 'adx_strong', 
                   'rsi_overbought', 'macd_signal']
        
        # Calculate correlation matrix
        correlation_data = df[features + ['bb_signal_strength']].corr()
        self.correlation_matrix = correlation_data
        
        # Get correlations with BB signal strength
        bb_correlations = correlation_data['bb_signal_strength'].drop('bb_signal_strength')
        
        print("Feature Correlations with BB Signal Strength:")
        print("=" * 50)
        for feature, corr in bb_correlations.sort_values(key=abs, ascending=False).items():
            print(f"{feature:15}: {corr:8.4f}")
        
        return bb_correlations
    
    def calculate_bb_signal_strength(self, df):
        """Calculate BB signal strength based on price position relative to bands"""
        bb_width = df['bb_upper'] - df['bb_lower']
        price_position = (df['close'] - df['bb_lower']) / bb_width
        
        # Signal strength: higher when price is near bands
        signal_strength = np.where(
            price_position <= 0.2, 1 - price_position * 5,  # Near lower band
            np.where(
                price_position >= 0.8, (price_position - 0.8) * 5,  # Near upper band
                0  # Middle zone
            )
        )
        
        return signal_strength
    
    def train_bb_enhancement_model(self, df):
        """Train model to enhance BB signals using all features"""
        
        # Prepare features
        features = ['close', 'low', 'high', 'open', 'bb_middle', 'bb_lower', 
                   'bb_upper', 'atr', 'tick_volume', 'adx', 'adx_strong', 
                   'rsi_overbought', 'macd_signal']
        
        # Create enhanced BB signal as target
        df['bb_enhanced_signal'] = self.create_enhanced_bb_signal(df)
        
        X = df[features].fillna(0)
        y = df['bb_enhanced_signal'].fillna(0)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest model
        self.bb_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.bb_model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.bb_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        self.feature_importance = dict(zip(features, self.bb_model.feature_importances_))
        
        print(f"\nBB Enhancement Model Performance:")
        print(f"MSE: {mse:.6f}")
        print(f"R²: {r2:.4f}")
        
        print(f"\nFeature Importance for BB Enhancement:")
        print("=" * 40)
        for feature, importance in sorted(self.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True):
            print(f"{feature:15}: {importance:8.4f}")
        
        return self.bb_model
    
    def create_enhanced_bb_signal(self, df):
        """Create enhanced BB signal combining multiple factors"""
        
        # BB position
        bb_width = df['bb_upper'] - df['bb_lower']
        price_position = (df['close'] - df['bb_lower']) / bb_width
        
        # Volume confirmation
        volume_factor = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        
        # Trend strength
        trend_factor = df['adx'] / 100
        
        # Volatility factor
        volatility_factor = df['atr'] / df['close']
        
        # Enhanced signal
        enhanced_signal = (
            price_position * 0.4 +
            volume_factor * 0.2 +
            trend_factor * 0.2 +
            volatility_factor * 0.2
        )
        
        return enhanced_signal
    
    def predict_enhanced_bb_signal(self, df):
        """Predict enhanced BB signals using trained model"""
        if self.bb_model is None:
            raise ValueError("Model not trained. Call train_bb_enhancement_model first.")
        
        features = ['close', 'low', 'high', 'open', 'bb_middle', 'bb_lower', 
                   'bb_upper', 'atr', 'tick_volume', 'adx', 'adx_strong', 
                   'rsi_overbought', 'macd_signal']
        
        X = df[features].fillna(0)
        predictions = self.bb_model.predict(X)
        
        return predictions
    
    def plot_correlation_analysis(self, df):
        """Plot comprehensive correlation analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        
        # 1. Correlation heatmap
        features = ['close', 'bb_middle', 'bb_lower', 'bb_upper', 'atr', 
                   'tick_volume', 'adx', 'adx_strong', 'rsi_overbought', 'macd_signal']
        
        corr_matrix = df[features].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=axes[0,0], fmt='.3f')
        axes[0,0].set_title('Feature Correlation Matrix')
        
        # 2. Feature importance
        if self.feature_importance:
            features_imp = list(self.feature_importance.keys())
            importance_vals = list(self.feature_importance.values())
            
            axes[0,1].barh(features_imp, importance_vals)
            axes[0,1].set_title('Feature Importance for BB Enhancement')
            axes[0,1].set_xlabel('Importance')
        
        # 3. BB signal strength over time
        df['bb_signal_strength'] = self.calculate_bb_signal_strength(df)
        axes[1,0].plot(df.index[-1000:], df['bb_signal_strength'].iloc[-1000:])
        axes[1,0].set_title('BB Signal Strength Over Time (Last 1000 bars)')
        axes[1,0].set_ylabel('Signal Strength')
        
        # 4. Enhanced vs Original BB signals
        if self.bb_model is not None:
            enhanced_signals = self.predict_enhanced_bb_signal(df)
            original_signals = df['bb_signal_strength']
            
            sample_size = min(1000, len(df))
            sample_idx = df.index[-sample_size:]
            
            axes[1,1].plot(sample_idx, original_signals.iloc[-sample_size:], 
                          label='Original BB Signal', alpha=0.7)
            axes[1,1].plot(sample_idx, enhanced_signals[-sample_size:], 
                          label='Enhanced BB Signal', alpha=0.7)
            axes[1,1].set_title('Original vs Enhanced BB Signals')
            axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig('data/correlation_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_enhanced_trading_signals(self, df, threshold=0.7):
        """Generate trading signals using enhanced BB model"""
        
        if self.bb_model is None:
            raise ValueError("Model not trained. Call train_bb_enhancement_model first.")
        
        enhanced_signals = self.predict_enhanced_bb_signal(df)
        df['enhanced_bb_signal'] = enhanced_signals
        
        # Generate buy/sell signals
        signals = []
        
        for i in range(1, len(df)):
            current_signal = enhanced_signals[i]
            prev_signal = enhanced_signals[i-1]
            
            # Buy signal: enhanced signal crosses above threshold
            if current_signal > threshold and prev_signal <= threshold:
                signals.append({
                    'timestamp': df.iloc[i]['time'],
                    'action': 'BUY',
                    'confidence': current_signal,
                    'price': df.iloc[i]['close'],
                    'tp': df.iloc[i]['close'] * 1.01,
                    'sl': df.iloc[i]['close'] * 0.99,
                    'lot_size': 0.01
                })
            
            # Sell signal: enhanced signal crosses below -threshold
            elif current_signal < -threshold and prev_signal >= -threshold:
                signals.append({
                    'timestamp': df.iloc[i]['time'],
                    'action': 'SELL',
                    'confidence': abs(current_signal),
                    'price': df.iloc[i]['close'],
                    'tp': df.iloc[i]['close'] * 0.99,
                    'sl': df.iloc[i]['close'] * 1.01,
                    'lot_size': 0.01
                })
        
        return signals