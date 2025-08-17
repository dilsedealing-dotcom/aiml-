import pandas as pd
import numpy as np
import os
from mt5_connector import MT5Connector
from technical_indicators import TechnicalIndicators
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.mt5 = MT5Connector()
        
    def download_and_process_data(self, symbol="XAUUSD", count=10000, file_path=None):
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
        else:
            if file_path:
                print(f"File {file_path} not found. Downloading from MT5...")
            self.mt5.connect()
            df = self.mt5.get_data(symbol, count=count)
            if df is None:
                return None
        
        # Add technical indicators
        df = TechnicalIndicators.calculate_all_indicators(df)
        
        # Add SL/TP patterns
        df = self.add_sl_tp_patterns(df)
        
        # Clean data
        df = self.clean_data(df)
        
        # Save processed data
        df.to_csv(f'data/{symbol}_processed.csv', index=False)
        return df
    
    def add_sl_tp_patterns(self, df):
        # Calculate consecutive losses/wins
        df['price_change'] = df['close'].pct_change()
        df['win'] = (df['price_change'] > 0).astype(int)
        df['loss'] = (df['price_change'] < 0).astype(int)
        
        # Consecutive patterns
        df['consecutive_losses'] = 0
        df['consecutive_wins'] = 0
        
        for i in range(1, len(df)):
            if df.iloc[i]['loss'] == 1:
                if df.iloc[i-1]['loss'] == 1:
                    df.iloc[i, df.columns.get_loc('consecutive_losses')] = df.iloc[i-1]['consecutive_losses'] + 1
                else:
                    df.iloc[i, df.columns.get_loc('consecutive_losses')] = 1
            
            if df.iloc[i]['win'] == 1:
                if df.iloc[i-1]['win'] == 1:
                    df.iloc[i, df.columns.get_loc('consecutive_wins')] = df.iloc[i-1]['consecutive_wins'] + 1
                else:
                    df.iloc[i, df.columns.get_loc('consecutive_wins')] = 1
        
        # Pattern recognition
        df['third_trade_pattern'] = ((df['consecutive_losses'] >= 2) & (df['consecutive_losses'].shift(1) < 2)).astype(int)
        
        return df
    
    def clean_data(self, df):
        # Remove unnecessary columns
        columns_to_keep = [
            'time', 'open', 'high', 'low', 'close', 'tick_volume',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_signal',
            'macd_signal',
            'rsi_overbought',
            'adx', 'adx_strong', 'atr'
        ]
        
        df = df[columns_to_keep].copy()
        df = df.dropna()
        return df
    
    def analyze_correlations(self, df):
        """Analyze correlations between features"""
        features = ['close', 'open', 'high', 'low', 'bb_upper', 'bb_middle', 'bb_lower', 
                   'atr', 'tick_volume', 'adx', 'adx_strong', 'rsi_overbought', 'macd_signal']
        
        # Ensure all features exist
        available_features = [f for f in features if f in df.columns]
        
        if len(available_features) < 2:
            print("Not enough features for correlation analysis")
            return None
            
        correlation_matrix = df[available_features].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.3f')
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.savefig('data/correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return correlation_matrix
    
    def feature_selection(self, df, threshold=0.05):
        correlation_matrix = self.analyze_correlations(df)
        close_correlations = correlation_matrix['close'].abs()
        
        # Select features with correlation > threshold
        selected_features = close_correlations[close_correlations > threshold].index.tolist()
        selected_features.remove('close')  # Remove target variable
        
        print(f"Selected features: {selected_features}")
        return selected_features