import pandas as pd
import numpy as np
import ta

class TechnicalIndicators:
    @staticmethod
    def add_bollinger_bands(df, period=20, std=2):
        df['bb_upper'] = ta.volatility.bollinger_hband(df['close'], window=period, window_dev=std)
        df['bb_middle'] = ta.volatility.bollinger_mavg(df['close'], window=period)
        df['bb_lower'] = ta.volatility.bollinger_lband(df['close'], window=period, window_dev=std)
        df['bb_signal'] = 0
        df.loc[df['close'] > df['bb_upper'], 'bb_signal'] = -1
        df.loc[df['close'] < df['bb_lower'], 'bb_signal'] = 1
        df.loc[(df['close'].shift(1) < df['bb_middle']) & (df['close'] > df['bb_middle']), 'bb_signal'] = 1
        df.loc[(df['close'].shift(1) > df['bb_middle']) & (df['close'] < df['bb_middle']), 'bb_signal'] = -1
        return df
    
    @staticmethod
    def add_macd(df, fast=12, slow=26, signal=9):
        df['macd'] = ta.trend.macd_diff(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        df['macd_signal'] = ta.trend.macd_signal(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        df['macd_histogram'] = ta.trend.macd_diff(df['close'], window_fast=fast, window_slow=slow, window_sign=signal)
        df['macd_buy'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_sell'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        return df
    
    @staticmethod
    def add_rsi(df, period=14):
        df['rsi'] = ta.momentum.rsi(df['close'], window=period)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        return df
    
    @staticmethod
    def add_adx(df, period=14):
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], window=period)
        df['adx_strong'] = (df['adx'] > 25).astype(int)
        return df
    
    @staticmethod
    def add_atr(df, period=14):
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=period)
        return df
    
    @staticmethod
    def add_trading_sessions(df):
        df['hour'] = df['time'].dt.hour
        df['london_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
        df['newyork_session'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
        df['overlap_session'] = ((df['hour'] >= 13) & (df['hour'] < 16)).astype(int)
        return df
    
    @staticmethod
    def calculate_all_indicators(df):
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_adx(df)
        df = TechnicalIndicators.add_atr(df)
        df = TechnicalIndicators.add_trading_sessions(df)
        return df