import pandas as pd
import numpy as np
from onnx_model import ONNXPricePredictor

class SignalGenerator:
    def __init__(self, predictor):
        self.predictor = predictor
        self.min_confidence = 0.6
        
    def generate_signals(self, df):
        signals = []
        
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Get prediction
            features = self.prepare_features(df, i)
            if features is None:
                continue
                
            prediction, confidence = self.predictor.get_prediction_confidence(features, row['close'])
            
            # Generate signal based on model prediction and BB
            signal = self.evaluate_signal(row, prediction, confidence)
            
            if signal['action'] != 'HOLD':
                signals.append(signal)
        
        return signals
    
    def prepare_features(self, df, index, lookback=10):
        if index < lookback:
            return None
        
        feature_names = self.predictor.feature_names
        
        features = []
        for j in range(lookback):
            idx = index - lookback + j
            for feature in feature_names:
                features.append(df.iloc[idx][feature])
        
        return np.array(features)
    
    def evaluate_signal(self, row, prediction, confidence):
        signal = {
            'timestamp': row['time'],
            'action': 'HOLD',
            'confidence': float(confidence),
            'prediction': float(prediction),
            'current_price': float(row['close']),
            'sl': None,
            'tp': None,
            'lot_size': 0.01
        }
        
        # Check confidence threshold
        if confidence < self.min_confidence:
            return signal
        
        # Price prediction direction
        price_direction = 1 if prediction > row['close'] else -1
        
        # Combine with BB signal
        if row['bb_signal'] == 1 and price_direction == 1:
            signal['action'] = 'BUY'
            signal['tp'] = float(prediction)
            signal['sl'] = float(row['close'] - (2 * row['atr']))
            signal['lot_size'] = float(self.calculate_lot_size(confidence))
            
        elif row['bb_signal'] == -1 and price_direction == -1:
            signal['action'] = 'SELL'
            signal['tp'] = float(prediction)
            signal['sl'] = float(row['close'] + (2 * row['atr']))
            signal['lot_size'] = float(self.calculate_lot_size(confidence))
        
        return signal

    def calculate_lot_size(self, confidence):
        # Dynamic lot sizing based on confidence
        base_lot = 0.01
        if confidence > 0.9:
            return base_lot * 3
        elif confidence > 0.8:
            return base_lot * 2
        else:
            return base_lot