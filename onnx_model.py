import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import pickle

class ONNXPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.onnx_model = None
        self.session = None
        self.feature_names = [
            'open', 'high', 'low', 'close', 'tick_volume',
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_signal',
            'macd_signal',
            'rsi_overbought',
            'adx', 'adx_strong', 'atr'
        ]
        
    def prepare_data(self, df, target_col='close', lookback=10):
        # Create features for prediction
        features = []
        targets = []
        
        for i in range(lookback, len(df)):
            # Use previous lookback periods as features
            feature_row = []
            for j in range(lookback):
                idx = i - lookback + j
                for feature in self.feature_names:
                    feature_row.append(df.iloc[idx][feature])
            
            features.append(feature_row)
            targets.append(df.iloc[i][target_col])
        
        return np.array(features), np.array(targets)
    
    def train_model(self, df, test_size=0.2):
        X, y = self.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance - MSE: {mse:.6f}, R2: {r2:.4f}")
        
        # Convert to ONNX
        self.convert_to_onnx(X_train_scaled)
        
        return mse, r2
    
    def convert_to_onnx(self, X_sample):
        # Define input type
        initial_type = [('float_input', FloatTensorType([None, X_sample.shape[1]]))]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(self.model, initial_types=initial_type)
        
        # Save ONNX model
        with open('onnx_models/price_predictor.onnx', 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        # Save scaler
        with open('onnx_models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Create ONNX runtime session
        self.session = ort.InferenceSession('onnx_models/price_predictor.onnx')
        
        print("Model converted to ONNX successfully")
    
    def predict(self, features):
        if self.session is None:
            self.load_onnx_model()
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict using ONNX
        input_name = self.session.get_inputs()[0].name
        prediction = self.session.run(None, {input_name: features_scaled.astype(np.float32)})
        
        # Ensure scalar return value
        result = prediction[0]
        if isinstance(result, (list, np.ndarray)):
            if hasattr(result, '__len__') and len(result) > 0:
                return float(result[0])
            else:
                return float(result)
        return float(result)
    
    def load_onnx_model(self):
        self.session = ort.InferenceSession('onnx_models/price_predictor.onnx')
        with open('onnx_models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
    
    def get_prediction_confidence(self, features, current_price):
        prediction = self.predict(features)
        # Ensure scalar values
        if isinstance(prediction, np.ndarray):
            prediction = float(prediction[0]) if len(prediction) > 0 else float(prediction)
        else:
            prediction = float(prediction)
        
        current_price = float(current_price)
        confidence = abs(prediction - current_price) / current_price
        return prediction, float(1 - confidence)  # Higher confidence for closer predictions
