from flask import Flask, jsonify
from flask_socketio import SocketIO, emit
import threading
import time
import schedule
from data_processor import DataProcessor
from onnx_model import ONNXPricePredictor
from signal_generator import SignalGenerator
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'trading_secret_key'
socketio = SocketIO(app, cors_allowed_origins="*")

class TradingPipeline:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.predictor = ONNXPricePredictor()
        self.signal_generator = SignalGenerator()
        self.is_running = False
        
    def run_pipeline(self):
        try:
            # Download and process data
            df = self.data_processor.download_and_process_data()
            if df is None:
                return
            
            # Train/retrain model
            mse, r2 = self.predictor.train_model(df)
            
            # Generate signals
            signals = self.signal_generator.generate_signals(df)
            
            # Emit signals via WebSocket
            for signal in signals:
                signal_data = {
                    'timestamp': signal['timestamp'].isoformat(),
                    'action': signal['action'],
                    'confidence': float(signal['confidence']),
                    'prediction': float(signal['prediction']),
                    'current_price': float(signal['current_price']),
                    'sl': float(signal['sl']) if signal['sl'] else None,
                    'tp': float(signal['tp']) if signal['tp'] else None,
                    'lot_size': float(signal['lot_size'])
                }
                
                socketio.emit('trading_signal', signal_data)
                print(f"Signal emitted: {signal_data}")
            
        except Exception as e:
            print(f"Pipeline error: {e}")

pipeline = TradingPipeline()

@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'pipeline_running': pipeline.is_running})

@app.route('/start_pipeline')
def start_pipeline():
    if not pipeline.is_running:
        pipeline.is_running = True
        # Schedule pipeline to run every 30 seconds
        schedule.every(30).seconds.do(pipeline.run_pipeline)
        
        def run_scheduler():
            while pipeline.is_running:
                schedule.run_pending()
                time.sleep(1)
        
        scheduler_thread = threading.Thread(target=run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        return jsonify({'status': 'Pipeline started'})
    else:
        return jsonify({'status': 'Pipeline already running'})

@app.route('/stop_pipeline')
def stop_pipeline():
    pipeline.is_running = False
    schedule.clear()
    return jsonify({'status': 'Pipeline stopped'})

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('status', {'message': 'Connected to trading server'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)