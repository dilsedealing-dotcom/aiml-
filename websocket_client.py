import websocket
import json
import threading
import time

class WebSocketClient:
    def __init__(self, url="ws://localhost:5000/socket.io/?EIO=4&transport=websocket"):
        self.url = url
        self.ws = None
        self.connected = False
        
    def connect(self):
        try:
            self.ws = websocket.WebSocketApp(
                self.url,
                on_open=self.on_open,
                on_message=self.on_message,
                on_error=self.on_error,
                on_close=self.on_close
            )
            
            # Start WebSocket in a separate thread
            wst = threading.Thread(target=self.ws.run_forever)
            wst.daemon = True
            wst.start()
            
            # Wait for connection
            time.sleep(2)
            return self.connected
            
        except Exception as e:
            print(f"WebSocket connection failed: {e}")
            return False
    
    def on_open(self, ws):
        print("WebSocket connected")
        self.connected = True
    
    def on_message(self, ws, message):
        print(f"Received: {message}")
    
    def on_error(self, ws, error):
        print(f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        print("WebSocket disconnected")
        self.connected = False
    
    def send_signal(self, signal):
        if not self.connected:
            print("WebSocket not connected")
            return False
        
        try:
            # Format signal for EA
            signal_data = {
                "action": signal['action'],
                "symbol": "XAUUSD",
                "confidence": signal['confidence'],
                "prediction": signal['prediction'],
                "current_price": signal['current_price'],
                "sl": signal['sl'],
                "tp": signal['tp'],
                "lot_size": signal['lot_size'],
                "timestamp": signal['timestamp'].isoformat() if hasattr(signal['timestamp'], 'isoformat') else str(signal['timestamp'])
            }
            
            message = json.dumps(signal_data)
            self.ws.send(message)
            print(f"Signal sent: {signal['action']} - Confidence: {signal['confidence']:.3f}")
            return True
            
        except Exception as e:
            print(f"Failed to send signal: {e}")
            return False
    
    def disconnect(self):
        if self.ws:
            self.ws.close()
            self.connected = False