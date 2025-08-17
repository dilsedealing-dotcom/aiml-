import asyncio
import websockets
import json
from datetime import datetime

class SignalSender:
    def __init__(self, url="ws://localhost:8765"):
        self.url = url
        
    async def send_signal(self, signal):
        try:
            async with websockets.connect(self.url) as websocket:
                # Format signal for EA
                signal_data = {
                    "action": signal['action'],
                    "symbol": "XAUUSD",
                    "confidence": float(signal['confidence']),
                    "prediction": float(signal['prediction']),
                    "current_price": float(signal['current_price']),
                    "sl": float(signal['sl']) if signal['sl'] else None,
                    "tp": float(signal['tp']) if signal['tp'] else None,
                    "lot_size": float(signal['lot_size']),
                    "timestamp": datetime.now().isoformat()
                }
                
                message = json.dumps(signal_data)
                await websocket.send(message)
                print(f"✓ Signal sent: {signal['action']} - Confidence: {signal['confidence']:.3f}")
                
                # Wait for acknowledgment
                response = await websocket.recv()
                print(f"Server response: {response}")
                
        except Exception as e:
            print(f"✗ Failed to send signal: {e}")
            
    def send_signal_sync(self, signal):
        """Synchronous wrapper for sending signals"""
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.send_signal(signal))
        except Exception as e:
            print(f"Signal sending error: {e}")