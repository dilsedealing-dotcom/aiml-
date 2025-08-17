import asyncio
import websockets
import json
from datetime import datetime

class SimpleWebSocketServer:
    def __init__(self, host="localhost", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        
    async def register(self, websocket):
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        
    async def unregister(self, websocket):
        self.clients.remove(websocket)
        print(f"Client disconnected. Total clients: {len(self.clients)}")
        
    async def handle_client(self, websocket, path):
        await self.register(websocket)
        try:
            async for message in websocket:
                print(f"Received: {message}")
                # Echo back to all clients (including EA)
                await self.broadcast(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister(websocket)
            
    async def broadcast(self, message):
        if self.clients:
            await asyncio.gather(
                *[client.send(message) for client in self.clients],
                return_exceptions=True
            )
            
    async def send_signal(self, signal_data):
        message = json.dumps(signal_data)
        await self.broadcast(message)
        
    def start_server(self):
        print(f"Starting WebSocket server on ws://{self.host}:{self.port}")
        print("EA should connect to this URL")
        
        start_server = websockets.serve(
            self.handle_client, 
            self.host, 
            self.port
        )
        
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

if __name__ == "__main__":
    server = SimpleWebSocketServer()
    server.start_server()