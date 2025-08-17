import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz

class MT5Connector:
    def __init__(self):
        self.connected = False
        
    def connect(self):
        if not mt5.initialize():
            print("MT5 initialization failed")
            return False
        self.connected = True
        print("MT5 connected successfully")
        return True
    
    def disconnect(self):
        mt5.shutdown()
        self.connected = False
    
    def get_data(self, symbol="XAUUSD", timeframe=mt5.TIMEFRAME_M5, count=10000):
        if not self.connected:
            self.connect()
        
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            print(f"Failed to get data for {symbol}")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    
    def get_account_info(self):
        return mt5.account_info()
    
    def send_order(self, symbol, order_type, volume, price=None, sl=None, tp=None, comment=""):
        if not self.connected:
            return False
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price or mt5.symbol_info_tick(symbol).ask,
            "sl": sl,
            "tp": tp,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result