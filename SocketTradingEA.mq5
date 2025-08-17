//+------------------------------------------------------------------+
//|                                              SocketTradingEA.mq5 |
//|                                    Socket-based Trading EA       |
//+------------------------------------------------------------------+
#property copyright "AI Trading System"
#property version   "1.00"

#include <Trade\Trade.mqh>

// Input Parameters
input string ServerHost = "127.0.0.1";
input int ServerPort = 8765;
input double RiskPercent = 2.0;
input double MaxSpread = 3.0;
input int TrailingStart = 550;
input int TrailingStep = 220;
input double MinConfidence = 0.7;
input int MaxPositions = 3;
input bool EnableAutoTrading = true;

// Global Variables
CTrade trade;
int socket_handle = INVALID_HANDLE;
bool is_connected = false;
datetime last_connection_attempt = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    Print("Socket Trading EA v1.0 Starting...");
    Print("Server: ", ServerHost, ":", ServerPort);
    Print("Risk per trade: ", RiskPercent, "%");
    
    ConnectToServer();
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    if(socket_handle != INVALID_HANDLE) {
        SocketClose(socket_handle);
        socket_handle = INVALID_HANDLE;
    }
    Print("Socket Trading EA stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    // Try to reconnect if disconnected
    if(!is_connected) {
        if(TimeCurrent() - last_connection_attempt > 30) { // Try every 30 seconds
            ConnectToServer();
            last_connection_attempt = TimeCurrent();
        }
        return;
    }
    
    // Check for incoming messages
    CheckForSignals();
    
    // Handle trailing stops
    HandleTrailingStops();
}

//+------------------------------------------------------------------+
//| Connect to WebSocket server                                      |
//+------------------------------------------------------------------+
void ConnectToServer() {
    if(socket_handle != INVALID_HANDLE) {
        SocketClose(socket_handle);
    }
    
    socket_handle = SocketCreate();
    if(socket_handle == INVALID_HANDLE) {
        Print("Failed to create socket");
        return;
    }
    
    if(SocketConnect(socket_handle, ServerHost, ServerPort, 5000)) {
        is_connected = true;
        Print("Connected to server successfully");
        
        // Send WebSocket handshake
        string handshake = "GET / HTTP/1.1\r\n";
        handshake += "Host: " + ServerHost + ":" + IntegerToString(ServerPort) + "\r\n";
        handshake += "Upgrade: websocket\r\n";
        handshake += "Connection: Upgrade\r\n";
        handshake += "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n";
        handshake += "Sec-WebSocket-Version: 13\r\n\r\n";
        
        uchar request[];
        StringToCharArray(handshake, request, 0, StringLen(handshake));
        SocketSend(socket_handle, request, ArraySize(request));
        
    } else {
        Print("Failed to connect to server");
        SocketClose(socket_handle);
        socket_handle = INVALID_HANDLE;
        is_connected = false;
    }
}

//+------------------------------------------------------------------+
//| Check for incoming signals                                       |
//+------------------------------------------------------------------+
void CheckForSignals() {
    if(socket_handle == INVALID_HANDLE || !is_connected) return;
    
    if(SocketIsReadable(socket_handle)) {
        uchar buffer[];
        uint len = SocketRead(socket_handle, buffer, 1024, 100);
        
        if(len > 0) {
            string message = CharArrayToString(buffer, 0, len);
            ProcessMessage(message);
        } else {
            // Connection lost
            is_connected = false;
            Print("Connection lost");
        }
    }
}

//+------------------------------------------------------------------+
//| Process incoming message                                         |
//+------------------------------------------------------------------+
void ProcessMessage(string message) {
    // Skip HTTP headers if present
    int json_start = StringFind(message, "{");
    if(json_start >= 0) {
        message = StringSubstr(message, json_start);
    }
    
    Print("Received: ", message);
    
    // Parse signal
    if(StringFind(message, "\"action\"") >= 0) {
        ProcessTradingSignal(message);
    }
}

//+------------------------------------------------------------------+
//| Process trading signal                                           |
//+------------------------------------------------------------------+
void ProcessTradingSignal(string json_message) {
    if(!EnableAutoTrading) {
        Print("Auto trading disabled");
        return;
    }
    
    // Extract signal data using simple string parsing
    string action = ExtractStringValue(json_message, "action");
    double confidence = ExtractDoubleValue(json_message, "confidence");
    double lot_size = ExtractDoubleValue(json_message, "lot_size");
    double sl = ExtractDoubleValue(json_message, "sl");
    double tp = ExtractDoubleValue(json_message, "tp");
    
    Print("Signal: ", action, " Confidence: ", confidence, " Lot: ", lot_size);
    
    // Validate signal
    if(!ValidateSignal(action, confidence, lot_size)) {
        return;
    }
    
    // Execute trade
    if(action == "BUY") {
        ExecuteBuyOrder(lot_size, sl, tp, confidence);
    } else if(action == "SELL") {
        ExecuteSellOrder(lot_size, sl, tp, confidence);
    }
}

//+------------------------------------------------------------------+
//| Extract string value from JSON                                   |
//+------------------------------------------------------------------+
string ExtractStringValue(string json, string key) {
    string search_key = "\"" + key + "\":\"";
    int start = StringFind(json, search_key);
    if(start < 0) return "";
    
    start += StringLen(search_key);
    int end = StringFind(json, "\"", start);
    if(end < 0) return "";
    
    return StringSubstr(json, start, end - start);
}

//+------------------------------------------------------------------+
//| Extract double value from JSON                                   |
//+------------------------------------------------------------------+
double ExtractDoubleValue(string json, string key) {
    string search_key = "\"" + key + "\":";
    int start = StringFind(json, search_key);
    if(start < 0) return 0.0;
    
    start += StringLen(search_key);
    int end = StringFind(json, ",", start);
    if(end < 0) end = StringFind(json, "}", start);
    if(end < 0) return 0.0;
    
    string value_str = StringSubstr(json, start, end - start);
    StringReplace(value_str, " ", "");
    
    return StringToDouble(value_str);
}

//+------------------------------------------------------------------+
//| Validate trading signal                                          |
//+------------------------------------------------------------------+
bool ValidateSignal(string action, double confidence, double lot_size) {
    // Check action
    if(action != "BUY" && action != "SELL") {
        Print("Invalid action: ", action);
        return false;
    }
    
    // Check confidence
    if(confidence < MinConfidence) {
        Print("Low confidence: ", confidence, " < ", MinConfidence);
        return false;
    }
    
    // Check positions limit
    if(PositionsTotal() >= MaxPositions) {
        Print("Max positions reached: ", PositionsTotal());
        return false;
    }
    
    // Check spread
    double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
    if(spread > MaxSpread * _Point) {
        Print("High spread: ", spread, " > ", MaxSpread * _Point);
        return false;
    }
    
    // Check lot size
    if(lot_size <= 0) {
        Print("Invalid lot size: ", lot_size);
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Execute buy order                                                |
//+------------------------------------------------------------------+
void ExecuteBuyOrder(double lot_size, double sl, double tp, double confidence) {
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    
    // Calculate risk-adjusted lot size
    double adjusted_lot = CalculateLotSize(lot_size, confidence);
    
    // Use ATR-based SL/TP if not provided
    double atr = GetATR();
    double stop_loss = (sl > 0) ? sl : ask - atr * 2;
    double take_profit = (tp > 0) ? tp : ask + atr * 3;
    
    string comment = "AI_BUY_" + DoubleToString(confidence, 2);
    
    if(trade.Buy(adjusted_lot, _Symbol, ask, stop_loss, take_profit, comment)) {
        Print("✓ BUY executed: ", adjusted_lot, " lots at ", ask);
    } else {
        Print("✗ BUY failed: ", trade.ResultRetcode());
    }
}

//+------------------------------------------------------------------+
//| Execute sell order                                               |
//+------------------------------------------------------------------+
void ExecuteSellOrder(double lot_size, double sl, double tp, double confidence) {
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    
    // Calculate risk-adjusted lot size
    double adjusted_lot = CalculateLotSize(lot_size, confidence);
    
    // Use ATR-based SL/TP if not provided
    double atr = GetATR();
    double stop_loss = (sl > 0) ? sl : bid + atr * 2;
    double take_profit = (tp > 0) ? tp : bid - atr * 3;
    
    string comment = "AI_SELL_" + DoubleToString(confidence, 2);
    
    if(trade.Sell(adjusted_lot, _Symbol, bid, stop_loss, take_profit, comment)) {
        Print("✓ SELL executed: ", adjusted_lot, " lots at ", bid);
    } else {
        Print("✗ SELL failed: ", trade.ResultRetcode());
    }
}

//+------------------------------------------------------------------+
//| Calculate risk-adjusted lot size                                 |
//+------------------------------------------------------------------+
double CalculateLotSize(double signal_lot, double confidence) {
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amount = balance * RiskPercent / 100;
    
    // Base lot calculation (simplified for XAUUSD)
    double base_lot = risk_amount / 1000;
    
    // Adjust based on confidence (higher confidence = larger position)
    double confidence_multiplier = MathMin(confidence * 1.5, 2.0);
    double adjusted_lot = base_lot * confidence_multiplier;
    
    // Don't exceed signal lot size
    adjusted_lot = MathMin(adjusted_lot, signal_lot);
    
    // Ensure within broker limits
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    
    adjusted_lot = MathMax(adjusted_lot, min_lot);
    adjusted_lot = MathMin(adjusted_lot, max_lot);
    
    return NormalizeDouble(adjusted_lot, 2);
}

//+------------------------------------------------------------------+
//| Get ATR value for dynamic SL/TP                                 |
//+------------------------------------------------------------------+
double GetATR() {
    int atr_handle = iATR(_Symbol, PERIOD_CURRENT, 14);
    if(atr_handle == INVALID_HANDLE) return 0.001;
    
    double atr_buffer[];
    ArraySetAsSeries(atr_buffer, true);
    
    if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) > 0) {
        IndicatorRelease(atr_handle);
        return atr_buffer[0];
    }
    
    IndicatorRelease(atr_handle);
    return 0.001; // Default fallback
}

//+------------------------------------------------------------------+
//| Handle trailing stops                                            |
//+------------------------------------------------------------------+
void HandleTrailingStops() {
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket)) {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol) {
                string comment = PositionGetString(POSITION_COMMENT);
                if(StringFind(comment, "AI_") == 0) {
                    UpdateTrailingStop(ticket);
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Update trailing stop for position                               |
//+------------------------------------------------------------------+
void UpdateTrailingStop(ulong ticket) {
    if(!PositionSelectByTicket(ticket)) return;
    
    double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
    double current_sl = PositionGetDouble(POSITION_SL);
    double current_tp = PositionGetDouble(POSITION_TP);
    
    ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
    
    if(pos_type == POSITION_TYPE_BUY) {
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double profit_points = (bid - open_price) / _Point;
        
        if(profit_points >= TrailingStart) {
            double new_sl = bid - TrailingStep * _Point;
            if(new_sl > current_sl) {
                if(trade.PositionModify(ticket, new_sl, current_tp)) {
                    Print("Trailing stop updated for BUY: ", new_sl);
                }
            }
        }
    }
    else if(pos_type == POSITION_TYPE_SELL) {
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double profit_points = (open_price - ask) / _Point;
        
        if(profit_points >= TrailingStart) {
            double new_sl = ask + TrailingStep * _Point;
            if(new_sl < current_sl || current_sl == 0) {
                if(trade.PositionModify(ticket, new_sl, current_tp)) {
                    Print("Trailing stop updated for SELL: ", new_sl);
                }
            }
        }
    }
}