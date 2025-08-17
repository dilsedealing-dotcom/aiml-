//+------------------------------------------------------------------+
//|                                                  AITradingEA.mq5 |
//|                                          AI Trading System v2.0  |
//+------------------------------------------------------------------+
#property copyright "AI Trading System"
#property version   "2.00"

#include <Trade\Trade.mqh>
#include <JAson.mqh>

// Input Parameters
input string WebSocketURL = "ws://localhost:8765";
input double MaxSpread = 3.0;
input double RiskPercent = 2.0;
input int TrailingStart = 550;
input int TrailingStep = 220;
input bool EnableTrailing = true;
input double MinConfidence = 0.7;
input int MaxPositions = 3;
input bool EnableAutoTrading = true;

// Global Variables
CTrade trade;
int websocket_handle = INVALID_HANDLE;
bool connected = false;
datetime last_signal_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    Print("AI Trading EA v2.0 Starting...");
    
    // Initialize WebSocket connection
    if(!ConnectWebSocket()) {
        Print("Failed to connect to WebSocket server");
        return INIT_FAILED;
    }
    
    Print("EA initialized successfully");
    Print("WebSocket URL: ", WebSocketURL);
    Print("Risk per trade: ", RiskPercent, "%");
    Print("Max spread: ", MaxSpread, " points");
    
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    if(websocket_handle != INVALID_HANDLE) {
        SocketClose(websocket_handle);
    }
    Print("AI Trading EA stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    // Check WebSocket connection
    if(!connected) {
        static datetime last_reconnect = 0;
        if(TimeCurrent() - last_reconnect > 30) { // Try reconnect every 30 seconds
            ConnectWebSocket();
            last_reconnect = TimeCurrent();
        }
        return;
    }
    
    // Read WebSocket messages
    string message = ReadWebSocketMessage();
    if(message != "") {
        ProcessTradingSignal(message);
    }
    
    // Handle trailing stops
    if(EnableTrailing) {
        HandleTrailingStops();
    }
    
    // Risk management check
    CheckRiskLimits();
}

//+------------------------------------------------------------------+
//| Connect to WebSocket server                                      |
//+------------------------------------------------------------------+
bool ConnectWebSocket() {
    string host = "localhost";
    int port = 8765;
    
    websocket_handle = SocketCreate();
    if(websocket_handle == INVALID_HANDLE) {
        Print("Failed to create socket");
        return false;
    }
    
    if(!SocketConnect(websocket_handle, host, port, 5000)) {
        Print("Failed to connect to WebSocket server");
        SocketClose(websocket_handle);
        websocket_handle = INVALID_HANDLE;
        return false;
    }
    
    connected = true;
    Print("Connected to WebSocket server: ", host, ":", port);
    return true;
}

//+------------------------------------------------------------------+
//| Read WebSocket message                                           |
//+------------------------------------------------------------------+
string ReadWebSocketMessage() {
    if(websocket_handle == INVALID_HANDLE || !connected) return "";
    
    string message = "";
    uint len;
    uchar buffer[];
    
    if(SocketIsReadable(websocket_handle)) {
        len = SocketRead(websocket_handle, buffer, 1024, 100);
        if(len > 0) {
            message = CharArrayToString(buffer, 0, len);
        }
    }
    
    return message;
}

//+------------------------------------------------------------------+
//| Process trading signal from WebSocket                           |
//+------------------------------------------------------------------+
void ProcessTradingSignal(string jsonMessage) {
    if(!EnableAutoTrading) {
        Print("Auto trading disabled - signal ignored");
        return;
    }
    
    // Parse JSON signal
    CJAVal signal;
    if(!signal.Deserialize(jsonMessage)) {
        Print("Failed to parse JSON signal: ", jsonMessage);
        return;
    }
    
    // Extract signal data
    string action = signal["action"].ToStr();
    double confidence = signal["confidence"].ToDbl();
    double prediction = signal["prediction"].ToDbl();
    double current_price = signal["current_price"].ToDbl();
    double sl = signal["sl"].ToDbl();
    double tp = signal["tp"].ToDbl();
    double lot_size = signal["lot_size"].ToDbl();
    
    Print("Signal received: ", action, " Confidence: ", confidence);
    
    // Validate signal
    if(!ValidateSignal(action, confidence, lot_size)) {
        return;
    }
    
    // Check spread
    if(!CheckSpread()) {
        Print("Spread too high - signal rejected");
        return;
    }
    
    // Execute trade
    if(action == "BUY") {
        ExecuteBuyOrder(lot_size, sl, tp, confidence);
    } else if(action == "SELL") {
        ExecuteSellOrder(lot_size, sl, tp, confidence);
    }
    
    last_signal_time = TimeCurrent();
}

//+------------------------------------------------------------------+
//| Validate trading signal                                          |
//+------------------------------------------------------------------+
bool ValidateSignal(string action, double confidence, double lot_size) {
    // Check confidence threshold
    if(confidence < MinConfidence) {
        Print("Signal confidence too low: ", confidence, " < ", MinConfidence);
        return false;
    }
    
    // Check if too many positions
    if(PositionsTotal() >= MaxPositions) {
        Print("Maximum positions reached: ", PositionsTotal());
        return false;
    }
    
    // Check lot size
    if(lot_size <= 0 || lot_size > 10) {
        Print("Invalid lot size: ", lot_size);
        return false;
    }
    
    // Check signal timing (avoid duplicate signals)
    if(TimeCurrent() - last_signal_time < 10) {
        Print("Signal too frequent - ignored");
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Check spread filter                                              |
//+------------------------------------------------------------------+
bool CheckSpread() {
    double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
    return spread <= MaxSpread * _Point;
}

//+------------------------------------------------------------------+
//| Execute buy order                                                |
//+------------------------------------------------------------------+
void ExecuteBuyOrder(double lot_size, double sl, double tp, double confidence) {
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    
    // Calculate risk-adjusted lot size
    double adjusted_lot = CalculateLotSize(lot_size, confidence);
    
    // Use ATR-based SL/TP if not provided
    double stop_loss = (sl > 0) ? sl : ask - GetATR() * 2;
    double take_profit = (tp > 0) ? tp : ask + GetATR() * 3;
    
    string comment = StringFormat("AI_BUY_%.2f", confidence);
    
    if(trade.Buy(adjusted_lot, _Symbol, ask, stop_loss, take_profit, comment)) {
        Print("BUY order executed: Lot=", adjusted_lot, " SL=", stop_loss, " TP=", take_profit);
    } else {
        Print("BUY order failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
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
    double stop_loss = (sl > 0) ? sl : bid + GetATR() * 2;
    double take_profit = (tp > 0) ? tp : bid - GetATR() * 3;
    
    string comment = StringFormat("AI_SELL_%.2f", confidence);
    
    if(trade.Sell(adjusted_lot, _Symbol, bid, stop_loss, take_profit, comment)) {
        Print("SELL order executed: Lot=", adjusted_lot, " SL=", stop_loss, " TP=", take_profit);
    } else {
        Print("SELL order failed: ", trade.ResultRetcode(), " - ", trade.ResultRetcodeDescription());
    }
}

//+------------------------------------------------------------------+
//| Calculate risk-adjusted lot size                                 |
//+------------------------------------------------------------------+
double CalculateLotSize(double signal_lot, double confidence) {
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amount = balance * RiskPercent / 100;
    
    // Base lot size calculation
    double base_lot = risk_amount / 1000; // Simplified for XAUUSD
    
    // Adjust based on confidence
    double confidence_multiplier = MathMin(confidence * 2, 2.0); // Max 2x multiplier
    double adjusted_lot = base_lot * confidence_multiplier;
    
    // Use signal lot size but cap it
    adjusted_lot = MathMin(adjusted_lot, signal_lot);
    
    // Ensure minimum and maximum lot sizes
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
        if(PositionSelectByTicket(PositionGetTicket(i))) {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol) {
                string comment = PositionGetString(POSITION_COMMENT);
                if(StringFind(comment, "AI_") == 0) { // Only trail AI positions
                    UpdateTrailingStop(PositionGetTicket(i));
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Update trailing stop for a position                             |
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
                trade.PositionModify(ticket, new_sl, current_tp);
                Print("Trailing stop updated for BUY position: ", new_sl);
            }
        }
    } else if(pos_type == POSITION_TYPE_SELL) {
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double profit_points = (open_price - ask) / _Point;
        
        if(profit_points >= TrailingStart) {
            double new_sl = ask + TrailingStep * _Point;
            if(new_sl < current_sl || current_sl == 0) {
                trade.PositionModify(ticket, new_sl, current_tp);
                Print("Trailing stop updated for SELL position: ", new_sl);
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Check risk limits                                                |
//+------------------------------------------------------------------+
void CheckRiskLimits() {
    double total_risk = 0;
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    
    // Calculate total exposure
    for(int i = 0; i < PositionsTotal(); i++) {
        if(PositionSelectByTicket(PositionGetTicket(i))) {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol) {
                double volume = PositionGetDouble(POSITION_VOLUME);
                total_risk += volume * 1000; // Simplified risk calculation
            }
        }
    }
    
    double max_risk = balance * RiskPercent * MaxPositions / 100;
    
    if(total_risk > max_risk) {
        Print("WARNING: Total risk exceeded: ", total_risk, " > ", max_risk);
        // Could implement emergency position closure here
    }
}