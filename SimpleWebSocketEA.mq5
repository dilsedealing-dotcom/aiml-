//+------------------------------------------------------------------+
//|                                            SimpleWebSocketEA.mq5 |
//|                                          Simple WebSocket EA     |
//+------------------------------------------------------------------+
#property copyright "AI Trading System"
#property version   "1.00"

#include <Trade\Trade.mqh>

// Input Parameters
input string ServerHost = "localhost";
input int ServerPort = 8765;
input double RiskPercent = 2.0;
input double MaxSpread = 3.0;
input int TrailingStart = 550;
input int TrailingStep = 220;
input double MinConfidence = 0.7;
input int MaxPositions = 3;

// Global Variables
CTrade trade;
int socket_handle = INVALID_HANDLE;
bool is_connected = false;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    Print("Simple WebSocket EA Starting...");
    
    // Create socket
    socket_handle = SocketCreate();
    if(socket_handle == INVALID_HANDLE) {
        Print("Failed to create socket");
        return INIT_FAILED;
    }
    
    // Connect to server
    if(!SocketConnect(socket_handle, ServerHost, ServerPort, 5000)) {
        Print("Failed to connect to server: ", ServerHost, ":", ServerPort);
        SocketClose(socket_handle);
        return INIT_FAILED;
    }
    
    is_connected = true;
    Print("Connected to WebSocket server successfully");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    if(socket_handle != INVALID_HANDLE) {
        SocketClose(socket_handle);
    }
    Print("WebSocket EA stopped");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    if(!is_connected) return;
    
    // Check for incoming messages
    if(SocketIsReadable(socket_handle)) {
        string message = ReadMessage();
        if(message != "") {
            ProcessSignal(message);
        }
    }
    
    // Handle trailing stops
    HandleTrailing();
}

//+------------------------------------------------------------------+
//| Read message from WebSocket                                      |
//+------------------------------------------------------------------+
string ReadMessage() {
    uchar buffer[];
    uint len = SocketRead(socket_handle, buffer, 1024, 100);
    
    if(len > 0) {
        return CharArrayToString(buffer, 0, len);
    }
    
    return "";
}

//+------------------------------------------------------------------+
//| Process trading signal                                           |
//+------------------------------------------------------------------+
void ProcessSignal(string message) {
    Print("Received signal: ", message);
    
    // Simple JSON parsing (basic implementation)
    if(StringFind(message, "\"action\":\"BUY\"") >= 0) {
        double confidence = ExtractConfidence(message);
        double lot_size = ExtractLotSize(message);
        
        if(ValidateSignal(confidence, lot_size)) {
            ExecuteBuy(lot_size, confidence);
        }
    }
    else if(StringFind(message, "\"action\":\"SELL\"") >= 0) {
        double confidence = ExtractConfidence(message);
        double lot_size = ExtractLotSize(message);
        
        if(ValidateSignal(confidence, lot_size)) {
            ExecuteSell(lot_size, confidence);
        }
    }
}

//+------------------------------------------------------------------+
//| Extract confidence from JSON message                             |
//+------------------------------------------------------------------+
double ExtractConfidence(string message) {
    int start = StringFind(message, "\"confidence\":");
    if(start < 0) return 0.5;
    
    start += 13; // Length of "confidence":
    int end = StringFind(message, ",", start);
    if(end < 0) end = StringFind(message, "}", start);
    
    string conf_str = StringSubstr(message, start, end - start);
    return StringToDouble(conf_str);
}

//+------------------------------------------------------------------+
//| Extract lot size from JSON message                               |
//+------------------------------------------------------------------+
double ExtractLotSize(string message) {
    int start = StringFind(message, "\"lot_size\":");
    if(start < 0) return 0.01;
    
    start += 11; // Length of "lot_size":
    int end = StringFind(message, ",", start);
    if(end < 0) end = StringFind(message, "}", start);
    
    string lot_str = StringSubstr(message, start, end - start);
    return StringToDouble(lot_str);
}

//+------------------------------------------------------------------+
//| Validate trading signal                                          |
//+------------------------------------------------------------------+
bool ValidateSignal(double confidence, double lot_size) {
    // Check confidence
    if(confidence < MinConfidence) {
        Print("Low confidence: ", confidence);
        return false;
    }
    
    // Check positions limit
    if(PositionsTotal() >= MaxPositions) {
        Print("Max positions reached");
        return false;
    }
    
    // Check spread
    double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
    if(spread > MaxSpread * _Point) {
        Print("High spread: ", spread);
        return false;
    }
    
    return true;
}

//+------------------------------------------------------------------+
//| Execute buy order                                                |
//+------------------------------------------------------------------+
void ExecuteBuy(double lot_size, double confidence) {
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double atr = GetATR();
    
    double sl = ask - atr * 2;
    double tp = ask + atr * 3;
    
    // Adjust lot size based on risk
    double risk_lot = CalculateRiskLot(confidence);
    lot_size = MathMin(lot_size, risk_lot);
    
    if(trade.Buy(lot_size, _Symbol, ask, sl, tp, "AI_BUY")) {
        Print("BUY executed: ", lot_size, " lots");
    }
}

//+------------------------------------------------------------------+
//| Execute sell order                                               |
//+------------------------------------------------------------------+
void ExecuteSell(double lot_size, double confidence) {
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double atr = GetATR();
    
    double sl = bid + atr * 2;
    double tp = bid - atr * 3;
    
    // Adjust lot size based on risk
    double risk_lot = CalculateRiskLot(confidence);
    lot_size = MathMin(lot_size, risk_lot);
    
    if(trade.Sell(lot_size, _Symbol, bid, sl, tp, "AI_SELL")) {
        Print("SELL executed: ", lot_size, " lots");
    }
}

//+------------------------------------------------------------------+
//| Calculate risk-based lot size                                    |
//+------------------------------------------------------------------+
double CalculateRiskLot(double confidence) {
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double risk_amount = balance * RiskPercent / 100;
    
    // Base calculation for XAUUSD
    double base_lot = risk_amount / 1000;
    
    // Adjust for confidence
    base_lot *= confidence;
    
    // Normalize
    double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    
    base_lot = MathMax(base_lot, min_lot);
    base_lot = MathMin(base_lot, max_lot);
    
    return NormalizeDouble(base_lot, 2);
}

//+------------------------------------------------------------------+
//| Get ATR value                                                    |
//+------------------------------------------------------------------+
double GetATR() {
    int atr_handle = iATR(_Symbol, PERIOD_CURRENT, 14);
    double atr_buffer[];
    
    if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) > 0) {
        IndicatorRelease(atr_handle);
        return atr_buffer[0];
    }
    
    return 0.001; // Default
}

//+------------------------------------------------------------------+
//| Handle trailing stops                                            |
//+------------------------------------------------------------------+
void HandleTrailing() {
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(PositionSelectByTicket(ticket)) {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol) {
                string comment = PositionGetString(POSITION_COMMENT);
                if(StringFind(comment, "AI_") == 0) {
                    TrailPosition(ticket);
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Trail individual position                                        |
//+------------------------------------------------------------------+
void TrailPosition(ulong ticket) {
    if(!PositionSelectByTicket(ticket)) return;
    
    double open_price = PositionGetDouble(POSITION_PRICE_OPEN);
    double current_sl = PositionGetDouble(POSITION_SL);
    ENUM_POSITION_TYPE pos_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
    
    if(pos_type == POSITION_TYPE_BUY) {
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double profit = (bid - open_price) / _Point;
        
        if(profit >= TrailingStart) {
            double new_sl = bid - TrailingStep * _Point;
            if(new_sl > current_sl) {
                trade.PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP));
            }
        }
    }
    else if(pos_type == POSITION_TYPE_SELL) {
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double profit = (open_price - ask) / _Point;
        
        if(profit >= TrailingStart) {
            double new_sl = ask + TrailingStep * _Point;
            if(new_sl < current_sl || current_sl == 0) {
                trade.PositionModify(ticket, new_sl, PositionGetDouble(POSITION_TP));
            }
        }
    }
}