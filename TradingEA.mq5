//+------------------------------------------------------------------+
//|                                                    TradingEA.mq5 |
//|                                                                  |
//+------------------------------------------------------------------+
#property copyright "AI Trading System"
#property version   "1.00"

#include <Trade\Trade.mqh>
#include <WebSocket\WebSocket.mqh>rate 

input string WebSocketURL = "ws://localhost:8765";
input double MaxSpread = 3.0;
input double RiskPercent = 2.0;
input int TrailingStart = 550;
input int TrailingStep = 220;
input bool EnableTrailing = true;

CTrade trade;
CWebSocket websocket;

struct TradingSignal {
    string action;
    double confidence;
    double prediction;
    double current_price;
    double sl;
    double tp;
    double lot_size;
    datetime timestamp;
};

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit() {
    // Initialize WebSocket connection
    if(!websocket.Connect(WebSocketURL)) {
        Print("Failed to connect to WebSocket server");
        return INIT_FAILED;
    }
    
    Print("EA initialized successfully");
    return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason) {
    websocket.Disconnect();
    Print("EA deinitialized");
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick() {
    // Check for new WebSocket messages
    string message = websocket.ReceiveMessage();
    if(message != "") {
        ProcessTradingSignal(message);
    }
    
    // Handle trailing stops
    if(EnableTrailing) {
        HandleTrailingStops();
    }
}

//+------------------------------------------------------------------+
//| Process trading signal from WebSocket                           |
//+------------------------------------------------------------------+
void ProcessTradingSignal(string jsonMessage) {
    TradingSignal signal;
    
    // Parse JSON message (simplified - in real implementation use proper JSON parser)
    if(!ParseSignalFromJSON(jsonMessage, signal)) {
        Print("Failed to parse signal: ", jsonMessage);
        return;
    }
    
    // Check spread filter
    double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * _Point;
    if(spread > MaxSpread * _Point) {
        Print("Spread too high: ", spread, " > ", MaxSpread * _Point);
        return;
    }
    
    // Execute trade based on signal
    if(signal.action == "BUY") {
        ExecuteBuyOrder(signal);
    } else if(signal.action == "SELL") {
        ExecuteSellOrder(signal);
    }
}

//+------------------------------------------------------------------+
//| Execute buy order                                                |
//+------------------------------------------------------------------+
void ExecuteBuyOrder(TradingSignal &signal) {
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double lotSize = CalculateLotSize(signal.lot_size);
    
    // Calculate ATR-based SL if not provided
    double sl = signal.sl > 0 ? signal.sl : ask - GetATR() * 2;
    double tp = signal.tp > 0 ? signal.tp : ask + GetATR() * 3;
    
    if(trade.Buy(lotSize, _Symbol, ask, sl, tp, "AI Signal Buy")) {
        Print("Buy order executed: Lot=", lotSize, " SL=", sl, " TP=", tp);
    } else {
        Print("Buy order failed: ", trade.ResultRetcode());
    }
}

//+------------------------------------------------------------------+
//| Execute sell order                                               |
//+------------------------------------------------------------------+
void ExecuteSellOrder(TradingSignal &signal) {
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double lotSize = CalculateLotSize(signal.lot_size);
    
    // Calculate ATR-based SL if not provided
    double sl = signal.sl > 0 ? signal.sl : bid + GetATR() * 2;
    double tp = signal.tp > 0 ? signal.tp : bid - GetATR() * 3;
    
    if(trade.Sell(lotSize, _Symbol, bid, sl, tp, "AI Signal Sell")) {
        Print("Sell order executed: Lot=", lotSize, " SL=", sl, " TP=", tp);
    } else {
        Print("Sell order failed: ", trade.ResultRetcode());
    }
}

//+------------------------------------------------------------------+
//| Calculate lot size based on risk management                     |
//+------------------------------------------------------------------+
double CalculateLotSize(double signalLotSize) {
    double balance = AccountInfoDouble(ACCOUNT_BALANCE);
    double riskAmount = balance * RiskPercent / 100;
    
    // Use signal lot size but cap it based on risk
    double maxLot = riskAmount / 1000; // Simplified calculation
    return MathMin(signalLotSize, maxLot);
}

//+------------------------------------------------------------------+
//| Get ATR value for dynamic SL/TP                                 |
//+------------------------------------------------------------------+
double GetATR() {
    int atrHandle = iATR(_Symbol, PERIOD_CURRENT, 14);
    double atrBuffer[];
    ArraySetAsSeries(atrBuffer, true);
    
    if(CopyBuffer(atrHandle, 0, 0, 1, atrBuffer) > 0) {
        return atrBuffer[0];
    }
    
    return 0.001; // Default fallback
}

//+------------------------------------------------------------------+
//| Handle trailing stops for open positions                        |
//+------------------------------------------------------------------+
void HandleTrailingStops() {
    for(int i = PositionsTotal() - 1; i >= 0; i--) {
        if(PositionSelectByTicket(PositionGetTicket(i))) {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol) {
                double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
                double currentPrice = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 
                                    SymbolInfoDouble(_Symbol, SYMBOL_BID) : 
                                    SymbolInfoDouble(_Symbol, SYMBOL_ASK);
                
                double profit = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 
                              (currentPrice - openPrice) : (openPrice - currentPrice);
                
                if(profit >= TrailingStart * _Point) {
                    UpdateTrailingStop(PositionGetTicket(i), currentPrice);
                }
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Update trailing stop for a position                             |
//+------------------------------------------------------------------+
void UpdateTrailingStop(ulong ticket, double currentPrice) {
    if(PositionSelectByTicket(ticket)) {
        double currentSL = PositionGetDouble(POSITION_SL);
        double newSL;
        
        if(PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY) {
            newSL = currentPrice - TrailingStep * _Point;
            if(newSL > currentSL) {
                trade.PositionModify(ticket, newSL, PositionGetDouble(POSITION_TP));
            }
        } else {
            newSL = currentPrice + TrailingStep * _Point;
            if(newSL < currentSL || currentSL == 0) {
                trade.PositionModify(ticket, newSL, PositionGetDouble(POSITION_TP));
            }
        }
    }
}

//+------------------------------------------------------------------+
//| Parse JSON signal (simplified implementation)                   |
//+------------------------------------------------------------------+
bool ParseSignalFromJSON(string json, TradingSignal &signal) {
    // Simplified JSON parsing - in production use proper JSON library
    if(StringFind(json, "\"action\":\"BUY\"") >= 0) {
        signal.action = "BUY";
        return true;
    } else if(StringFind(json, "\"action\":\"SELL\"") >= 0) {
        signal.action = "SELL";
        return true;
    }
    
    return false;
}