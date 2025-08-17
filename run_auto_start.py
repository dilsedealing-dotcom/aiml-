#!/usr/bin/env python3
"""
Quick Auto Start - One-Click Complete System Setup
"""

from auto_start_sequence import AutoStartSequence

def main():
    print("🚀 ONE-CLICK AUTO START")
    print("=" * 40)
    
    # Default symbol
    symbol = "XAUUSD"
    
    print(f"Auto-starting complete AI trading system for {symbol}...")
    print("This includes: Data → Training → Optimization → Live Trading")
    
    # Run auto sequence
    auto_start = AutoStartSequence(symbol)
    auto_start.run_complete_sequence()

if __name__ == "__main__":
    main()