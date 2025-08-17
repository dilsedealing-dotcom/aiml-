#!/usr/bin/env python3
"""
Launch Enhanced MT5 AI Trading System GUI
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from trading_app_ui import main
    
    if __name__ == "__main__":
        print("🚀 Launching Enhanced MT5 AI Trading System GUI...")
        main()
        
except ImportError as e:
    print(f"Error importing GUI: {e}")
    print("Please ensure all dependencies are installed:")
    print("pip install tkinter matplotlib pandas numpy scikit-learn")
except Exception as e:
    print(f"Error launching GUI: {e}")
    input("Press Enter to exit...")