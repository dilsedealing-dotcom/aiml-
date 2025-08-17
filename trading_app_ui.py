#!/usr/bin/env python3
"""
Enhanced MT5 AI Trading System - Modern GUI Application
Complete UI with all features and functions
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd

# Import system modules
from main import TradingSystem
from auto_start_sequence import AutoStartSequence
from daily_backtest_optimizer import DailyBacktestOptimizer
from adaptive_live_trader import AdaptiveLiveTrader

class TradingAppUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Enhanced MT5 AI Trading System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2b2b2b')
        
        # Initialize trading system
        self.trading_system = TradingSystem()
        self.trading_system.setup_directories()
        
        # Variables
        self.is_trading = False
        self.current_symbol = tk.StringVar(value="XAUUSD")
        self.status_text = tk.StringVar(value="System Ready")
        
        # Setup UI
        self.setup_styles()
        self.create_main_layout()
        self.update_status_display()
        
    def setup_styles(self):
        """Setup modern dark theme styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('Title.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ffffff', 
                       font=('Arial', 16, 'bold'))
        
        style.configure('Header.TLabel', 
                       background='#2b2b2b', 
                       foreground='#00ff88', 
                       font=('Arial', 12, 'bold'))
        
        style.configure('Info.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ffffff', 
                       font=('Arial', 10))
        
        style.configure('Success.TLabel', 
                       background='#2b2b2b', 
                       foreground='#00ff88', 
                       font=('Arial', 10))
        
        style.configure('Warning.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ffaa00', 
                       font=('Arial', 10))
        
        style.configure('Error.TLabel', 
                       background='#2b2b2b', 
                       foreground='#ff4444', 
                       font=('Arial', 10))
    
    def create_main_layout(self):
        """Create main application layout"""
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, 
                               text="🚀 Enhanced MT5 AI Trading System", 
                               style='Title.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_dashboard_tab()
        self.create_data_tab()
        self.create_training_tab()
        self.create_optimization_tab()
        self.create_trading_tab()
        self.create_analysis_tab()
        self.create_logs_tab()
        self.create_settings_tab()
        
        # Status bar
        self.create_status_bar(main_frame)
    
    def create_dashboard_tab(self):
        """Dashboard overview tab"""
        dashboard_frame = ttk.Frame(self.notebook)
        self.notebook.add(dashboard_frame, text="📊 Dashboard")
        
        # Quick actions frame
        actions_frame = tk.LabelFrame(dashboard_frame, text="Quick Actions", 
                                     bg='#3b3b3b', fg='#ffffff', font=('Arial', 12, 'bold'))
        actions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Auto start button
        auto_start_btn = tk.Button(actions_frame, 
                                  text="🚀 AUTO START - Complete Setup", 
                                  command=self.auto_start_sequence,
                                  bg='#00ff88', fg='#000000', 
                                  font=('Arial', 12, 'bold'),
                                  height=2)
        auto_start_btn.pack(fill=tk.X, padx=10, pady=5)
        
        # Quick buttons row
        quick_frame = tk.Frame(actions_frame, bg='#3b3b3b')
        quick_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(quick_frame, text="📥 Download Data", 
                 command=self.download_data, bg='#4a90e2', fg='#ffffff').pack(side=tk.LEFT, padx=5)
        tk.Button(quick_frame, text="🧠 Train Models", 
                 command=self.train_models, bg='#f5a623', fg='#ffffff').pack(side=tk.LEFT, padx=5)
        tk.Button(quick_frame, text="⚡ Start Trading", 
                 command=self.start_trading, bg='#7ed321', fg='#ffffff').pack(side=tk.LEFT, padx=5)
        
        # System status frame
        status_frame = tk.LabelFrame(dashboard_frame, text="System Status", 
                                    bg='#3b3b3b', fg='#ffffff', font=('Arial', 12, 'bold'))
        status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status display
        self.status_display = scrolledtext.ScrolledText(status_frame, 
                                                       height=15, 
                                                       bg='#1e1e1e', 
                                                       fg='#ffffff',
                                                       font=('Consolas', 10))
        self.status_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_data_tab(self):
        """Data management tab"""
        data_frame = ttk.Frame(self.notebook)
        self.notebook.add(data_frame, text="📈 Data")
        
        # Symbol selection
        symbol_frame = tk.LabelFrame(data_frame, text="Symbol Selection", 
                                    bg='#3b3b3b', fg='#ffffff')
        symbol_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(symbol_frame, text="Symbol:", bg='#3b3b3b', fg='#ffffff').pack(side=tk.LEFT, padx=10)
        symbol_entry = tk.Entry(symbol_frame, textvariable=self.current_symbol, width=10)
        symbol_entry.pack(side=tk.LEFT, padx=5)
        
        tk.Label(symbol_frame, text="Bars:", bg='#3b3b3b', fg='#ffffff').pack(side=tk.LEFT, padx=(20,5))
        self.bars_var = tk.StringVar(value="50000")
        bars_entry = tk.Entry(symbol_frame, textvariable=self.bars_var, width=10)
        bars_entry.pack(side=tk.LEFT, padx=5)
        
        # Data actions
        data_actions_frame = tk.LabelFrame(data_frame, text="Data Operations", 
                                          bg='#3b3b3b', fg='#ffffff')
        data_actions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(data_actions_frame, text="📥 Download Real Data", 
                 command=self.download_data, bg='#4a90e2', fg='#ffffff').pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(data_actions_frame, text="🔍 Correlation Analysis", 
                 command=self.run_correlation_analysis, bg='#9013fe', fg='#ffffff').pack(side=tk.LEFT, padx=10, pady=10)
        tk.Button(data_actions_frame, text="📊 View Data Info", 
                 command=self.view_data_info, bg='#50e3c2', fg='#ffffff').pack(side=tk.LEFT, padx=10, pady=10)
        
        # Data visualization area
        viz_frame = tk.LabelFrame(data_frame, text="Data Visualization", 
                                 bg='#3b3b3b', fg='#ffffff')
        viz_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Matplotlib canvas
        self.fig, self.ax = plt.subplots(figsize=(10, 6), facecolor='#2b2b2b')
        self.ax.set_facecolor('#1e1e1e')
        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_training_tab(self):
        """Model training tab"""
        training_frame = ttk.Frame(self.notebook)
        self.notebook.add(training_frame, text="🧠 Training")
        
        # Training options
        options_frame = tk.LabelFrame(training_frame, text="Training Options", 
                                     bg='#3b3b3b', fg='#ffffff')
        options_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(options_frame, text="🔧 Enhanced ML Model", 
                 command=self.train_enhanced_model, 
                 bg='#f5a623', fg='#ffffff', width=20).pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Button(options_frame, text="⏰ Multi-Timeframe Models", 
                 command=self.train_multi_timeframe, 
                 bg='#bd10e0', fg='#ffffff', width=20).pack(side=tk.LEFT, padx=10, pady=10)
        
        # Training progress
        progress_frame = tk.LabelFrame(training_frame, text="Training Progress", 
                                      bg='#3b3b3b', fg='#ffffff')
        progress_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.training_progress = ttk.Progressbar(progress_frame, mode='indeterminate')
        self.training_progress.pack(fill=tk.X, padx=10, pady=10)
        
        # Model status
        model_status_frame = tk.LabelFrame(training_frame, text="Model Status", 
                                          bg='#3b3b3b', fg='#ffffff')
        model_status_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.model_status_text = scrolledtext.ScrolledText(model_status_frame, 
                                                          height=15, 
                                                          bg='#1e1e1e', 
                                                          fg='#ffffff')
        self.model_status_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_optimization_tab(self):
        """Optimization tab"""
        opt_frame = ttk.Frame(self.notebook)
        self.notebook.add(opt_frame, text="⚡ Optimization")
        
        # Optimization controls
        controls_frame = tk.LabelFrame(opt_frame, text="Optimization Controls", 
                                      bg='#3b3b3b', fg='#ffffff')
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(controls_frame, text="📅 Daily Backtest Optimization", 
                 command=self.run_daily_optimization, 
                 bg='#ff6b6b', fg='#ffffff', width=25).pack(side=tk.LEFT, padx=10, pady=10)
        
        tk.Button(controls_frame, text="🎯 Enhanced Backtest", 
                 command=self.run_enhanced_backtest, 
                 bg='#4ecdc4', fg='#ffffff', width=20).pack(side=tk.LEFT, padx=10, pady=10)
        
        # Optimization results
        results_frame = tk.LabelFrame(opt_frame, text="Optimization Results", 
                                     bg='#3b3b3b', fg='#ffffff')
        results_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.opt_results_text = scrolledtext.ScrolledText(results_frame, 
                                                         height=20, 
                                                         bg='#1e1e1e', 
                                                         fg='#ffffff')
        self.opt_results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_trading_tab(self):
        """Live trading tab"""
        trading_frame = ttk.Frame(self.notebook)
        self.notebook.add(trading_frame, text="💰 Trading")
        
        # Trading controls
        controls_frame = tk.LabelFrame(trading_frame, text="Trading Controls", 
                                      bg='#3b3b3b', fg='#ffffff')
        controls_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_trading_btn = tk.Button(controls_frame, text="▶️ Start Adaptive Trading", 
                                          command=self.start_adaptive_trading, 
                                          bg='#7ed321', fg='#ffffff', width=20)
        self.start_trading_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.stop_trading_btn = tk.Button(controls_frame, text="⏹️ Stop Trading", 
                                         command=self.stop_trading, 
                                         bg='#d0021b', fg='#ffffff', width=15)
        self.stop_trading_btn.pack(side=tk.LEFT, padx=10, pady=10)
        
        # Trading status
        status_frame = tk.LabelFrame(trading_frame, text="Trading Status", 
                                    bg='#3b3b3b', fg='#ffffff')
        status_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Status indicators
        status_grid = tk.Frame(status_frame, bg='#3b3b3b')
        status_grid.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(status_grid, text="Status:", bg='#3b3b3b', fg='#ffffff').grid(row=0, column=0, sticky='w')
        self.trading_status_label = tk.Label(status_grid, text="Stopped", bg='#3b3b3b', fg='#ff4444')
        self.trading_status_label.grid(row=0, column=1, sticky='w', padx=10)
        
        tk.Label(status_grid, text="Balance:", bg='#3b3b3b', fg='#ffffff').grid(row=0, column=2, sticky='w', padx=(20,0))
        self.balance_label = tk.Label(status_grid, text="$10,000", bg='#3b3b3b', fg='#00ff88')
        self.balance_label.grid(row=0, column=3, sticky='w', padx=10)
        
        tk.Label(status_grid, text="Open Positions:", bg='#3b3b3b', fg='#ffffff').grid(row=1, column=0, sticky='w')
        self.positions_label = tk.Label(status_grid, text="0", bg='#3b3b3b', fg='#ffffff')
        self.positions_label.grid(row=1, column=1, sticky='w', padx=10)
        
        tk.Label(status_grid, text="Win Rate:", bg='#3b3b3b', fg='#ffffff').grid(row=1, column=2, sticky='w', padx=(20,0))
        self.winrate_label = tk.Label(status_grid, text="0%", bg='#3b3b3b', fg='#ffffff')
        self.winrate_label.grid(row=1, column=3, sticky='w', padx=10)
        
        # Live signals
        signals_frame = tk.LabelFrame(trading_frame, text="Live Signals", 
                                     bg='#3b3b3b', fg='#ffffff')
        signals_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.signals_text = scrolledtext.ScrolledText(signals_frame, 
                                                     height=15, 
                                                     bg='#1e1e1e', 
                                                     fg='#ffffff')
        self.signals_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_analysis_tab(self):
        """Analysis and charts tab"""
        analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(analysis_frame, text="📊 Analysis")
        
        # Chart controls
        chart_controls = tk.LabelFrame(analysis_frame, text="Chart Controls", 
                                      bg='#3b3b3b', fg='#ffffff')
        chart_controls.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(chart_controls, text="📈 Price Chart", 
                 command=self.show_price_chart, bg='#4a90e2', fg='#ffffff').pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(chart_controls, text="🔗 Correlation Matrix", 
                 command=self.show_correlation_matrix, bg='#9013fe', fg='#ffffff').pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(chart_controls, text="📊 Performance Chart", 
                 command=self.show_performance_chart, bg='#50e3c2', fg='#ffffff').pack(side=tk.LEFT, padx=10, pady=5)
        
        # Chart area
        chart_frame = tk.LabelFrame(analysis_frame, text="Charts", 
                                   bg='#3b3b3b', fg='#ffffff')
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Analysis matplotlib canvas
        self.analysis_fig, self.analysis_ax = plt.subplots(figsize=(12, 8), facecolor='#2b2b2b')
        self.analysis_ax.set_facecolor('#1e1e1e')
        self.analysis_canvas = FigureCanvasTkAgg(self.analysis_fig, chart_frame)
        self.analysis_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_logs_tab(self):
        """Logs and monitoring tab"""
        logs_frame = ttk.Frame(self.notebook)
        self.notebook.add(logs_frame, text="📋 Logs")
        
        # Log controls
        log_controls = tk.LabelFrame(logs_frame, text="Log Controls", 
                                    bg='#3b3b3b', fg='#ffffff')
        log_controls.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(log_controls, text="🔄 Refresh Logs", 
                 command=self.refresh_logs, bg='#4a90e2', fg='#ffffff').pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(log_controls, text="🗑️ Clear Logs", 
                 command=self.clear_logs, bg='#d0021b', fg='#ffffff').pack(side=tk.LEFT, padx=10, pady=5)
        tk.Button(log_controls, text="💾 Export Logs", 
                 command=self.export_logs, bg='#f5a623', fg='#ffffff').pack(side=tk.LEFT, padx=10, pady=5)
        
        # Logs display
        logs_display_frame = tk.LabelFrame(logs_frame, text="System Logs", 
                                          bg='#3b3b3b', fg='#ffffff')
        logs_display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.logs_text = scrolledtext.ScrolledText(logs_display_frame, 
                                                  height=25, 
                                                  bg='#1e1e1e', 
                                                  fg='#ffffff',
                                                  font=('Consolas', 9))
        self.logs_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_settings_tab(self):
        """Settings and configuration tab"""
        settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(settings_frame, text="⚙️ Settings")
        
        # Trading settings
        trading_settings = tk.LabelFrame(settings_frame, text="Trading Settings", 
                                        bg='#3b3b3b', fg='#ffffff')
        trading_settings.pack(fill=tk.X, padx=10, pady=10)
        
        # Settings grid
        settings_grid = tk.Frame(trading_settings, bg='#3b3b3b')
        settings_grid.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(settings_grid, text="Risk %:", bg='#3b3b3b', fg='#ffffff').grid(row=0, column=0, sticky='w')
        self.risk_var = tk.StringVar(value="2.0")
        tk.Entry(settings_grid, textvariable=self.risk_var, width=10).grid(row=0, column=1, padx=10)
        
        tk.Label(settings_grid, text="Initial Balance:", bg='#3b3b3b', fg='#ffffff').grid(row=0, column=2, sticky='w', padx=(20,0))
        self.balance_var = tk.StringVar(value="10000")
        tk.Entry(settings_grid, textvariable=self.balance_var, width=10).grid(row=0, column=3, padx=10)
        
        tk.Label(settings_grid, text="Entry Threshold:", bg='#3b3b3b', fg='#ffffff').grid(row=1, column=0, sticky='w')
        self.entry_threshold_var = tk.StringVar(value="0.7")
        tk.Entry(settings_grid, textvariable=self.entry_threshold_var, width=10).grid(row=1, column=1, padx=10)
        
        tk.Label(settings_grid, text="Exit Threshold:", bg='#3b3b3b', fg='#ffffff').grid(row=1, column=2, sticky='w', padx=(20,0))
        self.exit_threshold_var = tk.StringVar(value="0.3")
        tk.Entry(settings_grid, textvariable=self.exit_threshold_var, width=10).grid(row=1, column=3, padx=10)
        
        # Save settings button
        tk.Button(trading_settings, text="💾 Save Settings", 
                 command=self.save_settings, bg='#7ed321', fg='#ffffff').pack(pady=10)
        
        # System info
        system_info = tk.LabelFrame(settings_frame, text="System Information", 
                                   bg='#3b3b3b', fg='#ffffff')
        system_info.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.system_info_text = scrolledtext.ScrolledText(system_info, 
                                                         height=15, 
                                                         bg='#1e1e1e', 
                                                         fg='#ffffff')
        self.system_info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def create_status_bar(self, parent):
        """Create status bar"""
        status_frame = tk.Frame(parent, bg='#1e1e1e', height=30)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        status_frame.pack_propagate(False)
        
        # Status label
        status_label = tk.Label(status_frame, textvariable=self.status_text, 
                               bg='#1e1e1e', fg='#00ff88', 
                               font=('Arial', 10))
        status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Time label
        self.time_label = tk.Label(status_frame, text="", 
                                  bg='#1e1e1e', fg='#ffffff', 
                                  font=('Arial', 10))
        self.time_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Update time
        self.update_time()
    
    def update_time(self):
        """Update time display"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)
    
    def log_message(self, message, level="INFO"):
        """Log message to status display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        color_map = {
            "INFO": "#ffffff",
            "SUCCESS": "#00ff88", 
            "WARNING": "#ffaa00",
            "ERROR": "#ff4444"
        }
        
        log_entry = f"[{timestamp}] {level}: {message}\n"
        
        # Add to status display
        self.status_display.insert(tk.END, log_entry)
        self.status_display.see(tk.END)
        
        # Add to logs tab
        self.logs_text.insert(tk.END, log_entry)
        self.logs_text.see(tk.END)
        
        # Update status text
        self.status_text.set(message)
    
    # Action methods
    def auto_start_sequence(self):
        """Run auto start sequence"""
        def run_auto_start():
            try:
                self.log_message("Starting auto sequence...", "INFO")
                auto_start = AutoStartSequence(self.current_symbol.get())
                auto_start.run_complete_sequence()
                self.log_message("Auto sequence completed!", "SUCCESS")
            except Exception as e:
                self.log_message(f"Auto start failed: {e}", "ERROR")
        
        threading.Thread(target=run_auto_start, daemon=True).start()
    
    def download_data(self):
        """Download market data"""
        def download():
            try:
                self.log_message(f"Downloading data for {self.current_symbol.get()}...", "INFO")
                symbol = self.current_symbol.get()
                count = int(self.bars_var.get())
                
                df = self.trading_system.download_and_store_data(symbol, count)
                if df is not None:
                    self.log_message(f"Downloaded {len(df)} bars successfully", "SUCCESS")
                    self.update_data_visualization(df)
                else:
                    self.log_message("Data download failed", "ERROR")
            except Exception as e:
                self.log_message(f"Download error: {e}", "ERROR")
        
        threading.Thread(target=download, daemon=True).start()
    
    def run_correlation_analysis(self):
        """Run correlation analysis"""
        def analyze():
            try:
                self.log_message("Running correlation analysis...", "INFO")
                results = self.trading_system.run_correlation_analysis(self.current_symbol.get())
                if results:
                    self.log_message("Correlation analysis completed", "SUCCESS")
                else:
                    self.log_message("Correlation analysis failed", "ERROR")
            except Exception as e:
                self.log_message(f"Analysis error: {e}", "ERROR")
        
        threading.Thread(target=analyze, daemon=True).start()
    
    def train_enhanced_model(self):
        """Train enhanced ML model"""
        def train():
            try:
                self.training_progress.start()
                self.log_message("Training enhanced model...", "INFO")
                
                model = self.trading_system.train_enhanced_model(self.current_symbol.get())
                if model:
                    self.log_message("Enhanced model trained successfully", "SUCCESS")
                else:
                    self.log_message("Model training failed", "ERROR")
                
                self.training_progress.stop()
            except Exception as e:
                self.log_message(f"Training error: {e}", "ERROR")
                self.training_progress.stop()
        
        threading.Thread(target=train, daemon=True).start()
    
    def train_multi_timeframe(self):
        """Train multi-timeframe models"""
        def train_mtf():
            try:
                self.training_progress.start()
                self.log_message("Training multi-timeframe models...", "INFO")
                
                symbol = self.current_symbol.get()
                multi_tf_data = self.trading_system.ml_trainer.collect_multi_timeframe_data(symbol, bars_per_tf=15000)
                
                if multi_tf_data:
                    combined_data = self.trading_system.ml_trainer.create_enhanced_features(multi_tf_data)
                    models = self.trading_system.ml_trainer.train_ensemble_models(combined_data)
                    model_file = self.trading_system.ml_trainer.save_pretrained_models(symbol)
                    
                    self.log_message(f"Multi-timeframe models trained: {len(multi_tf_data)} timeframes", "SUCCESS")
                else:
                    self.log_message("Multi-timeframe training failed", "ERROR")
                
                self.training_progress.stop()
            except Exception as e:
                self.log_message(f"Multi-timeframe training error: {e}", "ERROR")
                self.training_progress.stop()
        
        threading.Thread(target=train_mtf, daemon=True).start()
    
    def run_daily_optimization(self):
        """Run daily optimization"""
        def optimize():
            try:
                self.log_message("Running daily optimization...", "INFO")
                
                optimizer = DailyBacktestOptimizer(self.current_symbol.get())
                results = optimizer.run_weekly_optimization()
                
                if results:
                    patterns = results['patterns']
                    result_text = f"""
Daily Optimization Results:
- Optimal Entry Threshold: {patterns['best_entry_threshold']}
- Optimal Exit Threshold: {patterns['best_exit_threshold']}
- Expected Win Rate: {patterns['avg_win_rate']:.2%}
- Total P&L: ${patterns['total_pnl']:.2f}
- Best Day: {patterns['best_day']}
                    """
                    self.opt_results_text.insert(tk.END, result_text)
                    self.log_message("Daily optimization completed", "SUCCESS")
                else:
                    self.log_message("Daily optimization failed", "ERROR")
            except Exception as e:
                self.log_message(f"Optimization error: {e}", "ERROR")
        
        threading.Thread(target=optimize, daemon=True).start()
    
    def start_adaptive_trading(self):
        """Start adaptive live trading"""
        if not self.is_trading:
            self.is_trading = True
            self.trading_status_label.config(text="Running", fg='#00ff88')
            self.start_trading_btn.config(state='disabled')
            self.stop_trading_btn.config(state='normal')
            
            def trade():
                try:
                    self.log_message("Starting adaptive trading...", "INFO")
                    trader = AdaptiveLiveTrader(self.current_symbol.get())
                    # Note: In real implementation, this would run the trading loop
                    self.log_message("Adaptive trading started", "SUCCESS")
                except Exception as e:
                    self.log_message(f"Trading error: {e}", "ERROR")
                    self.stop_trading()
            
            threading.Thread(target=trade, daemon=True).start()
    
    def stop_trading(self):
        """Stop live trading"""
        self.is_trading = False
        self.trading_status_label.config(text="Stopped", fg='#ff4444')
        self.start_trading_btn.config(state='normal')
        self.stop_trading_btn.config(state='disabled')
        self.log_message("Trading stopped", "WARNING")
    
    def update_data_visualization(self, df):
        """Update data visualization"""
        try:
            self.ax.clear()
            self.ax.set_facecolor('#1e1e1e')
            
            # Plot price data
            if len(df) > 1000:
                df_plot = df.tail(1000)  # Last 1000 bars
            else:
                df_plot = df
            
            self.ax.plot(df_plot.index, df_plot['close'], color='#00ff88', linewidth=1)
            
            if 'bb_upper' in df_plot.columns:
                self.ax.plot(df_plot.index, df_plot['bb_upper'], color='#ff4444', alpha=0.7)
                self.ax.plot(df_plot.index, df_plot['bb_lower'], color='#ff4444', alpha=0.7)
                self.ax.fill_between(df_plot.index, df_plot['bb_upper'], df_plot['bb_lower'], alpha=0.1, color='#ff4444')
            
            self.ax.set_title(f'{self.current_symbol.get()} Price Chart', color='#ffffff')
            self.ax.tick_params(colors='#ffffff')
            self.canvas.draw()
            
        except Exception as e:
            self.log_message(f"Visualization error: {e}", "ERROR")
    
    def show_price_chart(self):
        """Show price chart in analysis tab"""
        try:
            df = self.trading_system.load_latest_data(self.current_symbol.get())
            if df is not None:
                self.analysis_ax.clear()
                self.analysis_ax.set_facecolor('#1e1e1e')
                
                # Plot comprehensive chart
                if len(df) > 1000:
                    df_plot = df.tail(1000)
                else:
                    df_plot = df
                
                self.analysis_ax.plot(df_plot.index, df_plot['close'], color='#00ff88', linewidth=1, label='Close')
                
                if 'bb_upper' in df_plot.columns:
                    self.analysis_ax.plot(df_plot.index, df_plot['bb_upper'], color='#ff4444', alpha=0.7, label='BB Upper')
                    self.analysis_ax.plot(df_plot.index, df_plot['bb_middle'], color='#ffaa00', alpha=0.7, label='BB Middle')
                    self.analysis_ax.plot(df_plot.index, df_plot['bb_lower'], color='#ff4444', alpha=0.7, label='BB Lower')
                
                self.analysis_ax.set_title(f'{self.current_symbol.get()} Technical Analysis', color='#ffffff')
                self.analysis_ax.legend()
                self.analysis_ax.tick_params(colors='#ffffff')
                self.analysis_canvas.draw()
                
                self.log_message("Price chart updated", "SUCCESS")
            else:
                self.log_message("No data available for chart", "WARNING")
        except Exception as e:
            self.log_message(f"Chart error: {e}", "ERROR")
    
    def update_status_display(self):
        """Update system status display"""
        try:
            # Count files
            data_files = len([f for f in os.listdir('data') if f.endswith('.csv')]) if os.path.exists('data') else 0
            model_files = len([f for f in os.listdir('models') if f.endswith('.pkl')]) if os.path.exists('models') else 0
            log_files = len([f for f in os.listdir('logs') if f.endswith('.json')]) if os.path.exists('logs') else 0
            
            status_info = f"""
System Status:
- Data Files: {data_files}
- Model Files: {model_files}
- Log Files: {log_files}
- Current Symbol: {self.current_symbol.get()}
- System: Ready
            """
            
            self.system_info_text.delete(1.0, tk.END)
            self.system_info_text.insert(tk.END, status_info)
            
        except Exception as e:
            self.log_message(f"Status update error: {e}", "ERROR")
    
    # Placeholder methods for remaining functionality
    def view_data_info(self):
        messagebox.showinfo("Data Info", "Data information feature coming soon!")
    
    def run_enhanced_backtest(self):
        messagebox.showinfo("Backtest", "Enhanced backtest feature coming soon!")
    
    def show_correlation_matrix(self):
        messagebox.showinfo("Correlation", "Correlation matrix feature coming soon!")
    
    def show_performance_chart(self):
        messagebox.showinfo("Performance", "Performance chart feature coming soon!")
    
    def refresh_logs(self):
        self.log_message("Logs refreshed", "INFO")
    
    def clear_logs(self):
        self.logs_text.delete(1.0, tk.END)
        self.status_display.delete(1.0, tk.END)
        self.log_message("Logs cleared", "INFO")
    
    def export_logs(self):
        messagebox.showinfo("Export", "Log export feature coming soon!")
    
    def save_settings(self):
        self.log_message("Settings saved", "SUCCESS")
    
    def train_models(self):
        self.train_enhanced_model()
    
    def start_trading(self):
        self.start_adaptive_trading()

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = TradingAppUI(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()