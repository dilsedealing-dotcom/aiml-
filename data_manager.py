import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
from data_processor import DataProcessor

class DataManager:
    def __init__(self):
        self.data_processor = DataProcessor()
        self.data_dir = 'data'
        
    def collect_live_data(self, symbol="XAUUSD", days=30):
        """Collect live data for specified days"""
        print(f"Collecting {days} days of live data for {symbol}...")
        
        # Calculate data count (5min bars)
        bars_per_day = 24 * 60 / 5  # 288 bars per day
        count = int(days * bars_per_day)
        
        df = self.data_processor.download_and_process_data(symbol, count=count)
        
        if df is not None:
            # Store with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'{self.data_dir}/{symbol}_live_{timestamp}.csv'
            df.to_csv(filename, index=False)
            
            # Update metadata
            self.update_data_metadata(symbol, filename, len(df))
            
            print(f"Collected {len(df)} bars, saved to {filename}")
            return df
        
        return None
    
    def update_data_metadata(self, symbol, filename, bars_count):
        """Update data collection metadata"""
        metadata_file = f'{self.data_dir}/metadata.json'
        
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        if symbol not in metadata:
            metadata[symbol] = []
        
        metadata[symbol].append({
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'bars_count': bars_count,
            'collection_type': 'live'
        })
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    def get_training_dataset(self, symbol="XAUUSD", min_bars=10000):
        """Combine all data files for training"""
        print(f"Preparing training dataset for {symbol}...")
        
        # Find all data files for symbol
        data_files = []
        for file in os.listdir(self.data_dir):
            if file.startswith(f'{symbol}_') and file.endswith('.csv'):
                data_files.append(file)
        
        if not data_files:
            print(f"No data files found for {symbol}")
            return None
        
        # Load and combine all data
        all_data = []
        for file in sorted(data_files):
            try:
                df = pd.read_csv(f'{self.data_dir}/{file}')
                df['time'] = pd.to_datetime(df['time'])
                all_data.append(df)
                print(f"Loaded {len(df)} bars from {file}")
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not all_data:
            return None
        
        # Combine and deduplicate
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df = combined_df.drop_duplicates(subset=['time']).sort_values('time')
        
        print(f"Combined dataset: {len(combined_df)} bars")
        
        if len(combined_df) < min_bars:
            print(f"Warning: Dataset has only {len(combined_df)} bars, minimum recommended: {min_bars}")
        
        # Save combined dataset
        combined_file = f'{self.data_dir}/{symbol}_combined_training.csv'
        combined_df.to_csv(combined_file, index=False)
        
        return combined_df
    
    def setup_continuous_collection(self, symbols=['XAUUSD'], interval_minutes=60):
        """Setup continuous data collection"""
        import schedule
        import time
        
        def collect_data():
            for symbol in symbols:
                try:
                    # Collect last 500 bars
                    df = self.data_processor.download_and_process_data(symbol, count=500)
                    if df is not None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                        filename = f'{self.data_dir}/{symbol}_continuous_{timestamp}.csv'
                        df.to_csv(filename, index=False)
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Collected {len(df)} bars for {symbol}")
                except Exception as e:
                    print(f"Error collecting data for {symbol}: {e}")
        
        # Schedule collection
        schedule.every(interval_minutes).minutes.do(collect_data)
        
        print(f"Starting continuous data collection every {interval_minutes} minutes...")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            print("Data collection stopped")