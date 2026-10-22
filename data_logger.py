import csv
import os
from datetime import datetime
import pandas as pd
import numpy as np

class DataLogger:
    def __init__(self):
        self.columns = [
            'timestamp', 'density', 'velocity', 'anomaly_score',
            'alert_status', 'gps_lat', 'gps_lon', 'heading',
            'anomaly_location', 'object_count'
        ]
        self.file_path = self._init_log_file()
        self.df = pd.DataFrame(columns=self.columns)
        self.last_save = datetime.now()

    def _init_log_file(self):
        os.makedirs('logs', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f'logs/crowd_analysis_{timestamp}.csv'

    def log_data(self, data):
        new_row = {col: data.get(col, np.nan) for col in self.columns}
        self.df = pd.concat([self.df, pd.DataFrame([new_row])], ignore_index=True)
        
        if (datetime.now() - self.last_save).seconds >= 5:
            self.df.to_csv(self.file_path, index=False)
            self.last_save = datetime.now()

    def get_recent_data(self, window_seconds=30):
        now = datetime.now()
        mask = (now - pd.to_datetime(self.df['timestamp'])) <= pd.Timedelta(seconds=window_seconds)
        return self.df[mask].copy()
