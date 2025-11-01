import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from datetime import datetime, timedelta

class CrowdDashboard:
    def __init__(self, data_logger, gps_tracker):
        self.window_name = "Enhanced Crowd Analytics Dashboard"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 2560, 1440)
        
        self.data_logger = data_logger
        self.gps_tracker = gps_tracker
        self.alert_colors = [(0,255,0), (0,255,255), (0,0,255)]
        self.alert_status = 0
        self.fig = self._init_plots()
        
    def _init_plots(self):
        fig = plt.figure(figsize=(12,8), dpi=120)
        plt.tight_layout(pad=3.0)
        return fig

    def update(self, frame, processed_data):
        self._log_data(processed_data)
        dashboard = self._create_dashboard(frame, processed_data)
        cv2.imshow(self.window_name, dashboard)

    def _log_data(self, data):
        velocities = [np.linalg.norm(sf['force'])**2 for sf in data.get('social_force', [])]
        rms_velocity = np.sqrt(np.mean(velocities)) if velocities else 0
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'density': np.mean(data.get('density_map', np.zeros((10,10))))/255,
            'velocity': rms_velocity,
            'anomaly_score': np.mean([a['anomaly_score'] for a in data.get('anomaly_scores',[])]),
            'alert_status': self.alert_status,
            'object_count': len(data.get('object_detections',[]))
        }
        
        if self.gps_tracker.connected:
            self.gps_tracker.update()
            log_data.update({
                'gps_lat': self.gps_tracker.current_location[0],
                'gps_lon': self.gps_tracker.current_location[1],
                'heading': self.gps_tracker.heading
            })
        
        if any(a['anomaly_score'] > 0.7 for a in data.get('anomaly_scores',[])):
            anomaly = max(data['anomaly_scores'], key=lambda x: x['anomaly_score'])
            log_data['anomaly_location'] = f"{anomaly['position'][0]},{anomaly['position'][1]}"
        
        self.data_logger.log_data(log_data)

    def _create_dashboard(self, frame, data):
        viz_frame = self._draw_detections(frame.copy(), data)
        plot_img = self._generate_plots()
        status_panel = self._create_status_panel(data)
        
        dashboard = np.zeros((1440,2560,3), dtype=np.uint8)
        dashboard[0:1080, 0:1440] = cv2.resize(viz_frame, (1440,1080))
        dashboard[0:1080, 1440:2560] = plot_img
        dashboard[1080:1440, 0:2560] = status_panel
        
        return dashboard

    def _draw_detections(self, frame, data):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for obj in data.get('object_detections',[]):
            x,y,w,h = obj['box']
            cv2.rectangle(frame_rgb, (x,y), (x+w,y+h), (255,0,0), 2)
        for anomaly in data.get('anomaly_scores',[]):
            if anomaly['anomaly_score'] > 0.5:
                x,y = anomaly['position']
                cv2.circle(frame_rgb, (int(x),int(y)), 20, (0,0,255), -1)
        return frame_rgb

    def _generate_plots(self):
        recent_data = self.data_logger.get_recent_data(30)
        if recent_data.empty: return np.zeros((1080,1120,3), dtype=np.uint8)
        
        axs = [self.fig.add_subplot(3,1,i+1) for i in range(3)]
        for ax in axs: ax.clear()
        
        axs[0].plot(recent_data['timestamp'], recent_data['density'], 'b-')
        axs[1].plot(recent_data['timestamp'], recent_data['velocity'], 'r-')
        axs[2].plot(recent_data['timestamp'], recent_data['anomaly_score'], 'g-')
        
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        plot_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        return cv2.resize(plot_img.reshape(canvas.get_width_height()[::-1] + (3,)), (1120,1080))

    def _create_status_panel(self, data):
        panel = np.zeros((360,2560,3), dtype=np.uint8)
        cv2.rectangle(panel, (0,0), (2560,360), (50,50,50), -1)
        
        status_text = ["Normal","Suspicious","High Alert"][self.alert_status]
        cv2.putText(panel, f"STATUS: {status_text}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.alert_colors[self.alert_status], 3)
        
        if self.gps_tracker.connected:
            cv2.putText(panel, f"GPS: {self.gps_tracker.current_location[0]:.6f}, {self.gps_tracker.current_location[1]:.6f}", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(panel, f"Heading: {self.gps_tracker.heading:.1f}°", (50,140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        return panel

    def should_exit(self):
        return cv2.waitKey(1) & 0xFF == ord('q')
