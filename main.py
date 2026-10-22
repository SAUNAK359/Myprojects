import cv2
import sys
from processor import CrowdProcessor
from dashboard import CrowdDashboard
from data_logger import DataLogger
from gps_tracker import GPSTracker
from yolo_detection import YOLODetectionModule


def main(video_source, model_path):
    data_logger = DataLogger()
    gps_tracker = GPSTracker()
    gps_tracker.connect()
    
    processor = CrowdProcessor()
    dashboard = CrowdDashboard(data_logger, gps_tracker)
    
    try:
        processor.add_module(RTDETRDetectionModule(r"D:\Behaviour Analysis\Final Project VisDrone  AI4Bharat.Ltd\Dashboard Developed\rtdetr-x.pt"), priority=1)
    except:
        processor.add_module(YOLODetectionModule(r"C:\Users\hp\Downloads\yolov3.weights", r"D:\Behaviour Analysis\Final Project VisDrone  AI4Bharat.Ltd\Dashboard Developed\yolov3.cfg"), priority=1)
    
    processor.add_module(OpticalFlowModule(), priority=1)
    processor.add_module(DensityEstimationModule(), priority=2)
    processor.add_module(SocialForceModule(), priority=2)
    processor.add_module(AnomalyDetectionModule(), priority=3)
    
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        processed_data = processor.machine_cycle(frame)
        dashboard.update(frame, processed_data)
        
        if dashboard.should_exit(): break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else 0, 
         sys.argv[2] if len(sys.argv)>2 else "rtdetr-x.pt")
