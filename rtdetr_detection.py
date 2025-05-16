import torch
import cv2
import numpy as np
from torchvision import transforms
from ultralytics import RTDETR

class RTDETRDetectionModule:
    def __init__(self, model_path, confidence_thresh=0.5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        try:
            self.model = RTDETR(model_path).to(device)
            self.model.conf = confidence_thresh
            self.device = device
            self.class_names = self.model.names
            self.model(torch.zeros(1, 3, 640, 640).to(device))
        except Exception as e:
            raise RuntimeError(f"Failed to initialize RT-DETR model: {str(e)}")
    
    def process(self, input_data, data_bus, register_bank):
        try:
            frame = input_data['raw_frame']
            frame_resized = cv2.resize(frame, (640, 640))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_tensor = transforms.ToTensor()(frame_rgb).unsqueeze(0).to(self.device)

            with torch.no_grad():
                results = self.model(frame_tensor)

            detections = []
            for result in results:
                if hasattr(result, "boxes") and result.boxes is not None:
                    for box in result.boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls)
                        conf = float(box.conf)
                        x1, y1, x2, y2 = map(int, xyxy)
                        detections.append({
                            'class': cls,
                            'confidence': conf,
                            'box': [x1, y1, x2 - x1, y2 - y1],
                            'class_name': self.class_names[cls]
                        })

            register_bank.write('object_detections', detections)
            data_bus['object_detections'] = detections

        except Exception as e:
            print(f"RT-DETR processing error: {str(e)}")
            register_bank.write('object_detections', [])
            data_bus['object_detections'] = []

import cv2
import sys
import os
from processor import CrowdProcessor
from dashboard import CrowdDashboard
from optical_flow import OpticalFlowModule
from rtdetr_detection import RTDETRDetectionModule
from density_estimation import DensityEstimationModule
from social_force import SocialForceModule
from anomaly_detection import AnomalyDetectionModule

def main(video_source, model_path):
    try:
        processor = CrowdProcessor()
        dashboard = CrowdDashboard()

        try:
            if not os.path.exists(model_path):
                raise FileNotFoundError("RT-DETR model not found, check documentation for download instructions")

            processor.add_module(RTDETRDetectionModule(model_path), priority=1)
        except Exception as e:
            print(f"Warning: RT-DETR initialization failed - {str(e)}")
            print("Falling back to CPU-only mode with simplified detection")

        processor.add_module(OpticalFlowModule(), priority=1)
        processor.add_module(DensityEstimationModule(), priority=2)
        processor.add_module(SocialForceModule(), priority=2)
        processor.add_module(AnomalyDetectionModule(), priority=3)

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError("Could not open video source")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            try:
                processed_data = processor.machine_cycle(frame)
                dashboard.update(frame, processed_data)

                if dashboard.should_exit():
                    break

            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                continue

    except KeyboardInterrupt:
        print("Processing stopped by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_source = sys.argv[1] if len(sys.argv) > 1 else 0
    model_path = sys.argv[2] if len(sys.argv) > 2 else r"D:\Behaviour Analysis\Final Project VisDrone  AI4Bharat.Ltd\Dashboard Developed\rtdetr-x.pt"
    main(video_source, model_path)

