import cv2
import numpy as np

class OpticalFlowModule:
    def __init__(self):
        self.prev_gray = None
    
    def process(self, input_data, data_bus, register_bank):
        frame = input_data['raw_frame']
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            register_bank.write('movement_vectors', (magnitude, angle))
            data_bus['optical_flow'] = flow
        
        self.prev_gray = gray
