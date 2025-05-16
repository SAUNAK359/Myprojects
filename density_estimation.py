import cv2

class DensityEstimationModule:
    def process(self, input_data, data_bus, register_bank):
        flow_data = register_bank.read('movement_vectors')
        if flow_data is None:
            return
        
        magnitude, _ = flow_data
        density_map = cv2.GaussianBlur(magnitude, (15, 15), 0)
        density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)
        
        register_bank.write('density_map', density_map)
        data_bus['density_map'] = density_map
