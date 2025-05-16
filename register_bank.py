class RegisterBank:
    def __init__(self):
        self.registers = {
            'movement_vectors': None,
            'density_map': None,
            'object_detections': None,
            'social_force': None,
            'anomaly_scores': None
        }
    
    def write(self, register_name, data):
        self.registers[register_name] = data
    
    def read(self, register_name):
        return self.registers.get(register_name)
    
    def flush_output(self):
        output = self.registers.copy()
        self.registers = {k: None for k in self.registers}
        return output
