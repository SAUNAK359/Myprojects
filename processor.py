import time
import threading
from queue import PriorityQueue
from register_bank import RegisterBank

class CrowdProcessor:
    def __init__(self):
        self.clock_cycle = 0.033  # 30fps
        self.modules = []
        self.data_bus = {}
        self.register_bank = RegisterBank()
        self.pipeline_queue = PriorityQueue()
        self.execution_thread = None
        self.running = False
    
    def add_module(self, module, priority=5):
        self.modules.append((priority, module))
    
    def machine_cycle(self, frame):
        start_time = time.time()
        
        # Prepare input data
        input_data = {
            'raw_frame': frame,
            'timestamp': time.time(),
            'clock': self.clock_cycle
        }
        
        # Process modules in priority order
        for priority, module in sorted(self.modules, key=lambda x: x[0]):
            module.process(input_data, self.data_bus, self.register_bank)
        
        # Get processed data
        processed_data = self.register_bank.flush_output()
        
        # Maintain frame rate
        processing_time = time.time() - start_time
        sleep_time = max(0, self.clock_cycle - processing_time)
        time.sleep(sleep_time)
        
        return processed_data
    
    def start_async_processing(self, frame_generator):
        self.running = True
        self.execution_thread = threading.Thread(
            target=self._async_process,
            args=(frame_generator,)
        )
        self.execution_thread.start()
    
    def _async_process(self, frame_generator):
        while self.running:
            try:
                frame = next(frame_generator)
                self.pipeline_queue.put(self.machine_cycle(frame))
            except StopIteration:
                break
    
    def stop_async_processing(self):
        self.running = False
        if self.execution_thread:
            self.execution_thread.join()
