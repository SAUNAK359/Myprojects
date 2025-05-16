import serial
import time
import pynmea2
from geopy.distance import geodesic

class GPSTracker:
    def __init__(self):
        self.connected = False
        self.serial_port = None
        self.current_location = (None, None)
        self.heading = None
        self.last_location = None
        self.last_update = None

    def connect(self, port='/dev/ttyAMA0', baudrate=9600):
        try:
            self.serial_port = serial.Serial(port, baudrate, timeout=1)
            self.connected = True
            self._warmup()
            return True
        except:
            self.connected = False
            return False

    def _warmup(self):
        for _ in range(10):
            self.update()
            time.sleep(0.1)

    def update(self):
        if not self.connected: return False
        
        try:
            line = self.serial_port.readline().decode('ascii', errors='replace')
            if line.startswith('$GPRMC'):
                msg = pynmea2.parse(line)
                if msg.latitude != 0 and msg.longitude != 0:
                    self.last_location = self.current_location
                    self.current_location = (msg.latitude, msg.longitude)
                    self.heading = msg.true_course if msg.true_course else self.heading
                    self.last_update = time.time()
                    return True
        except:
            pass
        return False

    def get_anomaly_vector(self, anomaly_position):
        if not self.current_location[0]: return (None, None)
        
        current = self.current_location
        anomaly = (anomaly_position[0]/1e6, anomaly_position[1]/1e6)  # Assuming position is in microdegrees
        distance = geodesic(current, anomaly).meters
        bearing = geodesic(current, anomaly).initial_bearing
        return (distance, bearing)
