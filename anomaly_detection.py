import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
from collections import deque

class AnomalyDetectionModule:
    def __init__(self):
        self.cluster_model = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=42
        )
        self.history = deque(maxlen=100)
        self.feature_scaler = np.array([1.0, 1.0, 0.01])

    def process(self, input_data, data_bus, register_bank):
        social_forces = register_bank.read('social_force')
        density_map = register_bank.read('density_map')
        detections = register_bank.read('object_detections')

        if None in [social_forces, density_map] or not social_forces:
            return

        features = []
        for sf in social_forces:
            x, y = sf['position']
            density = density_map[int(y), int(x)] if (
                0 <= y < density_map.shape[0] and 0 <= x < density_map.shape[1]
            ) else 0
            min_obj_dist = self._get_min_object_distance(sf['position'], detections)
            features.append([
                np.linalg.norm(sf['force']),
                density,
                min_obj_dist
            ])

        features = np.array(features) * self.feature_scaler

        if len(features) > 1:
            anomaly_scores = self.cluster_model.decision_function(features)
            anomaly_scores = 1 - (anomaly_scores - min(anomaly_scores)) / (max(anomaly_scores) - min(anomaly_scores) + 1e-8)
        else:
            anomaly_scores = [0.0] * len(features)

        results = []
        for i, sf in enumerate(social_forces):
            results.append({
                'id': sf['id'],
                'position': sf['position'],
                'anomaly_score': float(anomaly_scores[i]),
                'velocity': np.linalg.norm(sf['force']),
                'density': features[i][1] / self.feature_scaler[1]
            })

        register_bank.write('anomaly_scores', results)
        data_bus['anomaly_scores'] = results

    def _get_min_object_distance(self, position, detections):
        if not detections:
            return 1000
        min_dist = float('inf')
        pos = np.array(position)
        for obj in detections:
            obj_pos = np.array([
                obj['box'][0] + obj['box'][2] // 2,
                obj['box'][1] + obj['box'][3] // 2
            ])
            dist = distance.euclidean(pos, obj_pos)
            min_dist = min(min_dist, dist)
        return min_dist

