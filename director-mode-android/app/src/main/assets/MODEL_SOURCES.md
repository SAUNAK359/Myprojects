# Bundled runtime model sources

The CI build downloads these official model assets and packages them inside the APK. The installed app does not download models and requests no INTERNET permission.

- MediaPipe Pose Landmarker Lite (Google / MediaPipe):
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
- MediaPipe Face Landmarker (Google / MediaPipe):
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
- EfficientDet-Lite0 object detector (Google / MediaPipe model zoo):
  https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float32/1/efficientdet_lite0.tflite

Runtime libraries:
- AndroidX CameraX 1.6.2
- MediaPipe Tasks Vision 1.1.1
- LiteRT 2.3.0

The source project itself does not vendor model binaries so repository history stays manageable; the release APK/AAB built by CI contains the model files as local assets.
