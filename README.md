# face-hand-body-cam

Real-time webcam overlay that detects and draws wireframe landmarks for faces, hands, and body pose simultaneously using MediaPipe.

## What it does

- Face mesh tessellation and contours
- Hand skeleton with finger connections (up to 2 hands)
- Full body pose skeleton
- Mirror-flipped view (selfie orientation)
- Press `q` to quit

## Requirements

- Python 3.8+
- Webcam

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Download the MediaPipe model files and place them in the project directory:
   - [`hand_landmarker.task`](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
   - [`face_landmarker.task`](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
   - [`pose_landmarker.task`](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task)

3. Run:

```bash
python wireframe.py
```
