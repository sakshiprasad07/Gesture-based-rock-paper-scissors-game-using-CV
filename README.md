# Rock Paper Scissors — YOLOv8 Gesture Detection Game

A real-time Rock-Paper-Scissors game controlled entirely by hand gestures. Two players face a webcam — left hand = Player 1, right hand = Player 2 — and a custom-trained computer vision model detects each gesture live, applies game rules, and tracks the score on screen.

## Overview

Instead of clicking buttons or pressing keys, players simply show a rock, paper, or scissors hand shape to the camera. The system:

1. Captures live webcam video
2. Runs a fine-tuned YOLOv8 object detection model on each frame to detect and classify hand gestures
3. Sorts detections left-to-right to assign Player 1 / Player 2
4. Uses a stability buffer to confirm a gesture is held consistently (avoiding flicker/misfires) before locking it in
5. Applies classic Rock-Paper-Scissors rules to decide the round winner
6. Renders score, bounding boxes, lock-in progress bars, and result banners directly on the video feed

## Tech Stack

| Component | Purpose |
|---|---|
| **Python** | Core language |
| **OpenCV (`cv2`)** | Webcam capture, frame processing, UI drawing, display window |
| **Ultralytics YOLOv8** | Object detection model — locates and classifies hand gestures |
| **PyTorch** | Deep learning backend YOLOv8 runs on |
| **CUDA / NVIDIA GPU** | Accelerates model training and inference |

## Model Details

The gesture detector is a **fine-tuned YOLOv8s** model, built via transfer learning:

- **Base model:** `yolov8s.pt`, pretrained on the COCO dataset (80 general object classes)
- **Custom dataset:** ~6,000+ labeled images across 3 classes (`Rock`, `Paper`, `Scissors`), organized in Roboflow's YOLO format (`train/`, `valid/`, `test/`, `data.yaml`)
- **Training:** 50 epochs on an NVIDIA RTX 3050 (4GB VRAM), ~1.8 hours

### Results

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Paper | 0.973 | 0.899 | 0.950 | 0.746 |
| Rock | 0.943 | 0.939 | 0.951 | 0.752 |
| Scissors | 0.957 | 0.931 | 0.968 | 0.761 |
| **All** | **0.957** | **0.923** | **0.956** | **0.753** |

`mAP50` measures how well predicted bounding boxes match ground truth, combining precision (few false positives) and recall (few missed detections) at a 50% overlap threshold — ~95.6% here indicates strong, reliable detection across all three gestures.

## How the Game Loop Works

```
while True:
    read frame from webcam
    run YOLO inference → get bounding boxes, class labels, confidence scores
    sort detections left-to-right (leftmost = Player 1, next = Player 2)
    push each player's detected gesture into a rolling buffer
    if a player's buffer holds the SAME gesture for N consecutive frames → gesture is "locked in"
    once both players are locked in:
        apply Rock-Paper-Scissors rules
        update score
        show result banner
    draw UI (boxes, lock-in bars, score, banner) on frame and display it
```

The **stability buffer** is the key design choice that makes this usable in practice — a single frame's detection can be noisy or momentarily wrong, so gestures only count once they've been detected consistently over several frames in a row.

## Setup & Installation

```bash
pip install opencv-python ultralytics
```

Update `MODEL_PATH` in `rps_game.py` to point to your trained weights:

```python
MODEL_PATH = "runs/detect/train/weights/best.pt"
```

Run the game:

```bash
python rps_game.py
```

Press **Q** to quit.

## Training Your Own Model

If retraining on a new or expanded dataset:

```bash
yolo detect train data=data.yaml model=yolov8s.pt epochs=50 imgsz=640
```

Trained weights are saved to `runs/detect/train/weights/best.pt`.

## Engineering Challenges & Solutions

- **Restrictive enterprise security policy (Windows Device Guard / WDAC):** blocked `pip.exe` directly and later blocked PyTorch's native DLLs from loading on Windows.
  - *Fix:* used `python -m pip` to bypass the executable-level block; used **WSL (Windows Subsystem for Linux)** to bypass the Windows-level DLL restriction entirely for installing dependencies and training.
- **Missing trained model file:** the originally referenced `best.pt` no longer existed on the machine.
  - *Fix:* retrained from scratch using the existing labeled dataset and `data.yaml` config — verified via strong validation metrics that the retrained model matched expected performance.
- **Split environments for training vs. gameplay:** WSL has no webcam passthrough, so GPU-heavy training ran in WSL while the live, camera-dependent game runs natively in Windows.
- **Windows webcam access:** switched OpenCV's video backend to DirectShow (`cv2.CAP_DSHOW`) for more reliable camera access than the default Media Foundation backend on Windows.

## Possible Improvements

- Expand training data with more lighting conditions, skin tones, backgrounds, and hand angles for better generalization
- Add data augmentation and early stopping to training pipeline
- Try a larger model variant (`yolov8m`) if inference latency allows
- Replace left/right positional player assignment with proper object tracking for more robust multi-hand scenarios
- Add round history / best-of-N match mode
