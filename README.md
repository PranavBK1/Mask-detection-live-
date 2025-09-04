# Face Mask Detection

Lightweight face-mask detection project using a small Keras model, OpenCV Haar cascades for face detection, and a Flask-based live demo.

## Summary

This repository contains code to train a simple mask/no-mask classifier (`train.py`), run a standalone OpenCV-based detector (`detect_mask.py`), and serve a browser-based live demo (`app.py`) that streams webcam frames and overlays mask predictions. Utility scripts include `test_model.py` and `test_cascade.py` for quick checks.

## Files
- `train.py` — trains a small CNN on the dataset and saves `mask_detector.keras` in the project root.
- `app.py` — Flask app that streams webcam frames to `http://localhost:5000` and overlays predictions.
- `detect_mask.py` — standalone script that opens the camera, draws boxes and prints alerts to the console.
- `test_model.py` — CLI helper to run a single image through the saved model.
- `test_cascade.py` — checks that the Haar cascade XML loads correctly with OpenCV.
- `haarcascade_frontalface_default.xml` — Haar cascade file used for face detection.
- `requirements.txt` — python packages used by the project.
- `face-mask-dataset/` — dataset used for training (expected layout: `Dataset/train/train/{with_mask,without_mask}` and `Dataset/test/test/{with_mask,without_mask}`).

## Quickstart (PowerShell)

1. Create and activate a Python environment (recommended).

2. Install requirements:

```powershell
python -m pip install -r requirements.txt
```

3. Train the model (optional; large compute/time):

```powershell
# trains and saves mask_detector.keras in repo root
python train.py
```

4. Run the web demo:

```powershell
python app.py
# open http://localhost:5000 in your browser
```

5. Run the standalone detector (alternative to web demo):

```powershell
python detect_mask.py
```

6. Test the Haar cascade and model existence:

```powershell
python test_cascade.py
python test_model.py path/to/image.jpg
```





