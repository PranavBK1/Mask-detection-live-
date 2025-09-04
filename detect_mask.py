# detect_mask.py
import os
import cv2
from keras.models import load_model
import numpy as np

# Resolve model path (try project root .keras first, then legacy path)
BASE_DIR = os.path.dirname(__file__)
default_model = os.path.join(BASE_DIR, 'mask_detector.keras')
legacy_model = os.path.join(BASE_DIR, 'models', 'mask_detector.model')
model_path = default_model if os.path.exists(default_model) else legacy_model
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {default_model} or {legacy_model}")

model = load_model(model_path)

face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml'))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError('Could not open video capture (camera). Ensure a camera is connected and accessible.')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (100, 100))
        face_img = face_img / 255.0
        face_img = np.expand_dims(face_img, axis=0)

        try:
            pred = model.predict(face_img)
        except Exception as e:
            # if prediction fails, skip this face rather than crash
            print('Model prediction error:', e)
            continue

        # model.predict(...) returns an array; extract scalar probability robustly
        try:
            prob = float(pred.ravel()[0])
        except Exception:
            prob = float(np.asarray(pred).item())

        label = 'Mask' if prob > 0.5 else 'No Mask'
        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if label == 'No Mask':
            print("ALERT: No Mask Detected!")

    cv2.imshow('Face Mask Detector', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
