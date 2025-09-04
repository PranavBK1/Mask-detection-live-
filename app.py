from flask import Flask, render_template, Response, jsonify
import os
import traceback

import cv2
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'mask_detector.keras')
HAAR_PATH = os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')

# Load model safely
model = None
try:
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
        print('Loaded model:', MODEL_PATH)
    else:
        print('Model file not found:', MODEL_PATH)
except Exception as e:
    print('Failed to load model:', MODEL_PATH)
    traceback.print_exc()

# Load Haar cascade and verify
print('Haarcascade path:', HAAR_PATH)
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
if face_cascade.empty():
    print('Failed to load Haar cascade. Check the file path and contents:', HAAR_PATH)
else:
    print('Loaded Haar cascade successfully')


def gen_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print('Cannot open camera (index 0).')
        return

    try:
        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print('Failed to read frame from camera')
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = []
            try:
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            except Exception:
                # detection can fail if cascade not loaded
                print('Face detection failed')
                traceback.print_exc()

            for (x, y, w, h) in faces:
                # defensive slicing
                y1, y2 = max(0, y), y + h
                x1, x2 = max(0, x), x + w
                face_img = frame[y1:y2, x1:x2]
                try:
                    face_img = cv2.resize(face_img, (100, 100))
                    face_input = face_img.astype('float32') / 255.0
                    face_input = np.expand_dims(face_input, axis=0)

                    if model is not None:
                        pred = model.predict(face_input, verbose=0)
                        # support both (1,1) and (1,) outputs
                        try:
                            prob = float(np.squeeze(pred))
                        except Exception:
                            prob = float(pred[0][0])
                        label = 'Mask' if prob > 0.5 else 'No Mask'
                        color = (0, 255, 0) if label == 'Mask' else (0, 0, 255)
                    else:
                        label = 'Model missing'
                        color = (0, 255, 255)
                except Exception:
                    label = 'Error'
                    color = (0, 0, 255)
                    traceback.print_exc()

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        camera.release()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/health')
def health():
    return jsonify({
        'model_exists': os.path.exists(MODEL_PATH),
        'model_loaded': model is not None,
        'haarcascade_exists': os.path.exists(HAAR_PATH),
        'haarcascade_loaded': not face_cascade.empty()
    })


if __name__ == '__main__':
    # bind to all interfaces to make it easy to open in browser from host
    app.run(host='0.0.0.0', port=5000, debug=True)
