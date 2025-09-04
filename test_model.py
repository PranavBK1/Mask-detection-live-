from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import sys

# Usage: python test_model.py path_to_image.jpg
if len(sys.argv) != 2:
    print("Usage: python test_model.py path_to_image.jpg")
    sys.exit(1)

img_path = sys.argv[1]
model = load_model('mask_detector.keras')
img = image.load_img(img_path, target_size=(100, 100))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)
prediction = model.predict(img_array)
result = "Mask" if prediction[0][0] > 0.5 else "No Mask"
print(f"Prediction: {result}")
