# train.py
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 100

train_dir = r"C:\Users\Pranav\Downloads\Face_Mask_detection\face-mask-dataset\Dataset\train\train"
val_dir = r"C:\Users\Pranav\Downloads\Face_Mask_detection\face-mask-dataset\Dataset\test\test"

datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_directory(
    train_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=32, class_mode='binary'
)
val_gen = datagen.flow_from_directory(
    val_dir, target_size=(IMG_SIZE, IMG_SIZE), batch_size=32, class_mode='binary'
)

model = Sequential([
    Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_gen, epochs=10, validation_data=val_gen)
model.save('mask_detector.keras')