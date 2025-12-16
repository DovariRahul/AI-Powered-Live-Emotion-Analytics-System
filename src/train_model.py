import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# 1. PATHS - Double check these match your folders
TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax') 
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # 2. DATA PREPARATION
    # This part "looks" into your folders and prepares the images
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)

    print("Loading images...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical'
    )

    # 3. TRAINING
    model = build_model()
    print("Starting training... (This will take time)")
    
    # Create models folder if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    model.fit(
        train_generator,
        epochs=50,
        validation_data=test_generator
    )

    # 4. SAVE THE RESULT
    model.save('models/emotion_model.h5')
    print("Success! Brain file saved in models/emotion_model.h5")