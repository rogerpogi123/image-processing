import os
# --- MAC OS FIX ---
# This prevents the "mutex lock failed" error on some Macs by allowing duplicate libraries
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tensorflow as tf
from tensorflow.keras.models import Sequential # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pyright: ignore[reportMissingImports]
from tensorflow.keras.preprocessing import image  # pyright: ignore[reportMissingImports]
import numpy as np 
import sys # ADDED: Required for reading command-line arguments (the uploaded image path)

# --- 1. CONFIGURATION ---
DATA_DIR = '/Users/allanmartinez/Downloads/ImageRecognizerProject/Dataset' 
SAVE_MODEL_PATH = 'my_animal_model.keras'
# NOTE: TEST_IMAGE_PATH is no longer used, as we read the path dynamically
# from sys.argv in Step 5. We keep the old path definition for reference but comment it out.
# TEST_IMAGE_PATH = ''/Users/allanmartinez/Downloads/images.jpeg'' 

IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32

# --- 2. DATA PREPARATION (WITH AUGMENTATION) ---
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

print("Loading Training Data...")
train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

class_names = list(train_generator.class_indices.keys())
print(f"Classes found: {class_names}")

# --- 3. MODEL ARCHITECTURE (SIMPLIFIED) ---
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'), 
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
model.fit(train_generator, epochs=10) 

print("Training complete!")

# --- 4. SAVE THE MODEL ---
print(f"Saving model to {SAVE_MODEL_PATH}...")
model.save(SAVE_MODEL_PATH)
print("Model saved successfully!")

# --- 5. PREDICTION (TESTING) ---

# Check for a command-line argument for the image path (passed by interface.py)
if len(sys.argv) > 1:
    # Use the path passed from the interface.py subprocess call
    TEST_IMAGE_PATH_DYNAMIC = sys.argv[1]
else:
    # Fallback to the hardcoded path if running manually without an argument
    # Using the path provided in the original script for consistency
    TEST_IMAGE_PATH_DYNAMIC = '/Users/allanmartinez/Downloads/images.jpeg' 
    print(f"Warning: No image path provided via command line. Using hardcoded path: {TEST_IMAGE_PATH_DYNAMIC}")


if os.path.exists(TEST_IMAGE_PATH_DYNAMIC):
    print(f"\nTesting image: {TEST_IMAGE_PATH_DYNAMIC}")
    img = image.load_img(TEST_IMAGE_PATH_DYNAMIC, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    # Using the raw, true prediction output without any hack
    result_index = np.argmax(score)
    result = class_names[result_index]
    confidence = 100 * np.max(score)

    print(f"\n--- PREDICTION RESULT (Raw Model Output) ---")
    print(f"Image: {os.path.basename(TEST_IMAGE_PATH_DYNAMIC)}")
    print(f"Predicted Class: {result}")
    print(f"Confidence: {confidence:.2f}%")
else:
    print(f"Warning: Test image not found at {TEST_IMAGE_PATH_DYNAMIC}")