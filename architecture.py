import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import os
import numpy as np

features = []
label = []

folder_path = "./1"
image_files = os.listdir(folder_path)

for images in image_files:
    print(images)
    img = image.load_img(os.path.join(folder_path,images), target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Feature extraction
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
    feature = base_model.predict(x)
    features.append(feature)
    label.append(1)

folder_path = "./0"
image_files = os.listdir(folder_path)

for images in image_files:
    print(images)
    img = image.load_img(os.path.join(folder_path,images), target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Feature extraction
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
    feature = base_model.predict(x)
    features.append(feature)
    label.append(0)

# Load the input image features
features = np.array(features) # Replace with the code to extract features from the input image

# Load the corresponding label for the input image (assuming binary classification)
label = np.array(label) # Replace with the code to load the label for the input image

# Split the data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(features, label, test_size=0.2, random_state=42)

# Build the custom CNN architecture
model = Sequential([
    Flatten(input_shape=X_train.shape[1:]),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate model
score = model.evaluate(np.expand_dims(X_val, axis=-1), y_val, verbose=0)

model.save("AMD.h5")