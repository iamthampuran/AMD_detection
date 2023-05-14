import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the input image
img_path = '../ODIR-5K/ODIR-5K/Training Images/43_left.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Feature extraction
base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
features = base_model.predict(x)

# Display the extracted features
print("Features extracted from the image:")
print(features)
