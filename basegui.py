import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import cv2
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import os
import numpy as np

# Load the trained model
model = load_model('AMD.h5')

# Create a dictionary to map the class indices to their names
class_names = {0: 'AMD Free', 1: 'Presence of AMD'}

# Create a function to classify an image
def classify_image(image_path):

    img = image.load_img(image_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Feature extraction
    base_model = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_shape=(64, 64, 3))
    feature = base_model.predict(x)
    
    # Use the model to make a prediction
    new_features = np.array([feature])
    predictions = model.predict(np.expand_dims(new_features, axis=-1))
    
    # Get the class name of the predicted class
    class_idx = np.argmax(predictions[0])
    if predictions[0] > 0.040:
        class_idx = 1
    else:
        class_idx = 0
    print("Id = ", class_idx, "Predictions = ", predictions[0], "Classnames = ", class_names)
    print("Class = ", class_names[class_idx])
    class_name = class_names[class_idx]
    return class_name


# Create a function to open an image file dialog and display the selected image
def open_image_file():
    # Open a file dialog to select an image file
    file_path = filedialog.askopenfilename()
    if file_path:
        # Classify the selected image
        class_name = classify_image(file_path)
        print(file_path)
        # Update the GUI to display the selected image and its classification result
        img = Image.open(file_path)
        img = img.resize((224, 224))
        img = ImageTk.PhotoImage(img)
        image_label.configure(image=img)
        image_label.image = img
        class_label.configure(text='Classification: ' + class_name, font=('Arial', 16, 'bold'), fg='red')
        header_label.configure(text='Disease Detection', font=('Arial', 20, 'bold'), fg='blue')

# Create the GUI
root = tk.Tk()
root.title('DR Classification')
root.geometry('400x450')

# Add a header label
header_label = tk.Label(root, text='Disease Detection', font=('Arial', 24, 'bold'), fg='blue')
header_label.pack(pady=20)

# Create a button to open an image file dialog
button = tk.Button(root, text='Open Image', command=open_image_file, font=('Arial', 16, 'bold'), bg='white', fg='black')
button.pack(pady=10)

# Create a label to display the selected image
image_label = tk.Label(root)
image_label.pack()

# Create a label to display the classification result
class_label = tk.Label(root, text='Classification: ', font=('Arial', 16, 'bold'), fg='red')
class_label.pack(pady=10)

root.mainloop()
