from flask import Flask, render_template, request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
import os

app = Flask(__name__)

# Load the model

ARABIC_MODEL_PATH = 'Models/my_model.h5'
model = load_model(ARABIC_MODEL_PATH)

chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']

# Function to preprocess the image
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    image = np.reshape(image, (1, 32, 32, 1))
    return image

# Function to predict the image
def predict_image(image_path):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = chars[predicted_class_index]
    return predicted_class_label

# Route for home page
@app.route('/')
def home():
    return render_template('handwAR.html')

# Route for image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the image file from the request
        file = request.files['image']
        if file:
             filename = file.filename
             file_path = os.path.join('static/uploads', filename)
             file.save(file_path)
        # Predict the class of the image
        prediction = predict_image(file_path)
        # Return the prediction
        return render_template('handwAR.html', prediction=prediction, image_file=file_path)

if __name__ == '__main__':
    app.run(debug=True)
