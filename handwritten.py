from flask import Flask, request, render_template
from keras.models import load_model
import cv2
import numpy as np
import os
import imutils
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer


app = Flask(__name__, static_url_path='/static')

# Load the English and Arabic models
ENGLISH_MODEL_PATH = 'Models/OCR_handwrit.h5'
ARABIC_MODEL_PATH = 'Models/my_model.h5'

english_model = load_model(ENGLISH_MODEL_PATH)
arabic_model = load_model(ARABIC_MODEL_PATH)

LB = LabelBinarizer()
LB.fit_transform([i for i in range(28)])  # Assuming 28 classes for English and Arabic models

# Define the characters of the Arabic alphabet
chars_arabic = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']
chars_english = [chr(i) for i in range(97, 123)]  # Assuming lowercase English letters

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (32, 32))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    img = np.reshape(img, (1, 32, 32, 1))
    return img

# Function to predict the image
def predict_image(image_path, model, chars):
    preprocessed_image = preprocess_image(image_path)
    prediction = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_label = chars[predicted_class_index]
    return predicted_class_label

# Function to display the image with prediction
def display_image_with_prediction(image_path, model, chars):
    image = cv2.imread(image_path)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    prediction = predict_image(image_path, model, chars)
    plt.title(f"Predicted: {prediction}", fontsize=16)
    plt.axis('off')
    plt.show()

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    return (cnts, boundingBoxes)

def get_letters(img, model, chars):
    letters = []
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            thresh = cv2.resize(thresh, (32, 32), interpolation = cv2.INTER_CUBIC)
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = np.expand_dims(thresh, axis=0)
            ypred = model.predict(thresh)
            ypred = LB.inverse_transform(ypred)
            [x] = ypred
            letters.append(chars[x])
    return letters

# Define routes
@app.route('/')
def index():
    return render_template('handwritten.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)
        language = request.form['language']
        if language == 'english':
            model = english_model
            chars = chars_english
        else:
            model = arabic_model
            chars = chars_arabic
        predicted_letters = get_letters(file_path, model, chars)
        return render_template('handwritten.html', image_file=file_path, prediction=''.join(predicted_letters))
    
if __name__ == '__main__':
    app.run(debug=True)
