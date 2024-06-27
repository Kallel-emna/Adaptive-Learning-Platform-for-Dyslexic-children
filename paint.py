from flask import Flask, render_template, request
import base64
import io
import os
from PIL import Image
import cv2
import numpy as np
from flask_mysqldb import MySQL
import MySQLdb.cursors
from keras.models import load_model

app = Flask(__name__)

app.secret_key = 'xyzsdfg'
  
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'user-system'

mysql = MySQL(app)


# Load the model
ARABIC_MODEL_PATH = 'Models/my_model.h5'
model = load_model(ARABIC_MODEL_PATH)

chars = ['ا', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر', 'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف', 'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي']

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  # Convert PIL image to numpy array
    image = cv2.resize(image, (32, 32))
    image = image.astype('float32') / 255.0
    image = np.reshape(image, (1, 32, 32, 1))
    return image

# Function to predict the image
def predict_image(image):
    preprocessed_image = preprocess_image(image)
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
    cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
    if request.method == 'POST':
        # Get the base64 image data from the form
        image_data = request.form.get('imageData')
        chosen_letter = request.form.get('chosenLetter')
        
        
        if image_data:
            # Decode the base64 image data
            image_data = image_data.split(',')[1]  # Remove data URL prefix
            decoded_data = base64.b64decode(image_data)
            
            # Convert the decoded data to a PIL image
            input_image = Image.open(io.BytesIO(decoded_data))
            
            # Predict the class of the image
            prediction = predict_image(input_image)
            # Return the prediction
            is_correct = (prediction == chosen_letter)
            if is_correct:
                # Mettre à jour les scores de l'utilisateur
                query = "INSERT INTO score_arabe ({}) VALUES (%s)".format(chosen_letter)

                # Execute the query with the value 1 as the parameter.
                cursor.execute(query, (1,))
                mysql.connection.commit()
            else:
                query = "INSERT INTO score_arabe ({}) VALUES (%s)".format(chosen_letter)

                # Execute the query with the value 1 as the parameter.
                cursor.execute(query, (2,))
                mysql.connection.commit()
            
            mysql.connection.commit()
            
            return render_template('handwAR.html', prediction=prediction)

        else:
            # Handle case when no image data is received
            return "No image data received."

if __name__ == '__main__':
    app.run(debug=True)
