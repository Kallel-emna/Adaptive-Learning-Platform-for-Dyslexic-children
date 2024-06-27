from flask import Flask, render_template, request, url_for
import pandas as pd
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os

app = Flask(__name__)

# Load DataFrame containing image paths and descriptions
result_dataset = pd.read_excel("C:/Users/kalle/Downloads/fr5.xlsx")

# Add prefix to image paths
result_dataset['image_path'] = "C:/Users/kalle/Downloads/" + result_dataset['image_path'].astype(str)

def compare_images(image1_path, image2_path):
    # Open and convert images to grayscale
    image1 = Image.open(image1_path).convert('L')
    image2 = Image.open(image2_path).convert('L')

    # Resize images to a common size
    common_size = (256, 256)  # Choose a common size
    image1 = image1.resize(common_size)
    image2 = image2.resize(common_size)

    # Convert images to numpy arrays
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # Compute structural similarity between images
    similarity_score = ssim(image1_array, image2_array)

    return similarity_score

def find_best_match(image_path):
    similarities = []

    # Compare the given image with all images in the DataFrame
    for idx, row in result_dataset.iterrows():
        dataset_image_path = row["image_path"]
        similarity = compare_images(image_path, dataset_image_path)
        similarities.append((idx, similarity))

    # Find the index of the best match
    best_match_idx, best_similarity = max(similarities, key=lambda x: x[1])

    # Get the description of the best match
    best_description = result_dataset.loc[best_match_idx, "description"]
    
    return best_description

@app.route('/')
def index():
    return render_template('similarity.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image uploaded", 400
    
    image = request.files['image']
    if image.filename == '':
        return "No image selected", 400
    
    if image:
        filename = image.filename
        image_path = os.path.join('static/uploads', filename)
        image.save(image_path)

    # Find the best match for the uploaded image
    description = find_best_match(image_path)


    # Render template with best match description and uploaded image URL
    return render_template('similarity.html', description=description, image_path=image_path)


if __name__ == '__main__':
    app.run(debug=True)
