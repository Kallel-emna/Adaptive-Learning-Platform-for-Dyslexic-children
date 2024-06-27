from flask import Flask, render_template, request
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
import os
from googletrans import Translator  # Import Translator from googletrans module

app = Flask(__name__)

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

translator = Translator()  # Create a Translator object

def predict_caption(image_path, target_lang):
    raw_image = Image.open(image_path).convert('RGB')
    
    # Conditional image captioning
    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    conditional_caption = processor.decode(out[0], skip_special_tokens=True)
    
    # Unconditional image captioning
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs)
    unconditional_caption = processor.decode(out[0], skip_special_tokens=True)
    
    conditional_translation = get_translation(conditional_caption, target_lang)
    unconditional_translation = get_translation(unconditional_caption, target_lang)
    
    return conditional_translation, unconditional_translation

@app.route('/')
def index():
    return render_template('im2.html')

@app.route('/predict_caption_route', methods=['POST'])
def predict_caption_route():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = file.filename
        file_path = os.path.join('static/uploads', filename)
        file.save(file_path)
        target_lang = request.form['target_lang']  # Get the selected target language
        conditional_caption, unconditional_caption = predict_caption(file_path, target_lang)
        return render_template('im2.html', image_file=file_path, conditional_caption=conditional_caption, unconditional_caption=unconditional_caption)

def get_translation(text, dest_lang):
    # Translate the text to the specified destination language
    translated_text = translator.translate(text, dest=dest_lang)
    return translated_text.text

if __name__ == '__main__':
    app.run(debug=True)
