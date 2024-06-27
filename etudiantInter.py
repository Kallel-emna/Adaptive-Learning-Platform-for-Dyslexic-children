from flask import Flask, render_template, request,jsonify
from gtts import gTTS
from io import BytesIO
import pygame
import speech_recognition as sr
import threading
from googletrans import Translator
from diffusers import StableDiffusionPipeline
import base64
import torch
import os 
import soundfile as sf
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from skimage.metrics import structural_similarity as ssim
import numpy as np
import pandas as pd
import pyphen
import random


app = Flask(__name__)
audio_cache = {}

def play_audio(audio_data):
    pygame.mixer.init()
    pygame.mixer.music.load(BytesIO(audio_data))
    pygame.mixer.music.play()

def generate_word_audio(word, language):
    tts = gTTS(text=word, lang=language, slow=False)
    audio_bytes = BytesIO()
    tts.write_to_fp(audio_bytes)
    return audio_bytes.getvalue()

@app.route('/')
def etudiant_interface():
    return render_template('etudiantInter.html')


r = sr.Recognizer()

# Variable to control recording
is_recording = False
language = "en"
latest_recorded_text = ""  # Variable to store the latest recorded text

# Function to recognize text based on language
def record_text():
    global is_recording, language, latest_recorded_text
    while True:
        try:
            if is_recording:
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=0.2)
                    audio = r.listen(source)
                    if language == "ar":
                        try:
                            text = r.recognize_google(audio, language="ar-SA")  # Arabic
                            latest_recorded_text = text
                            print("Arabic:", text)  # Print only the generated text
                        except sr.UnknownValueError:
                            print("Could not understand audio")
                    elif language == "fr":
                        text = r.recognize_google(audio, language="fr-FR")  # French
                        latest_recorded_text = text
                        print("French:", text)  # Print only the generated text
                    else:
                        text = r.recognize_google(audio)  # English
                        latest_recorded_text = text
                        print("English:", text)  # Print only the generated text
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
            


# Function to start/stop recording
def start_stop_recording():
    global is_recording
    is_recording = not is_recording
    if is_recording:
        print("Recording started...")
    else:
        print("Recording stopped.")

# Function to run in a separate thread for recording
def record_thread():
    while True:
        record_text()

# Create a thread for recording
record_thread = threading.Thread(target=record_thread, daemon=True)
record_thread.start()

class CFG:
    device = "cpu"
    seed = 30
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 20
    image_gen_model_id = "CompVis/stable-diffusion-v1-4"
    image_gen_size = (200, 200)
    image_gen_guidance_scale = 7.5
    prompt_gen_model_id = "gpt3"
    prompt_dataset_size = 6
    prompt_max_length = 12

# Initialize Translator object
translator = Translator()

# Initialize the image generation model
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float32,
    revision="fp16", use_auth_token='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', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

def get_translation(text, dest_lang):
    # Translate the text to the specified destination language
    translated_text = translator.translate(text, dest=dest_lang)
    return translated_text.text

def generate_image(prompt, model, generate_cartoon=False):
    if generate_cartoon:
        prompt += " generate cartoon"

    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image


@app.route('/control', methods=['POST'])
def control():
    start_stop_recording()
    return "OK"

# Route to choose language and start recording
@app.route('/start', methods=['POST'])
def start():
    global language
    language = request.form['language']
    start_stop_recording()  # To start/stop recording based on current state
    return "OK"

# Route to get the latest recorded text
@app.route('/latest_text', methods=['GET'])
def get_latest_text():
    return jsonify({'latest_recorded_text': latest_recorded_text})



@app.route('/textToImage', methods=['GET', 'POST'])
def text_to_image():
    if request.method == 'POST':
        text = request.form['text']
        dest_lang = request.form['dest_lang']
        translation = get_translation(text, dest_lang)
        
        # Generate the image
        image = generate_image(translation, image_gen_model, generate_cartoon=True)
        
        # Save the image to static/uploads directory
        image_filename = f"{translation[:20]}.png"  # Generating a filename based on the translated text
        image_path = os.path.join(app.root_path, 'static', 'uploads', image_filename)
        image.save(image_path)
        
        # Construct URL to the saved image
        image_url = f"/static/uploads/{image_filename}"
        
        # Pass image URL to template
        return render_template('textToImage.html', image_url=image_url)
    
    return render_template('textToImage.html')

#############################image caption 
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


@app.route('/caption')
def caption():
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



####similarity 
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

@app.route('/similarity')
def similarity():
    return render_template('similarity.html')

#############################Syllable

def segment_into_syllables(text):
    # Create an instance of the Pyphen class for English language
    dic = pyphen.Pyphen(lang='en')
    
    # Tokenize the text into words
    words = text.split()

    # Initialize an empty list to store segmented syllables
    segmented_syllables = []

    # Iterate through each word and segment into syllables
    for word in words:
        # Segment the word into syllables using Pyphen
        syllables = dic.inserted(word).split('-')
        
        # Append segmented syllables to the list
        segmented_syllables.append(syllables)
    
    return segmented_syllables

def segment_into_syllables_FR(text):
    # Create an instance of the Pyphen class for English language
    dic = pyphen.Pyphen(lang='en')
    
    # Tokenize the text into words
    words = text.split()

    # Initialize an empty list to store segmented syllables
    segmented_syllables = []

    # Iterate through each word and segment into syllables
    for word in words:
        # Segment the word into syllables using Pyphen
        syllables = dic.inserted(word).split('-')
        
        # Append segmented syllables to the list
        segmented_syllables.append(syllables)
    
    return segmented_syllables

def generate_pastel_color():
    # Generate a pastel color with random RGB values
    r = random.randint(128, 255)
    g = random.randint(128, 255)
    b = random.randint(128, 255)
    return f'rgb({r}, {g}, {b})'

def generate_html_with_colored_syllables(segmented_syllables):
    # Create a HTML string
    html_code = ""

    # Pastel colors for coloring syllables
    pastel_colors = [
        "#FFCCCB", "#FFD700", "#7FFFD4", "#98FB98", "#F0E68C", "#D8BFD8", "#C1E1C1", "#87CEFA", "#FFB6C1", "#FFDEAD", 
        "#C0C0C0", "#FFE4E1", "#FAF0E6", "#FFA07A", "#B0C4DE", "#FF69B4", "#00FA9A", "#9370DB", "#8A2BE2", "#20B2AA", 
        "#FF6347", "#7B68EE", "#00BFFF", "#7FFF00", "#9932CC", "#32CD32", "#F08080", "#66CDAA", "#BA55D3", "#FF4500", 
        "#E6E6FA", "#FF1493", "#00CED1", "#00FF7F", "#48D1CC", "#9932CC", "#FF69B4", "#D2B48C", "#00FFFF", "#1E90FF", 
        "#FFD700", "#5F9EA0", "#7FFF00", "#FF8C00", "#FF69B4", "#F0E68C", "#6495ED", "#DDA0DD", "#B0E0E6", "#8A2BE2"
    ]

    # Loop through each word and syllables, assigning a pastel color to each syllable
    for word in segmented_syllables:
        for syllable in word:
            # Add a span tag with the colored syllable
            color = generate_pastel_color()  # Get a unique pastel color for each syllable
            html_code += f"<span style='background-color: {color}; padding: 2px;'>{syllable}</span>"
        # Add a non-breaking space between words
        html_code += "&nbsp;"

    return html_code

@app.route('/syllable', methods=['GET', 'POST'])
def syllable():
    colored_text = None
    if request.method == 'POST':
        text = request.form['text']
        # Segment the input text into syllables
        segmented_syllables = segment_into_syllables(text)
        # Call the function to generate HTML with colored syllables
        colored_text = generate_html_with_colored_syllables(segmented_syllables)
    return render_template('educationAng.html', colored_text=colored_text)

@app.route('/syllableFR', methods=['GET', 'POST'])
def syllableFR():
    colored_text = None
    if request.method == 'POST':
        text = request.form['text']
        # Segment the input text into syllables
        segmented_syllables = segment_into_syllables_FR(text)
        # Call the function to generate HTML with colored syllables
        colored_text = generate_html_with_colored_syllables(segmented_syllables)
    return render_template('educationFrench.html', colored_text=colored_text)

@app.route('/educationFrench', methods=['GET', 'POST'])
def education_french():
    if request.method == 'POST':
        word = request.form['word']

        # Check if audio for given word exists in cache
        if word in audio_cache:
            audio_data = audio_cache[word]
        else:
            # Generate audio
            audio_data = generate_word_audio(word, 'fr')
            # Store audio in cache
            audio_cache[word] = audio_data

        # Play audio
        play_audio(audio_data)
                
    
    return render_template('educationFrench.html',latest_recorded_text=latest_recorded_text)


@app.route('/educationAng', methods=['GET', 'POST'])
def education_english():
    if request.method == 'POST':
        word = request.form['word']

        # Check if audio for given word exists in cache
        if word in audio_cache:
            audio_data = audio_cache[word]
        else:
            # Generate audio
            audio_data = generate_word_audio(word, 'en')
            # Store audio in cache
            audio_cache[word] = audio_data

        # Play audio
        play_audio(audio_data)
    return render_template('educationAng.html', latest_recorded_text=latest_recorded_text)

@app.route('/educationAr', methods=['GET', 'POST'])
def education_arabe():
    if request.method == 'POST':
        word = request.form['word']

        # Check if audio for given word exists in cache
        if word in audio_cache:
            audio_data = audio_cache[word]
        else:
            # Generate audio
            audio_data = generate_word_audio(word, 'ar')
            # Store audio in cache
            audio_cache[word] = audio_data

        # Play audio
        play_audio(audio_data)
    return render_template('t3agriba.html', latest_recorded_text=latest_recorded_text)



# Initialize the recognize


if __name__ == '__main__':
    app.run(debug=True)
