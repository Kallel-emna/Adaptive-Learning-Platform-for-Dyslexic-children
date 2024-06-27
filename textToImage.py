from flask import Flask, request, render_template
from googletrans import Translator
from diffusers import StableDiffusionPipeline
import os
import base64
import io
import torch

app = Flask(__name__)

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

@app.route('/', methods=['GET', 'POST'])
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

if __name__ == '__main__':
    app.run(debug=True)
