from flask import Flask, render_template, request
import pandas as pd
import torch
import torchaudio
from datasets import Dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from gtts import gTTS
import os

app = Flask(__name__)

# Load your Wav2Vec2 processor and model here
processor = Wav2Vec2Processor.from_pretrained("C:/Users/kalle/OneDrive - ESPRIT/Esprit/4A/PI/Deploiement/Models/processor")
model = Wav2Vec2ForCTC.from_pretrained("C:/Users/kalle/OneDrive - ESPRIT/Esprit/4A/PI/Deploiement/Models/modelSYLL")


# Définir la fonction pour traiter les fichiers audio
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = torchaudio.load(batch["audio"])
    print(sampling_rate)
    resampler = torchaudio.transforms.Resample(sampling_rate, 16_000) # Les données originales étaient à 48 000 Hz. Vous pouvez modifier cela en fonction de votre entrée.
    batch["audio"] = resampler(speech_array).squeeze().numpy()
    return batch

# Définir la fonction pour segmenter le mot original en fonction de la transcription prédite
def segment_original_word(original_word, predicted_transcription):
    # Supprimer les diacritiques du mot original et de la transcription prédite
    original_word = remove_diacritics(original_word)
    predicted_transcription = remove_diacritics(predicted_transcription)

    # Diviser la transcription prédite par espace pour obtenir le nombre de caractères dans chaque syllabe
    predicted_syllables_lengths = [len(syllable) for syllable in predicted_transcription.split()]

    # Segmenter le mot original en fonction des longueurs de chaque syllabe prédite
    segmented_word = ""
    start_index = 0
    for length in predicted_syllables_lengths:
        if start_index + length <= len(original_word):
            segmented_word += original_word[start_index:start_index+length] + "-"
        else:
            segmented_word += original_word[start_index:] + "-"
            break
        start_index += length

    segmented_word = segmented_word.rstrip("-")
    return segmented_word

# Définir la fonction pour ajouter des tirets dans le mot original en fonction des segments dans le mot segmenté
def add_hyphens(original_word, segmented_word):
    # Initialiser le compteur
    segment_count = 0
    new_original_word = ""

    # Parcourir le mot segmenté
    for char in segmented_word:
        if char != "-":
            # Si le caractère est l'une des lettres spéciales, ajouter +1 au compteur
            if char in ["ئ", "أ", "إ"]:
                segment_count += 1
            else:
                # Compter chaque segment comme deux lettres
                segment_count += 2
        else:
            # Ajouter les lettres du mot original correspondant au segment
            new_original_word += original_word[:segment_count]
            # Ajouter le tiret sauf si c'est la dernière lettre et sa diacritique
            if original_word[segment_count-2] != original_word[-2] or original_word[segment_count-1] != original_word[-1]:
                new_original_word += "-"
            # Avancer dans le mot original
            original_word = original_word[segment_count:]
            # Réinitialiser le compteur pour le prochain segment
            segment_count = 0

    # Ajouter le reste du mot original
    new_original_word += original_word

    return new_original_word

# Fonction pour supprimer les diacritiques d'un mot
def remove_diacritics(word):
    diacritics = ['َ', 'ُ', 'ِ', 'ْ', 'ً', 'ٌ', 'ٍ']
    for char in diacritics:
        word = word.replace(char, "")
    return word

# Fonction pour conserver les diacritiques avec leurs positions et les lettres correspondantes
def retain_diacritics_with_position(word):
    diacritics_info = []
    diacritics = ['َ', 'ُ', 'ِ', 'ْ', 'ً', 'ٌ', 'ٍ']
    for char_index, char in enumerate(word):
        if char in diacritics:
            if char_index > 0:
                diacritics_info.append((char_index, char, word[char_index - 1]))
            else:
                diacritics_info.append((char_index, char, None))
    return diacritics_info

# Fonction pour restaurer les diacritiques au mot segmenté
def restore_diacritics_to_segmented_word(segmented_word, diacritics_info):
    word_with_diacritics = list(segmented_word)
    for position, diacritic, preceding_char in diacritics_info:
        if preceding_char is not None:
            try:
                preceding_char_index = word_with_diacritics.index(preceding_char)
                word_with_diacritics.insert(preceding_char_index + 1, diacritic)
            except ValueError:
                print(f"Character '{preceding_char}' not found in segmented word: {segmented_word}")
        else:
            word_with_diacritics.insert(0, diacritic)
    word_with_diacritics = "".join(word_with_diacritics)
    return word_with_diacritics

# Fonction pour ajouter les diacritiques au mot segmenté
def add_diacritics_to_segmented_word(segmented_word, diacritics_info):
    word_with_diacritics = list(segmented_word)
    index_offset = 0
    for position, diacritic, preceding_char in diacritics_info:
        position += index_offset
        if position < len(word_with_diacritics):
            # Trouver l'index de la lettre au-dessus de laquelle le diacritique doit être placé
            letter_index = position
            while word_with_diacritics[letter_index] == '-' and letter_index >= 0:
                letter_index -= 1
            # Ajouter le diacritique au-dessus de la lettre correspondante s'il n'est pas déjà ajouté
            if diacritic not in word_with_diacritics[letter_index]:
                word_with_diacritics[letter_index] += diacritic
        else:
            # Si la position est hors limites, ajouter le diacritique à la fin du mot
            word_with_diacritics.append(diacritic)
            index_offset += 1
    word_with_diacritics = "".join(word_with_diacritics)
    return word_with_diacritics

# Fonction pour convertir du texte en discours et sauvegarder le fichier audio
def text_to_speech(text, output_folder):
    tts = gTTS(text=text, lang='ar')
    filename = f"{text}.wav"
    filepath = os.path.join(output_folder, filename)
    tts.save(filepath)
    print(f"The word '{text}' has been converted to audio and saved as '{filename}' in '{output_folder}'.")
    return filepath
# Define the root route function
# Define the function to generate HTML with colored syllables
def generate_html_with_colored_syllables(segmented_word_with_diacritics):
    html_code = "<div style='font-size: 24px;'>"
    syllables = segmented_word_with_diacritics.split("-")
    pastel_colors = ["#FFCCCB", "#FFD700", "#7FFFD4", "#98FB98", "#F0E68C", "#D8BFD8", "#C1E1C1", "#87CEFA", "#FFB6C1", "#FFDEAD", "#C0C0C0", "#FFE4E1", "#FAF0E6"]
    for index, syllable in enumerate(syllables):
        if not syllable:
            continue
        html_code += f"<span style='background-color: {pastel_colors[index % len(pastel_colors)]}; padding: 2px;'>{syllable}</span>"
    html_code += "</div>"
    return html_code

@app.route('/')
def index():
    return render_template('syllableAR.html')

# Define a route to handle the form submission
@app.route('/split', methods=['POST'])
def split():
    # Initialize DataFrame for audio paths
    dftest = pd.DataFrame(columns=['audio'])

    # Get the phrase from the form
    phrase = request.form['phrase']

    # Split the phrase into words
    words = phrase.split()

    # Path to the main folder
    main_folder_path = "/static/uploads"

    # Create a folder for the phrase if it doesn't exist already
    phrase_folder_path = os.path.join(main_folder_path, phrase.replace(" ", "_"))
    if not os.path.exists(phrase_folder_path):
        os.makedirs(phrase_folder_path)

    # List to store segmented words with diacritics
    segmented_words_with_diacritics = []

    # Loop through each word in the phrase
    for word in words:
        # Convert the word to speech and save the audio file
        audio_path = text_to_speech(word, phrase_folder_path)

        # Update DataFrame with the audio path
        dftest.loc[len(dftest)] = [audio_path]

    # Create a Dataset object from the DataFrame
    dataset = Dataset.from_pandas(dftest)

    # Map audio files to arrays
    test_dataset = dataset.map(speech_file_to_array_fn)

   # Loop through each audio file and generate predictions
    for index, example in enumerate(test_dataset):
        word = words[index]
        inputs = processor(example["audio"], sampling_rate=16_000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values).logits
        # Ensure that logits is a tensor
        if isinstance(logits, torch.Tensor):
            logits = logits.numpy()
        else:
            # Handle the case when logits is a list
            logits = [logit.numpy() for logit in logits]
        transcription = processor.batch_decode(logits)[0]
        segmented_word = segment_original_word(word, transcription)
        diacritics_info = retain_diacritics_with_position(word)
        restored_segmented_word = restore_diacritics_to_segmented_word(segmented_word, diacritics_info)
        segmented_word_with_diacritics = add_hyphens(restored_segmented_word, segmented_word)
        segmented_words_with_diacritics.append(segmented_word_with_diacritics)

    # Join segmented words to form the segmented phrase
    segmented_phrase_with_diacritics = " ".join(segmented_words_with_diacritics)

    # Generate HTML with colored syllables
    html_content = generate_html_with_colored_syllables(segmented_phrase_with_diacritics)

    # Render the HTML template with the segmented phrase
    return render_template('syllableAR.html', result=html_content)


if __name__ == "__main__":
    app.run(debug=True)
