# app.py

from flask import Flask, render_template, request
import pyphen
import random

app = Flask(__name__)

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

@app.route('/', methods=['GET', 'POST'])
def index():
    colored_text = None
    if request.method == 'POST':
        text = request.form['text']
        # Segment the input text into syllables
        segmented_syllables = segment_into_syllables(text)
        # Call the function to generate HTML with colored syllables
        colored_text = generate_html_with_colored_syllables(segmented_syllables)
    return render_template('syllable.html', colored_text=colored_text)

if __name__ == '__main__':
    app.run(debug=True)
