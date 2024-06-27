from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import threading

app = Flask(__name__)

# Initialize the recognizer
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

# Main route to render HTML template
@app.route('/')
def index():
    return render_template('voicefinal.html', latest_recorded_text=latest_recorded_text)

# Route to start/stop recording
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

if __name__ == '__main__':
    app.run(debug=True)
