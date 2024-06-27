from flask import Flask, request, render_template
import speech_recognition as sr
import threading

app = Flask(__name__)

# Initialize the recognizer
recognizer = sr.Recognizer()

# Variable to control recording
is_recording = False

# Function to start or stop recording
def start_stop_recording():
    global is_recording
    is_recording = not is_recording
    if is_recording:
        print("Recording started...")
    else:
        print("Recording stopped.")

# Function to record speech text
def record_text(language):
    while True:
        try:
            if is_recording:
                # Use the microphone as source for input
                with sr.Microphone() as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.2)

                    # Listen for the user's input
                    audio = recognizer.listen(source)

                    # Use Google to recognize audio
                    if language == 'english':
                        text = recognizer.recognize_google(audio)
                    elif language == 'french':
                        text = recognizer.recognize_google(audio, language="fr-FR")  # French

                    return text

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))

        except sr.UnknownValueError:
            print("Unknown error occurred")
    return

# Function to save text to a file
def output_text(text):
    with open("output.txt", "a") as f:
        f.write(text)
        f.write("\n")

# Function to run in a separate thread for recording
def record_thread(language):
    while True:
        text = record_text(language)
        output_text(f"{language.capitalize()}: {text}")
        print(f"{language.capitalize()}: {text}")

# Define routes
@app.route('/')
def index():
    return render_template('voice.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    start_stop_recording()
    return '', 204

@app.route('/record_english', methods=['POST'])
def record_english():
    threading.Thread(target=record_thread, args=('english',), daemon=True).start()
    return '', 204

@app.route('/record_french', methods=['POST'])
def record_french():
    threading.Thread(target=record_thread, args=('french',), daemon=True).start()
    return '', 204

if __name__ == '__main__':
    app.run(debug=True)
