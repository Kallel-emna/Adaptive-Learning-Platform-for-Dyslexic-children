from flask import Flask, render_template, request
import threading
import speech_recognition as sr

app = Flask(__name__)

# Initialize the recognizer
r = sr.Recognizer()

# Variable to control recording
is_recording = False

# Recorded text
recorded_text = ""

# Function to handle recording based on language
def record_text(language):
    global is_recording, recorded_text
    while True:
        try:
            if is_recording:
                with sr.Microphone() as source:
                    r.adjust_for_ambient_noise(source, duration=0.2)
                    audio = r.listen(source)

                    if language == "ar":
                        text = r.recognize_google(audio, language="ar-SA")
                    elif language == "fr":
                        text = r.recognize_google(audio, language="fr-FR")
                    else:
                        text = r.recognize_google(audio)
                    
                    return text

        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except sr.UnknownValueError:
            print("Unknown error occurred")
            
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


@app.route('/')
def index():
    return render_template('voice2.html', recorded_text=recorded_text)


@app.route('/start_stop_recording', methods=['POST'])
def start_stop_recording():
    global is_recording
    is_recording = not is_recording
    language = request.form['language']
    if is_recording:
        print("Recording started...")
        threading.Thread(target=record_thread, args=(language,), daemon=True).start()
    return ''



if __name__ == '__main__':
    app.run(debug=True)
