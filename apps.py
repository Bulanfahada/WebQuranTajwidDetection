from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import numpy as np
import os
from flask_cors import CORS
import librosa

app = Flask(__name__)

# Load LSTM models
model_lstm_91 = load_model("model_lstm_91%.h5")
model_lstm_94 = load_model("model_lstm_94%.h5")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'wav'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(file_path):
    """Extract MFCC features from the audio file."""
    try:
        audio, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

@app.route("/", methods=['GET', 'POST'])
def main():
    """Render the main page."""
    return render_template("quranrecitation.html")

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    """Render the classification page."""
    return render_template("classifications.html")

@app.route('/submit', methods=['POST'])
def predict():
    """Handle the file upload and prediction."""
    if 'file' not in request.files or request.files['file'].filename == '':
        resp = jsonify({'message': 'No file selected'})
        resp.status_code = 400
        return resp
    
    file = request.files['file']
    filename = file.filename
    extension = filename.rsplit('.', 1)[1].lower()

    print(f"Uploaded file: {filename}, Extension: {extension}")  # Debugging: Print the filename and extension
    
    if file and allowed_file(filename):
        filename = "temp_audio.wav"  # Using a static name for testing; consider using a unique name in production
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Verify that the file was saved
        if os.path.exists(file_path):
            print(f"File saved successfully at: {file_path}")
        else:
            print("File was not saved correctly.")
            resp = jsonify({'message': 'File could not be saved'})
            resp.status_code = 500
            return resp
    else:
        print(f"File extension not allowed: {filename}")  # Debugging: Check if the file is disallowed
        resp = jsonify({'message': f'File type of {filename} is not allowed'})
        resp.status_code = 400
        return resp

    # Extract features from the audio file
    features = extract_features(file_path)
    if features is None:
        resp = jsonify({'message': 'Could not process audio file'})
        resp.status_code = 500
        return resp

    features = features.reshape(1, 1, -1)  # reshape for LSTM input

    # Predict using both models
    prediction_lstm_91 = model_lstm_91.predict(features)
    prediction_lstm_94 = model_lstm_94.predict(features)

    # Define class names
    class_names = ['Incorrect', 'Correct']

    # Render the classification page with results
    return render_template("classifications.html", 
                           audio_path=file_path,
                           prediction_lstm_91=class_names[int(prediction_lstm_91 > 0.5)],
                           confidence_lstm_91=f'{prediction_lstm_91[0][0] * 100:.2f}%',
                           prediction_lstm_94=class_names[int(prediction_lstm_94 > 0.5)],
                           confidence_lstm_94=f'{prediction_lstm_94[0][0] * 100:.2f}%'
                          )

if __name__ == '__main__':
    app.run(debug=True)
