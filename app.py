import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXT = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global model variable
model = None

def load_model():
    global model
    if model is None:
        print("Loading MobileNetV2 model...")
        # Load pre-trained MobileNetV2 (trained on ImageNet)
        model = MobileNetV2(weights='imagenet')
        print("Model loaded successfully.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

def preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    x = np.array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) # MobileNetV2 specific preprocessing
    return x

@app.route('/')
def home():
    # Main application page
    return render_template('home.html', model_ready=True)

@app.route('/intro')
def intro():
    # Project introduction page
    return render_template('index.html')

@app.route('/info')
def info():
    # Upload information page
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received prediction request")
    if 'file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return redirect(url_for('home'))

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    # Ensure model is loaded
    if model is None:
        load_model()

    try:
        x = preprocess_image(save_path)
        preds = model.predict(x)
        # decode_predictions returns a list of lists of tuples (class, description, probability)
        # We get the top 1 result
        decoded = decode_predictions(preds, top=3)[0]  # top 3 for better context if needed, but we show top 1
        
        # Example decoded: [('n02099712', 'Labrador_retriever', 0.85), ...]
        top_prediction = decoded[0]
        breed_name = top_prediction[1].replace('_', ' ')
        confidence = float(top_prediction[2])

        return render_template('output.html', 
                               image_url='uploads/' + filename, 
                               breed=breed_name, 
                               confidence=confidence)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('output.html', error=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    # Pre-load model on startup to avoid delay on first request
    # but inside if-main so it doesn't break imports if any
    try:
        load_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Ensure you have internet access to download weights on first run.")
        
    app.run(debug=True)
