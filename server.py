from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from langdetect import detect
from pyvi import ViTokenizer
import re
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load model and tokenizer
model_dir = os.path.join(os.getcwd(), "models")

with open(os.path.join(model_dir, "tokenizer.pkl"), 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model(os.path.join(model_dir, "toxic_comment_model.h5"))

MAX_SEQUENCE_LENGTH = 200

def clean_text(text, is_vietnamese=False):
    """Clean and preprocess the input text."""
    text = text.lower()
    
    # Tokenize and clean based on language
    if is_vietnamese:
        text = ViTokenizer.tokenize(text)
    else:
        text = re.sub(r"what's", "what is ", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"i'm", "i am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub('\W', ' ', text)  # Remove non-alphanumeric characters
        text = re.sub('\s+', ' ', text)  # Remove extra spaces

    return text.strip()

def predict_toxicity(text):
    """Predict whether the text is toxic or not."""
    try:
        lang = detect(text)
        is_vietnamese = lang == 'vi'
    except:
        is_vietnamese = False

    cleaned_text = clean_text(text, is_vietnamese=is_vietnamese)
    
    # Convert cleaned text to sequence
    sequences = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')

    # Predict the toxicity using the model
    prediction = model.predict(padded_sequence)[0][0]
    
    # Return the result based on the prediction
    return "Toxic" if prediction > 0.5 else "Non-Toxic"

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    """API endpoint to predict toxicity of the text."""
    data = request.get_json(force=True)
    text = data.get('text')
    prediction = predict_toxicity(text)
    return jsonify({"prediction": prediction})

@app.route('/predict', methods=['OPTIONS'])
@cross_origin() 
def options():
    """Handle CORS preflight requests."""
    response = jsonify({})
    response.status_code = 200
    return response

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
