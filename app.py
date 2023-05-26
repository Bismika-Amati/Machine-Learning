from flask import Flask, jsonify, request
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Create a TextVectorization layer
vectorizer = TextVectorization(max_tokens=10000, output_mode='count')
vectorizer.adapt([])  # Adapt with an empty list to load the vocabulary

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get input data from the request
    text = data['text']  # Extract the text from the data

    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Vectorize the preprocessed text
    vectorized_text = vectorize_text(preprocessed_text)

    # Make the prediction
    prediction = make_prediction(vectorized_text)

    # Create the response
    response = {
        'prediction': prediction
    }

    return jsonify(response)

# Preprocess the input text
def preprocess_text(text):
    text = text.lower()
    text = text.replace(r'[^\w\s]+', ' ')
    text = text.replace(r'_+', ' ')
    text = text.replace('\s+', ' ', regex=True)
    return text

# Vectorize the preprocessed text
def vectorize_text(text):
    vectorized_text = vectorizer([text]).numpy()
    return vectorized_text

# Make the prediction
def make_prediction(vectorized_text):
    prediction = model.predict(vectorized_text)
    predicted_class = np.argmax(prediction)
    return predicted_class

if __name__ == '__main__':
    app.run()
