from flask import Flask, request, jsonify
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import re

# Load the trained model
model = tf.keras.models.load_model('model.h5')
model.summary()  # Optional: Print model summary to verify its structure

# Create the Flask app
app = Flask(__name__)

# Define a route to handle the API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the JSON data from the request
    uraian = data['uraian']  # Extract the 'uraian' input from the JSON data

    # Preprocess the input data
    uraian = uraian.lower()
    uraian = str(uraian)
    uraian = re.sub(r'[^\w\s]+', ' ', uraian)
    uraian = re.sub(r'_+', ' ', uraian)
    uraian = re.sub('\s+', ' ', uraian, flags=re.UNICODE)
    uraian = [uraian]
    uraian_counts = vectorizer(uraian).numpy()


    # Perform the prediction using the loaded model
    predictions = model.predict(uraian_counts)

    # Get the top 3 predicted labels and their probabilities
    top_labels = np.argsort(-predictions[0])[:3]  # Get the indices of the top 3 labels
    predicted_labels = [label for label, index in label_mapping.items() if index in top_labels]
    label_probabilities = predictions[0][top_labels]

    # Create a JSON response with the predicted labels and probabilities
    response = {'predictions': []}
    for label, probability in zip(predicted_labels, label_probabilities):
        response['predictions'].append({'label': label, 'probability': float(probability)})

    return jsonify(response)  # Return the JSON response
  # Return the JSON response

if __name__ == '__main__':
    # Load the dataset .csv with ',' separator
    data = pd.read_csv('deskripsi_permasalahan.csv', sep=',', low_memory=False)
    df = pd.DataFrame(data)
    print('Dataframe has successfully created')

    # Data cleaning
    input_column = 'Uraian'
    output_column = 'Topik'
    df[input_column] = df[input_column].map(str)
    df[input_column] = df[input_column].str.lower()
    df[input_column] = df[input_column].str.replace(r'[^\w\s]+', ' ')
    df[input_column] = df[input_column].str.replace(r'_+', ' ')
    df[input_column] = df[input_column].str.replace('\s+', ' ', regex=True)
    print('Data has successfully cleaned')

    # Dividing data into input and output
    x = df.loc[:, input_column]
    y = df.loc[:, output_column]
    print('Data divided successfully into input & output')

    # Create a TextVectorization layer
    vectorizer = TextVectorization(max_tokens=1904, output_mode='count')
    vectorizer.adapt(x)

    # Convert labels to numerical values
    label_mapping = {label: index for index, label in enumerate(set(y))}
    
    app.run()