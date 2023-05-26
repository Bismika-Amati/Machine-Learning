from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Load the saved model
model = tf.keras.models.load_model('model.h5')

# Create the Flask app
app = Flask(__name__)

# Create a TextVectorization layer and adapt it to the training data
vectorizer = TextVectorization(max_tokens=10000, output_mode='count')
vectorizer.adapt(X)  # Adapt with the same X used during training

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = request.json
    uraian = data['uraian']

    # Preprocess the input data
    uraian_counts = vectorizer(uraian).numpy()

    # Convert the preprocessed data to TensorFlow tensor
    input_data = tf.convert_to_tensor(uraian_counts, dtype=tf.float32)

    # Make predictions using the loaded model
    predictions = model.predict(input_data)

    # Get the predicted labels
    predicted_labels = [list(label_mapping.keys())[pred.argmax()] for pred in predictions]

    # Return the predicted labels as the API response
    response = {'topik': predicted_labels}
    return jsonify(response)

if __name__ == '__main__':
    app.run()
