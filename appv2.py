from flask import Flask, request, jsonify
import pickle
import re

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the CountVectorizer
with open('count_vectorizer.pkl', 'rb') as f:
    count_vect = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    
    # Preprocess the input text
    text = text.lower()
    text = re.sub(r'[^\w\s]+', ' ', text)
    text = re.sub(r'_+', ' ', text)
    text = re.sub('\s+', ' ', text)

    # Vectorize the input text
    text_vectorized = count_vect.transform([text]).toarray()
    
    # Make the prediction
    predicted_probabilities = model.predict_proba(text_vectorized)[0]
    top_3_indices = predicted_probabilities.argsort()[-3:][::-1]
    top_3_classes = model.classes_[top_3_indices]
    top_3_probabilities = predicted_probabilities[top_3_indices]
    
    # Create the response dictionary with top 3 predictions
    response = {
        'predictions': [
            {'class': cls, 'probability': prob}
            for cls, prob in zip(top_3_classes, top_3_probabilities)
        ]
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run()
