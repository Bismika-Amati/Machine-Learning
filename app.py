from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pickle
import re
import psycopg2
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the CountVectorizer
with open('count_vectorizer.pkl', 'rb') as f:
    count_vect = pickle.load(f)

# Database Connection Configuration
db_config = {
    'host': '34.101.44.243',
    'database': 'dev_db',
    'user': 'postgres',
    'password': 'oA{-vc.5\LJ=I.mq'
}


def connect_db():
    try:
        conn = psycopg2.connect(**db_config)
        return conn
    except psycopg2.Error as error:
        return None


@app.route('/predict', methods=['POST'])
def predict():
    # Connect to the database
    conn = connect_db()
    if conn is None:
        error_message = 'Database connection error'
        response = make_response(jsonify({'error': error_message}), 500)
        return response

    try:
        # Request Body
        data = request.json
        villageId = data['villageId']
        text = data['text']
        createdAt = datetime.now()
        updatedAt = createdAt

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
        predictions_string = ', '.join(top_3_classes)

        cursor = conn.cursor()

        for topic in top_3_classes:
            id = str(uuid.uuid4())
            insert_query = "INSERT INTO \"ProblemStatement\" (id, description, \"villageId\", topic, \"createdAt\", \"updatedAt\") VALUES (%s, %s, %s, %s, %s, %s)"
            data = (id, text, villageId, topic, createdAt, updatedAt)
            cursor.execute(insert_query, data)

        conn.commit()
        cursor.close()

        # Create the response dictionary with top 3 predictions
        response = {
            'predictions': [
                {'class': cls, 'probability': prob}
                for cls, prob in zip(top_3_classes, top_3_probabilities)
            ]
        }

        return jsonify(response)
    except Exception as error:
        error_message = f"Prediction error: {str(error)}"
        response = make_response(jsonify({'error': error_message}), 500)
        return response
    finally:
        conn.close()


if __name__ == '__main__':
    app.run()
