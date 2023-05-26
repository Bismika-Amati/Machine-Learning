# -*- coding: utf-8 -*-
"""Text_Classifier_Amati.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-MLEfmtxbrmaYiEslRyAk0hyYEJKAIoK
"""

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import numpy as np

class TextClassificationModel:
    def __init__(self, input_shape, output_shape):
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(input_shape,)))
        self.model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, verbose=1):
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=verbose)

    def evaluate(self, X_val, y_val):
        _, accuracy = self.model.evaluate(X_val, y_val)
        return accuracy

    def save_model(self, filepath):
        self.model.save(filepath)


class TextClassificationPipeline:
    def __init__(self, path, input_column, output_column):
        self.path = path
        self.input_column = input_column
        self.output_column = output_column

    def input_dataset(self):
        data = pd.read_csv(self.path, sep=',', low_memory=False)
        df = pd.DataFrame(data)
        return df

    def data_cleaning(self, df):
        df[self.input_column] = df[self.input_column].map(str)
        df[self.input_column] = df[self.input_column].str.lower()
        df[self.input_column] = df[self.input_column].str.replace(r'[^\w\s]+', ' ')
        df[self.input_column] = df[self.input_column].str.replace(r'_+', ' ')
        df[self.input_column] = df[self.input_column].str.replace('\s+', ' ', regex=True)
        return df

    def input_output(self, df):
        x = df.loc[:, self.input_column]
        y = df.loc[:, self.output_column]
        print('Data Divided Successfully Into Input & Output')
        return x, y

    def text_vectorization(self, x):
        vectorizer = TextVectorization(max_tokens=10000, output_mode='count')
        vectorizer.adapt(x.tolist())  # Convert to list
        x_counts = vectorizer(x).numpy()
        return x_counts

    def label_mapping(self, y):
        label_mapping = {label: index for index, label in enumerate(set(y))}
        y = [label_mapping[label] for label in y]
        return y

    def split_data(self, X_counts, y, test_size=0.2, random_state=42):
        X_train, X_val, y_train, y_val = train_test_split(X_counts, y, test_size=test_size, random_state=random_state)
        return X_train, X_val, y_train, y_val

    def run_pipeline(self):
        # Load the dataset
        df = self.input_dataset()

        # Data cleaning
        cleaned_df = self.data_cleaning(df)

        # Dividing data into input and output
        x, y = self.input_output(cleaned_df)

        # Create a TextVectorization layer
        x_counts = self.text_vectorization(x)

        # Convert labels to numerical values
        y_mapped = self.label_mapping(y)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = self.split_data(x_counts, y_mapped)

        # Convert data to numpy arrays
        X_train = np.array(X_train)
        X_val = np.array(X_val)

        # Train the model
        model = TextClassificationModel(X_train.shape[1], len(set(y_train)))
        model.train(X_train, y_train, X_val, y_val)

        # Save the model
        model.save_model('model.h5')

        # Evaluate the model
        accuracy = model.evaluate(X_val, y_val)
        print('Accuracy of the Model is: {:.2f}%'.format(accuracy * 100))


pipeline = TextClassificationPipeline('Deskripsi_Permasalahan.csv', 'Uraian', 'Topik')
pipeline.run_pipeline()

model.save("model.h5")
print("Success")

from google.colab import files

# Download the saved model file
files.download('model.h5')

