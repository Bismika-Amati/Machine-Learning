#Library
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Load the dataset .csv with ',' separator and convert to TensorFlow Dataset
def inputDataset(path):
    data = pd.read_csv(path, sep=',', low_memory=False)
    df = pd.DataFrame(data)
    dataset = tf.data.Dataset.from_tensor_slices(dict(df))
    return dataset

# Data cleaning
def dataCleaning(df, input_column):
    df[input_column] = df[input_column].map(str)
    df[input_column] = df[input_column].str.lower()
    df[input_column] = df[input_column].str.replace(r'[^\w\s]+', ' ')
    df[input_column] = df[input_column].str.replace(r'_+', ' ')
    df[input_column] = df[input_column].str.replace('\s+', ' ', regex=True)
    return df

# Dividing data into input and output
def inputOutput(df, input_column, output_column):
    x = df.loc[:, input_column]
    y = df.loc[:, output_column]
    print('Data Divided Successfully Into Input & Output')
    return x, y

# Create a TextVectorization layer
def textVectorization(x):
    vectorizer = TextVectorization(max_tokens=10000, output_mode='count')
    vectorizer.adapt(x)
    x_counts = vectorizer(x).numpy()
    return x_counts

# Convert labels to numerical values
def labelMapping(y):
    label_mapping = {label: index for index, label in enumerate(set(y))}
    y = [label_mapping[label] for label in y]
    return y

# Split the data into training and validation sets
def splitData(X_counts, y, test_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X_counts, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val

# Convert the data to TensorFlow tensors
def dataToTensor(X_train, y_train, X_val, y_val):
    X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train_tensor = tf.keras.utils.to_categorical(y_train)
    X_val_tensor = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val_tensor = tf.keras.utils.to_categorical(y_val)
    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor

# Define the model architecture
def modelArch(input_shape, output_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(input_shape,)))
    model.add(tf.keras.layers.Dense(output_shape, activation='softmax'))
    return model

# Compile the model
def compileModel(model):
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the model
def trainModel(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, verbose=1):
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=verbose)
    return model

# Evaluate the model
def evaluateModel(model, X_val, y_val):
    _, accuracy = model.evaluate(X_val, y_val)
    return print('Accuracy of the Model is: {:.2f}%'.format(accuracy * 100))


# Example usage
dataset = inputDataset('Deskripsi_Permasalahan.csv')
df = pd.DataFrame(dataset.as_numpy_iterator())  # Convert the dataset back to a DataFrame

# Data cleaning
cleaned_df = dataCleaning(df, 'Uraian')

# Dividing data into input and output
x, y = inputOutput(cleaned_df, 'Uraian', 'Topik')

# Create a TextVectorization layer
x_counts = textVectorization(x)

# Convert labels to numerical values
y_mapped = labelMapping(y)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = splitData(x_counts, y_mapped)

# Convert the data to TensorFlow tensors
X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor = dataToTensor(X_train, y_train, X_val, y_val)

# Define the model architecture
model = modelArch(input_shape=X_train_tensor.shape[1], output_shape=len(set(y)))

# Compile the model
model = compileModel(model)

# Train the model
model = trainModel(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)

# Evaluate the model
evaluateModel(model, X_val_tensor, y_val_tensor)
