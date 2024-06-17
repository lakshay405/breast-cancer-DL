import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras

# Load the breast cancer dataset
dataset = load_breast_cancer()

# Create a DataFrame from the data
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

# Add the target column to the DataFrame
df['target'] = dataset.target

# Display the first few rows of the DataFrame
print("First few rows of the dataset:")
print(df.head())

# Display the last few rows of the DataFrame
print("Last few rows of the dataset:")
print(df.tail())

# Dataset dimensions and summary statistics
print("Dataset dimensions:", df.shape)
print("Dataset information:")
print(df.info())
print("Statistical summary of the dataset:")
print(df.describe())
print("Missing values in the dataset:")
print(df.isnull().sum())
print("Class distribution (0: Malignant, 1: Benign):")
print(df['target'].value_counts())

# Separate features and target variable
features = df.drop(columns='target')
target = df['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Set up the neural network model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=10, verbose=1)

# Function to plot training history
def plot_history(hist):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(hist.history['accuracy'], label='Train Accuracy')
    plt.plot(hist.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(hist.history['loss'], label='Train Loss')
    plt.plot(hist.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# Visualize training history
plot_history(history)

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test accuracy: {test_accuracy:.4f}')

# Make predictions on the test set
predictions = model.predict(X_test_scaled)
predicted_labels = np.argmax(predictions, axis=1)
print("Predicted labels on test set:", predicted_labels)

# Example prediction on new data
example_input = np.array([11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888, 
                          0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406, 
                          0.001769, 12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715, 
                          0.2433, 0.06563]).reshape(1, -1)

# Scale the example input data
example_input_scaled = scaler.transform(example_input)

# Predict the label for the example input
example_prediction = model.predict(example_input_scaled)
example_label = np.argmax(example_prediction)

# Display the prediction result
print(f'Example Prediction: {example_label} ({"Malignant" if example_label == 0 else "Benign"})')
