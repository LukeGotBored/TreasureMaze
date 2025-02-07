import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, regularizers, callbacks, utils 

import matplotlib.pyplot as plt

from utils import log

# Configuration parameters
NUM_CLASSES = 7  # Our 7 filtered classes (1,2,3,4,S,T,X)
EPOCHS = 100
BATCH_SIZE = 128
FEATURE_VECTOR_LENGTH = 784  # 28x28 pixels

def load_data():
    base_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(base_dir, "dataset")
    train_path = os.path.join(dataset_dir, "emnist-balanced-train.csv")
    test_path = os.path.join(dataset_dir, "emnist-balanced-test.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        log("Dataset not found.", "ERROR")
        sys.exit(1)

    train_data = np.loadtxt(train_path, delimiter=',', skiprows=1)
    test_data = np.loadtxt(test_path, delimiter=',', skiprows=1)
    log("Applying preprocessing...", "INFO")

    # Split features and labels
    y_train, x_train = train_data[:, 0], train_data[:, 1:]
    y_test, x_test = test_data[:, 0], test_data[:, 1:]
    
    # Filter specific classes and remap labels 0 to 6
    filtered_classes = [1, 2, 3, 4, 45, 46, 50]  # S=45, T=46, X=50
    train_mask = np.isin(y_train, filtered_classes)
    test_mask = np.isin(y_test, filtered_classes)

    # Apply masks
    x_train, y_train = x_train[train_mask], y_train[train_mask]
    x_test, y_test = x_test[test_mask], y_test[test_mask]

    # Remap labels to 0-6
    label_mapping = {old: new for new, old in enumerate(filtered_classes)}
    y_train = np.array([label_mapping[yy] for yy in y_train])
    y_test = np.array([label_mapping[yy] for yy in y_test])
    
    # Prepare data for models
    x_train = x_train.reshape(x_train.shape[0], FEATURE_VECTOR_LENGTH)
    x_test = x_test.reshape(x_test.shape[0], FEATURE_VECTOR_LENGTH)

    # Normalize pixel values
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return x_train, y_train, x_test, y_test


def main():
    # Load data
    x_train, y_train, x_test, y_test = load_data()
    log("Data loaded successfully.", "INFO")
    
    # Create model (TODO: Improve model)
    model = keras.Sequential([
        # Input layer, feature vector length is 784 (28x28 pixels)
        layers.Input(shape=(FEATURE_VECTOR_LENGTH,)),
        # Batch normalization is used to normalize the input layer by adjusting and scaling the activations (speeds up training)
        layers.BatchNormalization(),
        # Regularization is used to prevent overfitting by adding a penalty to the loss function
        layers.Dense(512, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
        # Dropout is used to prevent overfitting by randomly setting a fraction rate of input units to 0 at each update during training time
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(256, kernel_regularizer=regularizers.l2(0.001), activation='relu'),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        # Output layer, 7 classes
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.summary()
    
    # Compile & fit model
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

    # Evaluate model
    loss, accuracy = model.evaluate(x_test, y_test)
    log(f"Test loss: {loss}, Test accuracy: {accuracy}", "INFO")
    
    # Predict class
    predict_x = model.predict(x_test[:min(1, len(x_test))])
    classes_x = np.argmax(predict_x, axis=1)
    log(f"Predicted class: {classes_x}", "INFO")
    log(f"Sum: {np.sum(predict_x)}", "INFO")
    log(f"Class: {predict_x}", "INFO")
    
    import matplotlib.pyplot as plt
    x_test_vis = x_test[0].reshape(28, 28)
    _ = plt.imshow(x_test_vis, cmap = plt.cm.binary)
    
    history.history.keys()
    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()
    
    # Save model
    model.save("model/model.keras")
    log("Model saved successfully.", "INFO")
    
    # Save history
    np.save("model/history.npy", history.history)
    log("History saved successfully.", "INFO")
    
    
if __name__ == "__main__":
    main()