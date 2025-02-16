import sys
import os
import numpy as np
import random
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from colorama import Fore, init
from tensorflow import keras
from keras import layers, callbacks, regularizers
from sklearn.metrics import confusion_matrix
from logger import setup_logger

# Configuration parameters
CNN_EPOCHS = 50
MLP_EPOCHS = 50
BATCH_SIZE = 128

# Mapping and allowed labels
LABEL_MAP = {0: "1", 1: "2", 2: "3", 3: "4", 4: "S", 5: "T", 6: "X"}
ALLOWED_LABELS = [1, 2, 3, 4, 28, 29, 33]  # 1,2,3,4, S(28), T(29), X(33)

# Initialize Colorama
init(autoreset=True)

# Append project root to sys.path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def load_csv(file_path, logger_instance):
    try:
        # Attempt to load CSV data using NumPy
        data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    except Exception as e:
        logger_instance.error(f"Error loading CSV file: {e}")
        return None

    # Here you might add further processing (e.g. cleaning or type conversion)
    try:
        data = np.array(data)
    except Exception as e:
        logger_instance.error(f"Error converting data to array: {e}")
        return None

    # Filter rows using allowed labels (first column)
    labels = data[:, 0]
    mask = np.isin(labels, ALLOWED_LABELS)
    return data[mask]

def preprocess_data(dataset):
    labels = dataset[:, 0]
    images = dataset[:, 1:].reshape(-1, 28, 28, 1) / 255.0
    remap = {val: i for i, val in enumerate(ALLOWED_LABELS)}
    mapped_labels = np.array([remap[label] for label in labels])
    return images, mapped_labels

def test_random_sample(model, test_images, test_labels, model_name, logger_instance):
    idx = random.randint(0, test_images.shape[0] - 1)
    sample_image = test_images[idx : idx + 1]
    actual_label = int(test_labels[idx])
    pred_probs = model.predict(sample_image)
    pred_class = int(np.argmax(pred_probs, axis=1)[0])
    logger_instance.info(f"[{model_name}] Random sample prediction:")
    logger_instance.info(f"  Actual label (0-6): {actual_label} ({LABEL_MAP[actual_label]})")
    logger_instance.info(f"  Predicted label (0-6): {pred_class} ({LABEL_MAP[pred_class]})")

def build_mlp(input_shape, num_classes, use_batchnorm=False):
    inputs = keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_cnn(input_shape, num_classes, use_batchnorm=False):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(
        32, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(1e-4)
    )(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)            
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.MaxPooling2D((2, 2))(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    if use_batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def save_model(model, model_name):
    base_path = os.path.join(os.getcwd(), "model", "out")
    os.makedirs(base_path, exist_ok=True)
    save_file = os.path.join(base_path, f"{model_name}.keras")
    if os.path.exists(save_file):
        os.remove(save_file)  # Overwrite existing model file
    model.save(save_file)
    print(f"{Fore.GREEN}Model saved as {save_file}{Fore.RESET}")

def plot_confusion_matrix(true_labels, predictions, logger_instance):
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - CNN")
    plt.tight_layout()
    plt.show()
    logger_instance.info("Confusion matrix plotted.")

def main():
    parser = argparse.ArgumentParser(
        description="Train MLP and CNN on a filtered subset of EMNIST."
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--es-threshold",
        type=float,
        default=0.001,
        help="Minimum change in loss to be considered an improvement.",
    )
    parser.add_argument(
        "--es-patience",
        type=int,
        default=3,
        help="Number of epochs with no improvement after which training will be stopped.",
    )
    parser.add_argument(
        "--disable-early-stopping", action="store_true", help="Disable early stopping."
    )
    parser.add_argument(
        "--use-batchnorm",
        action="store_true",
        help="Use batch normalization in the models.",
    )
    args = parser.parse_args()

    logger_instance = setup_logger(args.debug)
    if args.debug:
        logger_instance.setLevel(logging.DEBUG)
        logger_instance.debug("Debug mode enabled!")
    else:
        logger_instance.setLevel(logging.INFO)
    
    dataset_base = os.path.join(os.getcwd(), "model", "dataset")
    train_csv = os.path.join(dataset_base, "emnist-balanced-train.csv")
    test_csv = os.path.join(dataset_base, "emnist-balanced-test.csv")

    logger_instance.info("Loading and filtering datasets...")
    train_data = load_csv(train_csv, logger_instance)
    test_data = load_csv(test_csv, logger_instance)
    if train_data is None or test_data is None:
        logger_instance.error("Failed to load datasets.")
        return 1

    train_images, train_labels = preprocess_data(train_data)
    test_images, test_labels = preprocess_data(test_data)
    logger_instance.info(f"Training data shape: {train_images.shape}")
    logger_instance.info(f"Test data shape: {test_images.shape}")

    num_classes = len(ALLOWED_LABELS)
    early_stopping = None
    if not args.disable_early_stopping:
        early_stopping = callbacks.EarlyStopping(
            monitor="loss", min_delta=args.es_threshold,
            patience=args.es_patience, verbose=1
        )

    # Build and train MLP (placeholder for training logic)
    mlp_model = build_mlp(train_images.shape[1:], num_classes, use_batchnorm=args.use_batchnorm)
    # mlp_model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=MLP_EPOCHS,
    #               validation_data=(test_images, test_labels), callbacks=[early_stopping] if early_stopping else [])
    
    # Build and train CNN (placeholder for training logic)
    cnn_model = build_cnn(train_images.shape[1:], num_classes, use_batchnorm=args.use_batchnorm)
    # cnn_model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=CNN_EPOCHS,
    #               validation_data=(test_images, test_labels), callbacks=[early_stopping] if early_stopping else [])
    
    # Optionally test a random sample from one of the models
    test_random_sample(cnn_model, test_images, test_labels, "CNN", logger_instance)
    
    # Save the trained CNN model (as an example)
    save_model(cnn_model, "cnn")
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("Exiting...")
        exit(0)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)
