import os
import sys
import csv
import argparse
import logging
import random
import numpy as np

from tqdm import tqdm
from colorama import Fore, init
from tensorflow import keras
from keras import layers, callbacks, regularizers
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
CNN_EPOCHS = 30
MLP_EPOCHS = 50
BATCH_SIZE = 128

LABEL_MAP = {0: "1", 1: "2", 2: "3", 3: "4", 4: "S", 5: "T", 6: "X"}
ALLOWED_LABELS = [1, 2, 3, 4, 28, 29, 33]

init(autoreset=True)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logger import setup_logger

def load_csv(file_path, logger_instance):
    data = []
    try:
        with open(file_path, "r") as f:
            total = sum(1 for _ in f)
            f.seek(0)
            reader = csv.reader(f)
            for row in tqdm(reader, total=total, desc=f"Loading {os.path.basename(file_path)}"):
                data.append(row)
    except Exception as e:
        logger_instance.error(f"Error reading file {file_path}: {e}")
        raise
    try:
        data = np.array(data, dtype=float)
    except Exception as e:
        logger_instance.error(f"Error converting CSV data to float from {file_path}: {e}")
        raise
    return data[np.isin(data[:, 0], ALLOWED_LABELS)]

def preprocess_data(dataset):
    labels = dataset[:, 0]
    images = dataset[:, 1:].reshape(-1, 28, 28, 1) / 255.0
    remap = {val: i for i, val in enumerate(ALLOWED_LABELS)}
    mapped_labels = np.array([remap[label] for label in labels])
    return images, mapped_labels

def test_random_sample(model, test_images, test_labels, model_name, logger_instance):
    idx = random.randint(0, test_images.shape[0] - 1)
    sample_image = test_images[idx: idx + 1]
    pred_probs = model.predict(sample_image)
    actual, predicted = int(test_labels[idx]), int(np.argmax(pred_probs, axis=1)[0])
    logger_instance.info(f"[{model_name}] Random sample prediction:")
    logger_instance.info(f"  Actual label (0-6): {actual} ({LABEL_MAP[actual]})")
    logger_instance.info(f"  Predicted label (0-6): {predicted} ({LABEL_MAP[predicted]})")

def build_mlp(input_shape, num_classes, use_bn=False):
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        *( [layers.BatchNormalization()] if use_bn else [] ),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        *( [layers.BatchNormalization()] if use_bn else [] ),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def build_cnn(input_shape, num_classes, use_bn=False):
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(1e-4), input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        *( [layers.BatchNormalization()] if use_bn else [] ),
        layers.Dropout(0.3),
        layers.Conv2D(64, (3, 3), activation="relu", padding="same",
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.MaxPooling2D((2, 2)),
        *( [layers.BatchNormalization()] if use_bn else [] ),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        *( [layers.BatchNormalization()] if use_bn else [] ),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def save_model(model, model_name):
    base_path = os.path.join(os.getcwd(), "model", "out")
    os.makedirs(base_path, exist_ok=True)
    save_file = os.path.join(base_path, f"{model_name}.keras")
    if os.path.exists(save_file):
        input(f"{Fore.RED}Model {model_name} already exists. Press Enter to overwrite or Ctrl+C to cancel...{Fore.RESET}")
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
    parser = argparse.ArgumentParser(description="Train MLP and CNN on a filtered EMNIST subset.")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--es-threshold", type=float, default=0.001, help="EarlyStopping min_delta.")
    parser.add_argument("--es-patience", type=int, default=3, help="EarlyStopping patience.")
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--use-batchnorm", action="store_true")
    args = parser.parse_args()

    logger_instance = setup_logger()
    if args.debug:
        logger_instance.setLevel(logging.DEBUG)
        logger_instance.debug("Debug enabled!")

    dataset_base = os.path.join(os.getcwd(), "model", "dataset")
    train_csv = os.path.join(dataset_base, "emnist-balanced-train.csv")
    test_csv = os.path.join(dataset_base, "emnist-balanced-test.csv")

    logger_instance.info("Loading and filtering datasets...")
    train_data = load_csv(train_csv, logger_instance)
    test_data = load_csv(test_csv, logger_instance)
    train_images, train_labels = preprocess_data(train_data)
    test_images, test_labels = preprocess_data(test_data)
    logger_instance.info(f"Training data shape: {train_images.shape}")
    logger_instance.info(f"Test data shape: {test_images.shape}")

    num_classes = len(ALLOWED_LABELS)
    early_stopping = None
    if not args.disable_early_stopping:
        early_stopping = callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=args.es_threshold,
            patience=args.es_patience,
            restore_best_weights=True,
            verbose=1,
        )

    logger_instance.info("Building and training MLP...")
    mlp = build_mlp(train_images.shape[1:], num_classes, args.use_batchnorm)
    mlp.fit(
        train_images,
        train_labels,
        epochs=MLP_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stopping] if early_stopping else [],
        verbose=1,
    )
    mlp_loss, mlp_acc = mlp.evaluate(test_images, test_labels, verbose=0)
    logger_instance.info(f"MLP -> Loss: {mlp_loss:.4f}, Accuracy: {mlp_acc:.4f}")

    logger_instance.info("Building and training CNN...")
    cnn = build_cnn(train_images.shape[1:], num_classes, args.use_batchnorm)
    cnn.fit(
        train_images,
        train_labels,
        epochs=CNN_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        callbacks=[early_stopping] if early_stopping else [],
        verbose=1,
    )
    cnn_loss, cnn_acc = cnn.evaluate(test_images, test_labels, verbose=0)
    logger_instance.info(f"CNN -> Loss: {cnn_loss:.4f}, Accuracy: {cnn_acc:.4f}")

    logger_instance.info("Testing random samples:")
    test_random_sample(mlp, test_images, test_labels, "MLP", logger_instance)
    test_random_sample(cnn, test_images, test_labels, "CNN", logger_instance)

    choice = input("Save models? (y=both, m=MLP, c=CNN): ").strip().lower()
    if choice == "y":
        save_model(mlp, "mlp")
        save_model(cnn, "cnn")
        logger_instance.info("Saved MLP and CNN models.")
    elif choice == "m":
        save_model(mlp, "mlp")
        logger_instance.info("Saved MLP model.")
    elif choice == "c":
        save_model(cnn, "cnn")
        logger_instance.info("Saved CNN model.")

    logger_instance.info("Plotting confusion matrix for CNN...")
    predictions = np.argmax(cnn.predict(test_images), axis=1)
    plot_confusion_matrix(test_labels, predictions, logger_instance)
    logger_instance.info("Finished!")

if __name__ == "__main__":
    main()
