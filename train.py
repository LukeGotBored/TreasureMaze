import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configuration parameters
EPOCHS = 20
BATCH_SIZE = 128
WIDTH = 28
HEIGHT = 28

def load_data():
	base_dir = os.path.dirname(os.path.abspath(__file__))
	dataset_dir = os.path.join(base_dir, "dataset")
	train_path = os.path.join(dataset_dir, "emnist-balanced-train.csv")
	test_path = os.path.join(dataset_dir, "emnist-balanced-test.csv")
	
	if not os.path.exists(train_path) or not os.path.exists(test_path):
		sys.stderr.write("Dataset not found.\n")
		sys.exit(1)

	dataset = np.loadtxt(train_path, delimiter=',', skiprows=1)
	
