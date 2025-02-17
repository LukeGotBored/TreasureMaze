# TreasureMaze

TreasureMaze is a Python project for maze extraction using computer vision with OpenCV and pathfinding via advanced search algorithms. It also includes a GUI for visualizing maze analysis and a model training pipeline for digit recognition based on EMNIST.

## About

- **Maze Extraction & Segmentation:** Uses OpenCV to standardize and extract grids from maze images.  
- **Pathfinding Algorithms:** Implements BFS, DFS, UCS, Best First, and A* to solve the maze.
- **Digit Recognition:** A trained CNN (or MLP) model predicts maze cell values from image segments.
- **GUI Visualization:** Offers an interactive interface to load images, view analysis, and step through solution paths.
- **Model Training:** Includes a module to train and evaluate models on filtered EMNIST datasets.

## Requirements

- Python 3.x  
- OpenCV ≥ 4.0  
- NumPy  
- TensorFlow & Keras  
- Other requirements as listed in `requirements.txt`

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Maze Analyzer

To process a maze image and execute pathfinding, run:
```bash
python main.py -f path/to/image.jpg -m path/to/model.keras -a "A*"
```
Additional flags:
- `-d/--debug`: Enable debug output.
- `-t/--treasures`: Specify the number of treasures to find.

### Model Training

To train the digit recognition models (MLP/CNN) on the EMNIST dataset:
```bash
python model/train.py [--debug] [--use-batchnorm] [--disable-early-stopping]
```
Follow the prompts to optionally save the models.

## Contributing

This repository is mainly for educational purposes and research. Contributions are welcome but not actively accepted at this time.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.