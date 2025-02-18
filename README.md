# TreasureMaze
TreasureMaze is a project that implements a maze extraction and analysis pipeline using computer vision, digit recognition, and pathfinding algorithms. The project demonstrates a practical application of image processing, algorithm design, and machine learning in Python.

## Project Overview
This project performs the following tasks:
- **Maze Extraction & Segmentation:** Uses OpenCV to standardize and extract grid structures from maze images.
- **Digit Recognition:** Applies a trained neural network (CNN or MLP) to predict digit values from segmented maze cells.
- **Pathfinding:** Implements breadth-first, depth-first, uniform cost, best-first, and A* search algorithms to navigate the maze.
- **Graphical Visualization:** Provides a GUI to display image analysis, maze grid extraction, and solution path visualization.
- **Model Training:** Includes a training pipeline for digit recognition using filtered EMNIST data.

## Installation

### Requirements
- Python 3.x (any version supported by TensorFlow)
- OpenCV (≥ 4.0)  
- NumPy  
- TensorFlow & Keras  
- PyQt6 (for the GUI)  
- Other dependencies as outlined in `requirements.txt`

### Setup
1. Clone the repository.
2. Install the required dependencies by running:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Maze Analyzer
To process a maze image and perform pathfinding, run:
```bash
python main.py -f path/to/image.jpg -m path/to/model.keras -a "A*"
```
Additional options:
- `-d/--debug`: Enable debug output.
- `-t/--treasures`: Specify the number of treasures to locate.

### Model Training
To train the digit recognition model:
```bash
python model/train.py [--debug] [--use-batchnorm] [--disable-early-stopping]
```
Follow the prompts to save the trained model.

### GUI Application
To launch the graphical interface:
```bash
python gui.py
```
Use the provided options within the GUI to load images, analyze them, and visualize the solution path.

## Technologies Used
- **OpenCV:** For image preprocessing and grid extraction.
- **TensorFlow & Keras:** For building and training the neural network model.
- **Python Standard Libraries:** For implementing algorithms and application logic.
- **PyQt6:** For developing the responsive graphical user interface.
- **AIMA Search Algorithms:** For demonstrating various pathfinding techniques.

## Credits
- Developed by [Gianluca Suriani](https://github.com/LukeGotBored) and [Andrea Riccardi](https://github.com/andr3wpixel).
- EMNIST dataset provided by [NIST](https://www.nist.gov/itl/iad/image-group/emnist-dataset).

## License
This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for further details.