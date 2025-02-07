# TreasureMaze

TreasureMaze is a Python-based project for maze detection and pathfinding optimization.

## About

This project combines computer vision (using OpenCV) for maze extraction and segmentation with advanced pathfinding algorithms to generate optimal solutions. Recent enhancements include:
- **Modularization**: Grid processing and digit extraction functions have been moved to `utils.py`.
- **Code Optimization**: Redundant operations have been minimized to improve performance.
- **Enhanced Logging**: A helper function now formats console messages compactly and pleasantly.
- **Improved CLI**: Command-line arguments allow customization of various parameters.

## Core Features

### Maze Structure
- **Start (S)**: The agent's starting position.
- **Treasure (T)**: Points representing treasures.
- **Wall (X)**: Breakable obstacles with an associated cost (5).
- **Paths (1/4)**: Traversable cells with their respective costs.

## Requirements
- Python 3.x
- OpenCV >= 4.0
- NumPy

## Installation
1. Ensure [Python 3.x](https://www.python.org/downloads/) is installed.
2. Clone the repository.
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the main script with optional command-line arguments:
```bash
python main.py [-d | --debug] [-s | --size SIZE] [--output-size OUTPUT_SIZE] [--output-padding OUTPUT_PADDING] [--img-resize IMG_RESIZE]
```
- `-d/--debug`: Enable debug mode.
- `-s/--size`: Maximum size for the input image (default: 1024).
- `--output-size`: Output digit size (default: 28).
- `--output-padding`: Padding for the output (default: 10).
- `--img-resize`: Override the default image resize value.

After running, provide the maze image path when prompted. Monitor the console for processing details via compact log messages.

## Contributing
Contributions are not currently accepted, but the repository is public and can be used for educational purposes.

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.