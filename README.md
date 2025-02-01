# TreasureMaze
A Python-based project for maze detection and pathfinding optimization.

## About
TreasureMaze implements an AI-driven solution for maze navigation and treasure collection problems. 
The project combines computer vision for maze detection with advanced pathfinding algorithms for optimal solution generation.

## Core Features

### Maze Structure
- **Start (S)**: Initial agent position
- **Treasure (T)**: Collection points
- **Wall (X)**: Destroyable obstacles (cost: 5)
- **Paths (1/4)**: Traversable cells with associated costs


## Requirements
- Python 3.x
- OpenCV >= 4.0
- NumPy

## Installation
1. Ensure [Python 3.x](https://www.python.org/downloads/) is installed
2. Clone the repository
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Execute the main script:
   ```bash
   python main.py
   ```
2. Input the maze image path
3. Monitor console output for process status

## Contributing
At the moment, the project is not open to external contributions.
However, this repository is public and can be used for educational purposes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.