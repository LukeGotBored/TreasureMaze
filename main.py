import argparse
import logging
import os
import math
import cv2
import numpy as np
from colorama import Fore, init
from time import time
from collections.abc import Callable
from aima import Problem, Node, manhattan_distance, euclidean_distance
from aima import breadth_first_graph_search, depth_first_graph_search, uniform_cost_search, best_first_graph_search, astar_search
import tensorflow as tf
from keras import models
from logger import setup_logger

logger = logging.getLogger("TreasureMaze")

# region Processing constants
MAX_SIZE = 1024
MIN_BLUR_THRESHOLD = 100
ADAPTIVE_THRESH_BLOCK_SIZE = 57
ADAPTIVE_THRESH_C = 7
EMN_DIGIT_SIZE = 28
EMN_BORDER_SIZE = 10

# region Utility methods
def display_image(image: np.ndarray, title: str = "Image") -> None:
    cv2.imshow(title, image)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        exit(0)
    else:
        cv2.destroyAllWindows()

def check_blurriness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def get_img(file_path: str) -> np.ndarray:
    if not file_path or not os.path.exists(file_path):
        logger.error(f"Invalid file path: {file_path}")
        return None
    image = cv2.imread(file_path)
    if image is None:
        logger.error(f"Failed to load image: {file_path}")
        return None
    return image


# region Image Loading & Preprocessing
# Load the image from disk and apply preprocessing steps to standardize it for digit extraction
def preprocess(image: np.ndarray) -> np.ndarray:
    if image is None:
        logger.error("Invalid image")
        return None
    try:
        h, w = image.shape[:2]
        logger.debug(f"Image dimensions: {w}x{h}")

        if max(h, w) > MAX_SIZE or min(h, w) < MAX_SIZE:
            try:
                ratio = MAX_SIZE / float(max(h, w))
                new_size = (int(w * ratio), int(h * ratio))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
                logger.debug(f"Resized image to {new_size[0]}x{new_size[1]}")
            except Exception as e:
                logger.error(f"Could not extract the grid: {e}")
                return None

        fm = check_blurriness(image)
        logger.debug(f"Focus measure: {fm:.2f}")
        if fm < MIN_BLUR_THRESHOLD:
            logger.warning(f"{'Very blurry' if fm < 10 else 'Blurry'} image (FM={fm:.2f})")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.GaussianBlur(gray, (9, 9), 0)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPTIVE_THRESH_BLOCK_SIZE,
            ADAPTIVE_THRESH_C,
        )
    except Exception as e:
        logger.error(f"Unable to preprocess image: {e}")
        return None

def warp_image(thresh_image: np.ndarray) -> np.ndarray:
    # ? Source: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    try:
        contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logger.error("No contours found")
            return None
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.1 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        if len(approx) != 4:
            logger.error("Could not find 4 corners")
            return None

        src_pts = approx.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = src_pts[np.argmin(src_pts.sum(axis=1))]  # top-left
        rect[2] = src_pts[np.argmax(src_pts.sum(axis=1))]  # bottom-right
        rect[1] = src_pts[np.argmin(np.diff(src_pts, axis=1))]  # top-right
        rect[3] = src_pts[np.argmax(np.diff(src_pts, axis=1))]  # bottom-left

        padding = 25
        w = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])) + 2 * padding)
        h = int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[3] - rect[0])) + 2 * padding)

        dst_pts = np.array(
            [
                [padding, padding],
                [w - padding, padding],
                [w - padding, h - padding],
                [padding, h - padding],
            ],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(rect, dst_pts)
        return cv2.warpPerspective(thresh_image, M, (w, h))
    except Exception as e:
        logger.error(f"Unable to warp image: {e}")
        return None

def standardize(img) -> np.ndarray:
    """
    Standardize the input image by resizing, blurring, and applying adaptive thresholding.
    """
    if img is None:
        logger.error("Invalid image")
        return None
    
    processed = preprocess(img)
    if processed is None:
        logger.error("Image processing failed")
        return None
    warped = warp_image(processed)
    if warped is None:
        logger.error("Warp failed")
        return processed
    return warped


# region Digit Extraction
# Extract digits from the standardized image and prepare them for prediction.
def safe_boundingRect(contour):
    # tldr; this function handles the case where the height of the bounding rectangle is zero
    try:
        rect = cv2.boundingRect(contour)
        # Guard against division by zero if height is 0.
        if rect[3] == 0:
            raise ValueError("boundingRect height is zero")
        return rect
    except Exception as e:
        logger.error(f"Error in boundingRect: {e}")
        return (0, 0, 0, 0)

def get_next_cnt(h, i):
    try:
        return int(h[0][i][0])
    except Exception as e:
        logger.error(f"Error in get_next_cnt for index {i}: {e}")
        return -1

def get_first_child(h, i):
    try:
        return int(h[0][i][2])
    except Exception as e:
        logger.error(f"Error in get_first_child for index {i}: {e}")
        return -1

def get_largest_child(h, contours, i=-1):
    local_scan = get_first_child(h, i) if i != -1 else 0
    largest_cnt = []
    largest_cnt_idx = 0
    max_area = 0

    while local_scan != -1:
        child_cnt = contours[local_scan]
        area = cv2.contourArea(child_cnt)
        if area > max_area:
            max_area = area
            largest_cnt = child_cnt
            largest_cnt_idx = local_scan
        local_scan = get_next_cnt(h, local_scan)

    return largest_cnt, largest_cnt_idx

def estimate_grid_size(grid_rect, cells, n_cells):
    if len(cells) == 0:
        return (0, 0)
    avg_w = sum(c["rect"][2] for c in cells) / len(cells)
    avg_h = sum(c["rect"][3] for c in cells) / len(cells)
    if avg_h == 0:
        return (0, 0)
    rows = math.floor(grid_rect[3] / avg_h)
    columns = math.floor(n_cells / rows) if rows > 0 else 0
    return (rows, columns)

def extract_digits(img):
    logger = logging.getLogger("TreasureMaze")
    try:
        img_contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(hierarchy) == 0:
            raise ValueError("Hierarchy not found")
        grid, grid_idx = get_largest_child(hierarchy, img_contours, -1)
        if grid is None or ((isinstance(grid, np.ndarray) and grid.size == 0)) or grid_idx < 0:
            raise ValueError("Grid extraction failed")
        grid_rect = safe_boundingRect(grid)

        if logger.isEnabledFor(logging.DEBUG):
            newImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(newImage, img_contours, grid_idx, (0, 255, 0), 3)
            display_image(newImage)

        cells = []
        cell_scan = get_first_child(hierarchy, grid_idx)
        while cell_scan != -1:
            if cv2.contourArea(img_contours[cell_scan]) > 1000:
                try:
                    cell_rect = safe_boundingRect(img_contours[cell_scan])
                    cell_center = (math.floor(cell_rect[0] + cell_rect[2] / 2), math.floor(cell_rect[1] + cell_rect[3] / 2))
                    cells.append({"idx": cell_scan, "contour": img_contours[cell_scan], "rect": cell_rect, "center": cell_center})
                except Exception as e:
                    logger.error(f"Skipping cell at index {cell_scan}: {e}")
            cell_scan = get_next_cnt(hierarchy, cell_scan)

        n_cells = len(cells)
        grid_rows, grid_columns = estimate_grid_size(grid_rect, cells, n_cells)
        logger.info(f"Estimated size: {grid_rows}x{grid_columns} | Found {n_cells} cells")
        if grid_rows == 0 or grid_columns == 0:
            raise ValueError("Invalid grid dimensions extracted")

        digits = [0 for _ in range(n_cells)]
        for cell in cells:
            digit_cnt, digit_idx = get_largest_child(hierarchy, img_contours, cell["idx"])
            try:
                digit_rect = safe_boundingRect(digit_cnt)
            except Exception as e:
                logger.error(f"Error retrieving digit rectangle: {e}")
                continue
            digit_row = math.floor((cell["center"][1] - grid_rect[1]) / grid_rect[3] * grid_rows)
            digit_column = math.floor((cell["center"][0] - grid_rect[0]) / grid_rect[2] * grid_columns)
            idx = digit_row * grid_columns + digit_column
            if idx < 0 or idx >= len(digits):
                logger.error(f"Computed cell index {idx} out of range for grid size {grid_rows}x{grid_columns}")
            else:
                digits[idx] = {"idx": digit_idx, "contour": digit_cnt, "rect": digit_rect}

        if any(isinstance(item, int) for item in digits):
            raise ValueError("Grid extraction failed: some cells could not be processed.")

        logger.info(f"Found digits: {len(digits)}")
        newImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        def pad_to_square(image: np.ndarray, target_size: int) -> np.ndarray:
            h, w = image.shape[:2]
            if w != h:
                border = abs(w - h) // 2
                if w > h:
                    image = cv2.copyMakeBorder(image, border, border, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                else:
                    image = cv2.copyMakeBorder(image, 0, 0, border, border, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)

        digits_img = []
        for digit in digits:
            x, y, w, h = digit["rect"]
            digit_img = newImage[y:y + h, x:x + w]
            digit_img = pad_to_square(digit_img, EMN_DIGIT_SIZE)
            digit_img = cv2.copyMakeBorder(digit_img, EMN_BORDER_SIZE, EMN_BORDER_SIZE, EMN_BORDER_SIZE, EMN_BORDER_SIZE, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            digit_img = cv2.flip(digit_img, 1)
            if np.mean(digit_img) > 127:
                digit_img = cv2.bitwise_not(digit_img)
            digit_img = cv2.rotate(digit_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            digits_img.append(digit_img)

        if not digits_img:
            raise ValueError("No grid detected")
        return {"grid": {"rows": grid_rows, "columns": grid_columns}, "digits": digits_img}
    except Exception as e:
        logger.error(f"Could not extract the grid: {e}")
        raise

# region Digit Prediction
# Predict the values of the extracted digits using the trained model

label_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "S", 5: "T", 6: "X"}
def predict_digit(digits_img: list, model_path: str) -> list:
    logger = logging.getLogger("TreasureMaze")
    try:
        model = models.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise

    if not digits_img:
        logger.warning("No digits to predict.")
        return []

    batch_data = []
    for digit_img in digits_img:
        try:
            if len(digit_img.shape) == 3:
                digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
            digit_img = cv2.resize(digit_img, (EMN_DIGIT_SIZE, EMN_DIGIT_SIZE), interpolation=cv2.INTER_AREA)
            digit_img = digit_img.astype("float32") / 255.0
            digit_img = digit_img.reshape(EMN_DIGIT_SIZE, EMN_DIGIT_SIZE, 1)
            batch_data.append(digit_img)
        except Exception as e:
            logger.error(f"Preprocessing failed for a digit: {e}")
            batch_data.append(np.zeros((EMN_DIGIT_SIZE, EMN_DIGIT_SIZE, 1), dtype="float32"))

    predictions = []
    try:
        predictions_array = model.predict(np.array(batch_data))
        for idx, prediction in enumerate(predictions_array):
            pred_class = int(np.argmax(prediction))
            predicted_label = label_map.get(pred_class, "Unknown")
            predicted_confidence = prediction[pred_class]
            predictions.append((predicted_label, predicted_confidence))
            logger.debug(f"Digit {idx}: Predicted {predicted_label} with confidence {predicted_confidence:.2f}")
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        return [None for _ in digits_img]

    return predictions


# region Pathfinding
# Define the problem and pathfinding methods for the treasure maze
class MazeState:
    def __init__(self, maze, position, treasures):
        self.maze = maze
        self.position = position
        self.treasures = treasures

    def __lt__(self, state):
        return True

    def __eq__(self, state):
        return self.position == state.position and self.treasures == state.treasures

    def __hash__(self):
        return hash((self.position, self.treasures))

class TreasureMaze(Problem):
    def __init__(self, maze, n_treasures=None):
        # Validate maze dimensions and consistency
        if not maze or not all(maze) or any(len(row) != len(maze[0]) for row in maze):
            logger.error("Maze is empty or has inconsistent row lengths.")
            raise ValueError("Invalid maze structure.")
        self.rows = len(maze)
        self.columns = len(maze[0])

        # Convert to immutable type
        initial_maze = tuple(tuple(str(cell) for cell in row) for row in maze)

        # Find the 'S' start position
        start_found = False
        initial_position = (0, 0)
        for r, row in enumerate(maze):
            if "S" in row:
                initial_position = (row.index("S"), r)
                start_found = True
                break
        if not start_found:
            logger.error("Start position 'S' not found!")
            raise ValueError("No start in maze.")

        self.initial = MazeState(initial_maze, initial_position, ())

        # Count total treasures and validate requested treasure count
        treasures = self.count_treasures(maze)
        if n_treasures:
            if n_treasures <= 0:
                logger.error("Requested treasure count must be positive.")
                raise ValueError("Invalid treasure count.")
            if n_treasures > treasures:
                logger.error("Not enough treasures in maze for requested count!")
                raise ValueError("Too few treasures in maze.")
            self.n_treasures = n_treasures
        else:
            self.n_treasures = treasures

    def count_treasures(self, maze):
        return sum(row.count("T") for row in maze)

    def actions(self, state):
        possible_actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        agent_x, agent_y = state.position

        if agent_x <= 0:
            possible_actions.remove("LEFT")
        if agent_y <= 0:
            possible_actions.remove("UP")
        if agent_x >= self.columns - 1:
            possible_actions.remove("RIGHT")
        if agent_y >= self.rows - 1:
            possible_actions.remove("DOWN")
        return possible_actions

    def result(self, state, action):
        new_maze = [list(row) for row in state.maze]
        new_position = list(state.position)
        new_treasures = list(state.treasures)

        delta = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
        new_position[0] += delta[action][0]
        new_position[1] += delta[action][1]

        current_cell = str(new_maze[new_position[1]][new_position[0]])
        if current_cell == "X":
            new_maze[new_position[1]][new_position[0]] = "1"
        elif current_cell == "T":
            new_maze[new_position[1]][new_position[0]] = "1"
            new_treasures.append(tuple(new_position))

        return MazeState(
            tuple(tuple(row) for row in new_maze),
            tuple(new_position),
            tuple(new_treasures)
        )

    def goal_test(self, state):
        return len(state.treasures) == self.n_treasures

    def path_cost(self, c, state1, action, state2):
        new_x, new_y = state2.position
        new_cell = str(state1.maze[new_y][new_x])

        if new_cell == "X":
            cell_cost = 5
        elif new_cell in ["T", "S"]:
            cell_cost = 1
        else:
            try:
                cell_cost = int(new_cell)
            except ValueError:
                cell_cost = 1  # Fallback if cell is non-numeric
        return c + cell_cost

    def mean_distance(self, node):
        maze = node.state.maze
        agent_position = node.state.position
        distances = [
            euclidean_distance(agent_position, (x, y))
            for y, row in enumerate(maze)
            for x, cell in enumerate(row)
            if cell == "T"
        ]
        return sum(distances) / len(distances) if distances else 0

    def approx_distance(self, node):
        maze = node.state.maze
        agent_position = node.state.position
        treasure_positions = []
        treasures_collected = len(node.state.treasures)
        total_distance = 0

        if treasures_collected < self.n_treasures:
            global position_scan
            position_scan = agent_position

            for r, row in enumerate(maze):
                for c, column in enumerate(row):
                    if row[c] == "T":
                        treasure_positions.append((c, r))
            
            for i in range(0, self.n_treasures - treasures_collected):
                closest = min(treasure_positions, key = lambda pos: manhattan_distance(position_scan, pos))
                total_distance += manhattan_distance(position_scan, closest)
                position_scan = closest
                treasure_positions.remove(closest)

                if len(treasure_positions) == 0:
                    break

        return total_distance
        
def pathfind(algorithm: str, rows: int, columns: int, predicted_digits: list, treasures: int = None):
    if rows <= 0 or columns <= 0:
        logger.error("Invalid grid dimensions")
        return None
    
    if not predicted_digits:
        logger.error("No digits predicted, cannot build maze.")
        return None

    # Rebuild the maze from predictions
    maze = []
    idx = 0

    # If no treasure count is provided, assume from predicted 'T'
    if treasures is None:
        treasures = sum(1 for pred in predicted_digits if pred and pred[0] == "T")

    if treasures is not None and treasures <= 0:
        logger.warning(f"Invalid treasure count: {treasures}. Using default.")
        treasures = None
    
    if len(predicted_digits) < rows * columns:
        logger.warning("Fewer predicted digits than expected grid cells.")

    for r in range(rows):
        row_cells = []
        for c in range(columns):
            if idx < len(predicted_digits):
                pred = predicted_digits[idx]
                cell = pred[0] if pred and pred[0] != "Unknown" else "1"
                row_cells.append(cell)
            else:
                row_cells.append("1")
            idx += 1
        maze.append(row_cells)

    algo_map = {
        "BFS": breadth_first_graph_search,
        "DFS": depth_first_graph_search,
        "UCS": uniform_cost_search,
        "Best First": lambda p: best_first_graph_search(p, p.approx_distance),
        "A*": lambda p: astar_search(p, p.approx_distance)
    }

    if algorithm not in algo_map:
        logger.info(f"Algorithm {algorithm} not recognized")
        return

    try:
        problem = TreasureMaze(maze, n_treasures=treasures)
    except ValueError as ve:
        logger.error(f"Problem creation failed: {ve}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during problem creation: {e}")
        return None

    logger.info(f"Executing {algorithm}...")
    start_time = time()
    solution_node = algo_map[algorithm](problem)
    end_time = time()

    if solution_node is None:
        logger.info("No solution found!")
    else:
        logger.info(f"Path: {solution_node.solution()}")
        logger.info(f"Path cost: {solution_node.path_cost}")
        logger.info(f"Path length: {solution_node.depth}")
    logger.info(f"Execution time: {end_time - start_time:.4f}s")
    return {
        "solution": solution_node.solution() if solution_node else None,
        "cost": solution_node.path_cost if solution_node else None,
        "length": solution_node.depth if solution_node else None
    }

# region Main
def main():
    
    # region Argument Parsing
    parser = argparse.ArgumentParser(description="Treasure Maze Image Processor")
    parser.add_argument("-d", "--debug", action="store_true", help="Show debug info")
    parser.add_argument("-f", "--file", required=True, help="Path to image file")
    parser.add_argument("-m", "--model", required=True, help="Path to trained model")
    parser.add_argument("-t", "--treasures", type=int, help="Number of treasures to find")
    parser.add_argument(
        "-a", "--algorithm", 
        required=True, 
        choices=["BFS", "DFS", "UCS", "Best First", "A*"], 
        help="Choose the type of algorithm for pathfinding"
    )
    args = parser.parse_args()
    logger = setup_logger(args.debug)
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled!")
    else:
        logger.setLevel(logging.INFO)

    logger.info("Treasure Maze | Initializing...")
    
    # region Image Loading & Preprocessing
    img = get_img(args.file)
    img_standard = standardize(img)
    if img_standard is None:
        logger.error("Failed to standardize image")
        return 1

    logger.info("Image standardized, attempting to extract digits...")
    extracted = extract_digits(img_standard)
    grid_info = extracted.get("grid", {})
    rows = grid_info.get("rows", 0)
    columns = grid_info.get("columns", 0)

    # exit if grid extraction failed
    if rows == 0 or columns == 0 or not extracted.get("digits"):
        logger.error("Grid extraction did not complete successfully; aborting the process.")
        return 1

    logger.info(f"Grid size: {rows}x{columns}")
    
    # region Digit Extraction & Prediction
    predicted_digits = predict_digit(extracted["digits"], args.model)
    logger.info(f"Grid size: {rows}x{columns}")

    # region Grid Processing & Output
    grid_output = ""
    for r in range(rows):
        row_predictions = []
        for c in range(columns):
            idx = r * columns + c
            if idx < len(predicted_digits):
                pred = predicted_digits[idx]
                if not pred or pred[0] == "Unknown":
                    cell_output = "1"  # Fallback digit
                else:
                    label_val, conf = pred[0], pred[1]
                    cell_output = f"{label_val} (c: {conf:.2f})"
            else:
                cell_output = "N/A"
            row_predictions.append(cell_output)
        grid_output += "  ".join(row_predictions) + "\n"
    logger.info(f"Grid predictions:\n{grid_output}")
    if logger.isEnabledFor(logging.DEBUG):
        display_image(img_standard)
    logger.info(f"Predicted {len(predicted_digits)} digits ({len(predicted_digits) / (rows * columns) * 100:.2f}% of expected)")
    logger.info(f"Pathfinding algorithm: {args.algorithm}")
    
    # region Pathfinding
    pathfinding = pathfind(args.algorithm, rows, columns, predicted_digits, args.treasures)

    # region Extract Start Position
    start_pos = None
    for r in range(rows):
        for c in range(columns):
            idx = r * columns + c
            if idx < len(predicted_digits):
                pred = predicted_digits[idx]
                if pred and pred[0] == "S":
                    start_pos = (c, r)
                    break
        if start_pos:
            break

    return 0

if __name__ == "__main__":    
    try:
        exit(main())
    except KeyboardInterrupt:
        logger.info("Exiting...")
        exit(0)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        exit(1)