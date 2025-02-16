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

# Processing constants
MAX_SIZE = 1024
MIN_BLUR_THRESHOLD = 100
ADAPTIVE_THRESH_BLOCK_SIZE = 57
ADAPTIVE_THRESH_C = 7
EMN_DIGIT_SIZE = 28
EMN_BORDER_SIZE = 10

# Utility Functions
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

# Image Preprocessing Functions
def preprocess(image: np.ndarray) -> np.ndarray:
    if image is None:
        logger.error("Invalid image")
        return None
    try:
        h, w = image.shape[:2]
        logger.debug(f"Image dimensions: {w}x{h}")

        if max(h, w) > MAX_SIZE or min(h, w) < MAX_SIZE:
            ratio = MAX_SIZE / float(max(h, w))
            new_size = (int(w * ratio), int(h * ratio))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image to {new_size[0]}x{new_size[1]}")

        fm = check_blurriness(image)
        logger.debug(f"Focus measure: {fm:.2f}")
        if fm < MIN_BLUR_THRESHOLD:
            logger.warning(f"{'Very blurry' if fm < 10 else 'Blurry'} image (FM={fm:.2f})")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPTIVE_THRESH_BLOCK_SIZE,
            ADAPTIVE_THRESH_C,
        )
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return None

def warp_image(thresh_image: np.ndarray) -> np.ndarray:
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
        logger.error(f"Warp error: {e}")
        return None

def get_img(file_path: str) -> np.ndarray:
    if not file_path or not os.path.exists(file_path):
        logger.error(f"Invalid file path: {file_path}")
        return None
    image = cv2.imread(file_path)
    if image is None:
        logger.error(f"Failed to load image: {file_path}")
        return None
    return image

def standardize(img) -> np.ndarray:
    processed = preprocess(img)
    if processed is None:
        logger.error("Image processing failed")
        return None
    warped = warp_image(processed)
    if warped is None:
        logger.error("Warp failed")
        return processed
    return warped

def get_next_cnt(h, i):
    return int(h[0][i][0])

def get_first_child(h, i):
    return int(h[0][i][2])

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
    avg_w = sum(c["rect"][2] for c in cells) / len(cells)
    avg_h = sum(c["rect"][3] for c in cells) / len(cells)
    rows = math.floor(grid_rect[3] / avg_h)
    columns = math.floor(n_cells / rows)
    return (rows, columns)

def extract_digits(img):
    logger = logging.getLogger("TreasureMaze")
    img_contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    grid, grid_idx = get_largest_child(hierarchy, img_contours, -1)
    grid_rect = cv2.boundingRect(grid)

    if logger.isEnabledFor(logging.DEBUG):
        newImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(newImage, img_contours, grid_idx, (0, 255, 0), 3)
        display_image(newImage)

    cells = []
    cell_scan = get_first_child(hierarchy, grid_idx)
    while cell_scan != -1:
        if cv2.contourArea(img_contours[cell_scan]) > 1000:
            cell_rect = cv2.boundingRect(img_contours[cell_scan])
            cell_center = (math.floor(cell_rect[0] + cell_rect[2] / 2), math.floor(cell_rect[1] + cell_rect[3] / 2))
            cells.append({"idx": cell_scan, "contour": img_contours[cell_scan], "rect": cell_rect, "center": cell_center})
        cell_scan = get_next_cnt(hierarchy, cell_scan)

    n_cells = len(cells)
    grid_rows, grid_columns = estimate_grid_size(grid_rect, cells, n_cells)
    logger.info(f"Estimated size: {grid_rows}x{grid_columns} | Found {n_cells} cells ({'match' if grid_rows * grid_columns == n_cells else 'mismatch'})")

    digits = [0 for _ in range(n_cells)]
    for cell in cells:
        digit_cnt, digit_idx = get_largest_child(hierarchy, img_contours, cell["idx"])
        digit_row = math.floor((cell["center"][1] - grid_rect[1]) / grid_rect[3] * grid_rows)
        digit_column = math.floor((cell["center"][0] - grid_rect[0]) / grid_rect[2] * grid_columns)
        digits[digit_row * grid_columns + digit_column] = {"idx": digit_idx, "contour": digit_cnt, "rect": cv2.boundingRect(digit_cnt)}

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

    if logger.isEnabledFor(logging.DEBUG):
        for i, digit_img in enumerate(digits_img):
            display_image(digit_img, f"Digit {i+1}")

    return {"grid": {"rows": grid_rows, "columns": grid_columns}, "digits": digits_img}

label_map = {0: "1", 1: "2", 2: "3", 3: "4", 4: "S", 5: "T", 6: "X"}

def predict_digit(digits_img: list, model_path: str) -> list:
    logger = logging.getLogger("TreasureMaze")
    try:
        model = models.load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise

    predictions = []
    for idx, digit_img in enumerate(digits_img):
        try:
            if len(digit_img.shape) == 3:
                digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
            digit_img = cv2.resize(digit_img, (EMN_DIGIT_SIZE, EMN_DIGIT_SIZE), interpolation=cv2.INTER_AREA)
            digit_img = digit_img.astype("float32") / 255.0
            digit_img = digit_img.reshape(EMN_DIGIT_SIZE, EMN_DIGIT_SIZE, 1)
            prediction = model.predict(np.expand_dims(digit_img, axis=0))
            pred_class = int(np.argmax(prediction, axis=1)[0])
            predicted_label = label_map.get(pred_class, "Unknown")
            predicted_confidence = prediction[0][pred_class]
            predictions.append((predicted_label, predicted_confidence))
            logger.debug(f"Digit {idx}: Predicted {predicted_label} with confidence {predicted_confidence:.2f}")
        except Exception as e:
            logger.error(f"Prediction failed for digit at index {idx}: {e}")
            predictions.append(None)

    return predictions

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
        self.rows = len(maze)
        self.columns = len(maze[0])
        initial_maze = tuple(tuple(row) for row in maze)

        num_s = 0
        for r, row in enumerate(maze):
            if "S" in row:
                initial_position = (row.index("S"), r)
                num_s += 1
                break

        if num_s == 0:
            logger.error(f"Start position not set!")
            raise ValueError

        self.initial = MazeState(initial_maze, initial_position, ())
        treasures = self.count_treasures(maze)

        if n_treasures:
            if n_treasures <= treasures:
                self.n_treasures = n_treasures
            else:
                logger.error(f"Not enough treasures in maze! Insert a lower number.")
                raise ValueError
        else:
            self.n_treasures = treasures

    def count_treasures(self, maze):
        return sum(row.count("T") for row in maze)

    def actions(self, state):
        possible_actions = ['UP', 'DOWN', 'LEFT', 'RIGHT']
        agent_x, agent_y = state.position

        if agent_x == 0:
            possible_actions.remove('LEFT')
        if agent_y == 0:
            possible_actions.remove('UP')
        if agent_x == self.columns - 1:
            possible_actions.remove('RIGHT')
        if agent_y == self.rows - 1:
            possible_actions.remove('DOWN')

        return possible_actions

    def result(self, state, action):
        new_maze = list(list(row) for row in state.maze)
        new_position = list(state.position)
        new_treasures = list(state.treasures)

        delta = {'UP': (0, -1), 'DOWN': (0, 1), 'LEFT': (-1, 0), 'RIGHT': (1, 0)}

        new_position[0] += delta[action][0]
        new_position[1] += delta[action][1]

        current_cell = new_maze[new_position[1]][new_position[0]]
        if current_cell == "X":
            new_maze[new_position[1]][new_position[0]] = 1
        elif current_cell == "T":
            new_maze[new_position[1]][new_position[0]] = 1
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
        new_cell = state1.maze[new_y][new_x]

        if new_cell == "X":
            cell_cost = 5
        elif new_cell == "T" or new_cell == "S":
            cell_cost = 1
        else:
            cell_cost = int(new_cell)

        return c + cell_cost

    def mean_distance(self, node):
        maze = node.state.maze
        agent_position = node.state.position
        distances = [euclidean_distance(agent_position, (c, r)) for r, row in enumerate(maze) for c, column in enumerate(row) if row[c] == "T"]
        return sum(distances) / len(distances) if distances else 0

    def approx_distance(self, node):
        maze = node.state.maze
        agent_position = node.state.position
        distances = [(agent_position, 0)]

        for r, row in enumerate(maze):
            for c, column in enumerate(row):
                if row[c] == "T":
                    treasure_position = (c, r)
                    treasure_distance = manhattan_distance(agent_position, treasure_position)
                    distances.append((treasure_position, treasure_distance))

        if len(distances) > 1:
            distances.sort(key=lambda point: point[1])
            approx_dist = 0
            treasures_left = self.n_treasures - len(node.state.treasures)
            for i in range(1, treasures_left + 1):
                approx_dist += manhattan_distance(distances[i][0], distances[i - 1][0])
            return approx_dist
        else:
            return 0

def pathfind(algorithm: str, rows: int, columns: int, predicted_digits: list) -> None:
    maze = []
    idx = 0
    for r in range(rows):
        row_cells = []
        for c in range(columns):
            if idx < len(predicted_digits):
                pred = predicted_digits[idx]
                cell = pred[0] if pred and pred[0] != "Unknown" else "?"
                row_cells.append(cell)
                idx += 1
            else:
                row_cells.append("?")
        maze.append(row_cells)

    algo_map = {
        "BFS": lambda problem: breadth_first_graph_search(problem),
        "DFS": lambda problem: depth_first_graph_search(problem),
        "UCS": lambda problem: uniform_cost_search(problem),
        "Best First Graph": lambda problem: best_first_graph_search(problem, problem.approx_distance),
        "A*": lambda problem: astar_search(problem, problem.approx_distance)
    }
    if algorithm not in algo_map:
        logger.info(f"Algorithm {algorithm} not recognized")
        return

    try:
        problem = TreasureMaze(maze)
    except Exception as e:
        logger.info(f"Error creating TreasureMaze: {e}")
        return

    search_algo = algo_map[algorithm]
    logger.info(f"\nExecuting {algorithm} pathfinding algorithm...")
    start_time = time()
    solution_node = search_algo(problem)
    end_time = time()

    if solution_node is None:
        logger.info("No solution found!")
    else:
        logger.info(f"Path: {solution_node.solution()}")
        logger.info(f"Path cost: {solution_node.path_cost}")
        logger.info(f"Path length: {solution_node.depth}")
    logger.info(f"Execution Time: {end_time - start_time:.4f}s")
    
    return {"solution": solution_node.solution(), "cost": solution_node.path_cost, "length": solution_node.depth}

def main():
    parser = argparse.ArgumentParser(description="Treasure Maze Image Processor")
    parser.add_argument("-d", "--debug", action="store_true", help="Show debug info")
    parser.add_argument("-f", "--file", required=True, help="Path to image file")
    parser.add_argument("-m", "--model", required=True, help="Path to trained model")
    parser.add_argument(
        "-a", "--algorithm", 
        required=True, 
        choices=["BFS", "DFS", "UCS", "Best First Graph", "A*"], 
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

    img = get_img(args.file)
    img_standard = standardize(img)

    if img_standard is None:
        logger.error("Failed to standardize image")
        return 1

    logger.info("Image standardized, attempting to extract digits...")
    extracted = extract_digits(img_standard)
    predicted_digits = predict_digit(extracted["digits"], args.model)
    grid_info = extracted.get("grid", {})
    rows = grid_info.get("rows", 0)
    columns = grid_info.get("columns", 0)
    logger.info(f"Grid size: {rows}x{columns}")
    
    grid_output = ""
    for r in range(rows):
        row_predictions = []
        for c in range(columns):
            idx = r * columns + c
            if idx < len(predicted_digits):
                pred = predicted_digits[idx]
                if not pred or pred[0] == "Unknown":
                    cell_output = "?"
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
    pathfinding = pathfind(args.algorithm, rows, columns, predicted_digits)

    # let's draw a clean version of the grid as a separate image
    grid_img = np.zeros((EMN_DIGIT_SIZE * rows, EMN_DIGIT_SIZE * columns, 3), np.uint8)
    for r in range(rows):
        for c in range(columns):
            idx = r * columns + c
            if idx < len(predicted_digits):
                pred = predicted_digits[idx]
                if pred and pred[0] != "Unknown":
                    digit_img = extracted["digits"][idx]
                    digit_img = cv2.resize(digit_img, (EMN_DIGIT_SIZE, EMN_DIGIT_SIZE))
                    if len(digit_img.shape) == 2:
                        digit_img = cv2.cvtColor(digit_img, cv2.COLOR_GRAY2BGR)
                    digit_img = cv2.flip(digit_img, 1)
                    digit_img = cv2.rotate(digit_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    digit_img = cv2.bitwise_not(digit_img)
                    
                    grid_img[r * EMN_DIGIT_SIZE:(r + 1) * EMN_DIGIT_SIZE, c * EMN_DIGIT_SIZE:(c + 1) * EMN_DIGIT_SIZE] = digit_img  
                    
    display_image(grid_img, "Predicted Grid")
    
    # get the position of the start point and draw a bounding box around it (S is the start point)
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
        
    if start_pos:
        start_x, start_y = start_pos
        start_x *= EMN_DIGIT_SIZE
        start_y *= EMN_DIGIT_SIZE
        cv2.rectangle(grid_img, (start_x, start_y), (start_x + EMN_DIGIT_SIZE, start_y + EMN_DIGIT_SIZE), (0, 255, 0), 2)
        display_image(grid_img, "Predicted Grid with Start Point")
        
    # based on the instructions from the pathfinding, draw the path on the grid
    path = pathfinding.solution()
    if path:
        for action in path:
            if action == "UP":
                start_y -= EMN_DIGIT_SIZE
            elif action == "DOWN":
                start_y += EMN_DIGIT_SIZE
            elif action == "LEFT":
                start_x -= EMN_DIGIT_SIZE
            elif action == "RIGHT":
                start_x += EMN_DIGIT_SIZE
            cv2.rectangle(grid_img, (start_x, start_y), (start_x + EMN_DIGIT_SIZE, start_y + EMN_DIGIT_SIZE), (0, 0, 255), 2)
            display_image(grid_img, "Predicted Grid with Path")
        
        
    
    
    
    

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
