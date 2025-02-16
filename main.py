import argparse
import logging
import os
import math

import cv2
import numpy as np
from colorama import Fore, init
from logger import setup_logger # type: ignore
import colorsys # imagine importing a whole library just for a single function, lol

init(autoreset=True) #? this is to reset colorama colors after each print

# Processing constants
MAX_SIZE = 1024                     
MIN_BLUR_THRESHOLD = 100            
ADAPTIVE_THRESH_BLOCK_SIZE = 57     
ADAPTIVE_THRESH_C = 7

# ---------------- Utility Functions ----------------

def display_image(image: np.ndarray, title: str = "Image") -> None:
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def check_blurriness(image: np.ndarray) -> float:
    #? Source: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

# ---------------- Image Preprocessing Functions ----------------

def preprocess(image: np.ndarray) -> np.ndarray:
    logger = logging.getLogger('TreasureMaze')
    if image is None:
        logger.error("Invalid image")
        return None
    try:
        h, w = image.shape[:2]
        logger.debug(f"Image dimensions: {w}x{h}")
        
        # Resize image if out of bounds (e.g. too big or too small)
        if max(h, w) > MAX_SIZE or min(h, w) < MAX_SIZE:
            ratio = MAX_SIZE / float(max(h, w))
            new_size = (int(w * ratio), int(h * ratio))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image to {new_size[0]}x{new_size[1]}")
                
        fm = check_blurriness(image)
        color = Fore.RED if fm < MIN_BLUR_THRESHOLD * 0.33 else (
            Fore.YELLOW if fm < MIN_BLUR_THRESHOLD * 0.66 else Fore.GREEN)
        logger.debug(f"Focus measure: {color}{fm:.2f}{Fore.RESET}")
        if fm < MIN_BLUR_THRESHOLD:
            logger.warning(f"{'Very blurry' if fm < 10 else 'Blurry'} image (FM={fm:.2f})")
            
        # Apply image processing steps
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_C
        )
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        return None

def warp_image(thresh_image: np.ndarray) -> np.ndarray:
    # Partially adapted from https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    # ? I've mainly changed the code to use numpy arrays and added some error handling
    logger = logging.getLogger('TreasureMaze')
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
        rect[0] = src_pts[np.argmin(src_pts.sum(axis=1))]      # top-left
        rect[2] = src_pts[np.argmax(src_pts.sum(axis=1))]      # bottom-right
        rect[1] = src_pts[np.argmin(np.diff(src_pts, axis=1))]   # top-right
        rect[3] = src_pts[np.argmax(np.diff(src_pts, axis=1))]   # bottom-left
        
        padding = 25
        w = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))) + 2 * padding
        h = int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[3] - rect[0]))) + 2 * padding
        
        dst_pts = np.array([
            [padding, padding],
            [w - padding, padding],
            [w - padding, h - padding],
            [padding, h - padding],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst_pts)
        return cv2.warpPerspective(thresh_image, M, (w, h))
    except Exception as e:
        logger.error(f"Warp error: {e}")
        return None

def get_img(file_path: str) -> np.ndarray:
    logger = logging.getLogger('TreasureMaze')
    if not file_path or not os.path.exists(file_path):
        logger.error(f"Invalid file path: {file_path}")
        return None
    image = cv2.imread(file_path)
    if image is None:
        logger.error(f"Failed to load image: {file_path}")
        return None
    return image

def standardize(img) -> np.ndarray:
    logger = logging.getLogger('TreasureMaze')
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
    global child_scan, child_cnt

    if i != -1:
        child_scan = get_first_child(h, i)
    else:
        child_scan = 0

    global largest_cnt
    global largest_cnt_idx
    global max_area
    largest_cnt = []
    largest_cnt_idx = 0
    max_area = 0

    while child_scan != -1:
        child_cnt = contours[child_scan]
        if cv2.contourArea(child_cnt) > max_area:
            max_area = cv2.contourArea(child_cnt)
            largest_cnt = child_cnt
            largest_cnt_idx = child_scan
        child_scan = get_next_cnt(h, child_scan) 
    
    return largest_cnt, largest_cnt_idx

def estimate_grid_size(grid_rect, cells, n_cells):
   avg_w = 0
   avg_h = 0

   for c in cells:
      avg_w += c["rect"][2]
      avg_h += c["rect"][3]

   avg_w /= len(cells)
   avg_h /= len(cells)

   rows = math.floor(grid_rect[3] / avg_h)
   columns = math.floor(n_cells / rows)

   return (rows, columns)

def extract_digits(img):
    logger = logging.getLogger('TreasureMaze')

    img_contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    grid, grid_idx = get_largest_child(hierarchy, img_contours, -1)
    grid_rect = cv2.boundingRect(grid)

    newImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(newImage, img_contours, grid_idx, (0, 255, 0), 3)
    #cv2.circle(newImage, cell["center"], 3, (0, 0, 255), 3)
    display_image(newImage)

    cells = []

    global cell_scan
    cell_scan = get_first_child(hierarchy, grid_idx)
    while cell_scan != -1:

        #print("Area: ", cv2.contourArea(img_contours[cell_scan]))

        if cv2.contourArea(img_contours[cell_scan]) > 1000:
    
            new_cell = {}
            
            new_cell["idx"] = cell_scan
            new_cell["contour"] = img_contours[cell_scan]
            cell_rect = cv2.boundingRect(img_contours[cell_scan])

            cell_center = (math.floor(cell_rect[0] + cell_rect[2]/2), math.floor(cell_rect[1] + cell_rect[3]/2))

            new_cell["rect"] = cell_rect
            new_cell["center"] = cell_center

            cells.append(new_cell)
            
        cell_scan = get_next_cnt(hierarchy, cell_scan)

    n_cells = len(cells)
    logger.info(f"Estimated cells: {n_cells}")

    grid_rows, grid_columns = estimate_grid_size(grid_rect, cells, n_cells)
    logger.info(f"Estimated size: {grid_rows} rows - {grid_columns} columns")

    digits = [0 for i in range(0, n_cells)]

    for cell in cells:
    
        new_digit = {}

        digit_cnt, digit_idx = get_largest_child(hierarchy, img_contours, cell["idx"])

        new_digit["idx"] = digit_idx
        new_digit["contour"] = digit_cnt
        new_digit["rect"] = cv2.boundingRect(digit_cnt)

        digit_row = math.floor((cell["center"][1] - grid_rect[1]) / grid_rect[3] * grid_rows) 
        digit_column = math.floor((cell["center"][0] - grid_rect[0]) / grid_rect[2] * grid_columns)

        digits[digit_row * grid_columns + digit_column] = new_digit

    logger.info(f"Found digits: {len(digits)}")

    for digit in digits:
        newImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(newImage, img_contours, digit["idx"], (0, 255, 0), 3)
        #cv2.circle(newImage, cell["center"], 3, (0, 0, 255), 3)
        display_image(newImage)

# ---------------- Main ----------------

label_map = {
    0: "1",
    1: "2",
    2: "3",
    3: "4",
    4: "S",
    5: "T",
    6: "X"
}

def main():
    parser = argparse.ArgumentParser(description="Treasure Maze Image Processor")
    parser.add_argument("-d", "--debug", action="store_true", help="Show debug info")
    parser.add_argument("-f", "--file", required=True, help="Path to image file")
    #parser.add_argument("-m", "--model", required=True, help="Path to trained model")
    args = parser.parse_args()

    logger = setup_logger()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Debug {Fore.GREEN}enabled{Fore.RESET}!")
    
    img = get_img(args.file)
    img_standard = standardize(img)

    if img_standard is None:
        return 1
    
    display_image(img_standard)
    extract_digits(img_standard)
    return 0
        
if __name__ == "__main__":
    exit(main())