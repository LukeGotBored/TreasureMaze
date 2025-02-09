import argparse
import logging
import os
import math

import cv2
import numpy as np
from colorama import Fore, init
from logger import setup_logger

init(autoreset=True)

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
    # Source: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def contains(r1: tuple, r2: tuple) -> bool:
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    return x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2)

def extract_digit(image: np.ndarray, rect: tuple) -> np.ndarray:
    x, y, w, h = rect
    d_img = image[y:y+h, x:x+w]
    d_img = cv2.copyMakeBorder(d_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
    d_img = cv2.resize(d_img, (28, 28), interpolation=cv2.INTER_AREA)
    if len(d_img.shape) == 3:
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2GRAY)
    return cv2.bitwise_not(d_img)

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
            logger.error("Contour does not have 4 vertices")
            return None
        src_pts = approx.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        rect[0] = src_pts[np.argmin(src_pts.sum(axis=1))]      # top-left
        rect[2] = src_pts[np.argmax(src_pts.sum(axis=1))]      # bottom-right
        rect[1] = src_pts[np.argmin(np.diff(src_pts, axis=1))]   # top-right
        rect[3] = src_pts[np.argmax(np.diff(src_pts, axis=1))]   # bottom-left
        
        padding = 15
        w = int(max(np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3]))) + 2 * padding
        h = int(max(np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[3] - rect[0]))) + 2 * padding
        
        dst_pts = np.array([
            [padding, padding],
            [w - padding, padding],
            [w - padding, h - padding],
            [padding, h - padding],
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(rect, dst_pts)
        return cv2.warpPerspective(cv2.cvtColor(thresh_image, cv2.COLOR_GRAY2BGR), M, (w, h))
    except Exception as e:
        logger.error(f"Warp error: {e}")
        return None

def standardize(file_path: str) -> np.ndarray:
    logger = logging.getLogger('TreasureMaze')
    if not file_path or not os.path.exists(file_path):
        logger.error(f"Invalid file path: {file_path}")
        return None
    image = cv2.imread(file_path)
    if image is None:
        logger.error(f"Failed to load image: {file_path}")
        return None
    processed = preprocess(image)
    if processed is None:
        logger.error("Image processing failed")
        return None
    warped = warp_image(processed)
    if warped is None:
        logger.error("Warp failed")
        return None
    return warped

def estimate_grid_size(grid_rect: tuple, cells: list) -> tuple:
    if not cells:
        return (0, 0)
    avg_cell_w = sum(cell[2] for cell in cells) / len(cells)
    avg_cell_h = sum(cell[3] for cell in cells) / len(cells)
    cols = round(grid_rect[2] / avg_cell_w)
    rows = round(grid_rect[3] / avg_cell_h)
    return (rows, cols)

# ---------------- Grid & Digit Extraction ----------------

def process(image: np.ndarray) -> dict:
    logger = logging.getLogger('TreasureMaze')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or hierarchy is None:
        logger.error("No contours or missing hierarchy")
        return {"cells": [], "digits": [], "est_rows": 0, "est_cols": 0}
    hierarchy = hierarchy[0]

    grid_contour = max(contours, key=cv2.contourArea)
    max_area = cv2.contourArea(grid_contour)
    grid_rect = cv2.boundingRect(grid_contour)
    grid_idx = next((i for i, cnt in enumerate(contours)
                    if abs(cv2.contourArea(cnt) - max_area) < 1e-3), None)
    if grid_idx is None:
        logger.error("Grid contour index not found")
        return {"cells": [], "digits": [], "est_rows": 0, "est_cols": 0}

    cells = []
    candidate_digits = []
    debug_img = image.copy()

    for i, cnt in enumerate(contours):
        if i == grid_idx:
            continue
        area = cv2.contourArea(cnt)
        if area < 100 or area > 0.1 * max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        if w < 10 or h < 10:
            continue
        if hierarchy[i][3] == grid_idx:
            cells.append((x, y, w, h))
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        else:
            candidate_digits.append((x, y, w, h))
    
    # Optimize cell filtering using list comprehension
    filtered_cells = [cell for i, cell in enumerate(cells)
                      if not any(i != j and contains(cell, cells[j]) for j in range(len(cells)))]
    
    cells = filtered_cells
    rows, cols = estimate_grid_size(grid_rect, cells)
    expected_cells = rows * cols
    if expected_cells == 0:
        logger.warning("Estimated grid size is 0, falling back to candidate digits")
    elif len(cells) < expected_cells:
        logger.warning(f"Cells detected: {len(cells)}; expected ~{expected_cells}")

    final_digits = [rect for i, rect in enumerate(candidate_digits)
                    if not any(i != j and contains(rect, candidate_digits[j])
                               for j in range(len(candidate_digits)))]
    
    if rows == 0 or cols == 0:
        if final_digits:
            n = len(final_digits)
            rows = int(math.sqrt(n)) or 1
            cols = math.ceil(n / rows)
            expected_cells = rows * cols
            logger.info(f"Fallback grid estimated: {rows} rows x {cols} cols (digits detected: {n})")
        else:
            logger.error("No candidate digits detected for grid estimation")
            return {"cells": cells, "digits": [], "est_rows": 0, "est_cols": 0}

    final_digits.sort(key=lambda r: r[1])
    groups = [sorted(final_digits[i:i + cols], key=lambda r: r[0])
              for i in range(0, len(final_digits), cols)]
    grouped = [d for group in groups for d in group]

    if expected_cells and len(grouped) != expected_cells:
        logger.warning(f"Grouped digits: {len(grouped)}; expected: {expected_cells}")
    else:
        logger.info(f"Grouped {len(grouped)} digits with grid {rows}x{cols}")
    
    if logger.isEnabledFor(logging.DEBUG):
        display_image(debug_img, "Cells (Red) and Digits (Green)")
    
    return {"cells": cells, "digits": grouped, "est_rows": rows, "est_cols": cols}

# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser(description="Treasure Maze Image Processor")
    parser.add_argument("-d", "--debug", action="store_true", help="Show debug info")
    parser.add_argument("-f", "--file", required=True, help="Path to image file")
    args = parser.parse_args()

    logger = setup_logger()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Debug {Fore.GREEN}enabled{Fore.RESET}!")
    
    warped = standardize(args.file)
    if warped is None:
        return 1
    
    result = process(warped)
    
    grid = np.zeros((28 * result["est_rows"], 28 * result["est_cols"]), dtype=np.uint8)
    for i, rect in enumerate(result["digits"]):
        digit_img = extract_digit(warped, rect)
        row, col = divmod(i, result["est_cols"])
        grid[row*28:(row+1)*28, col*28:(col+1)*28] = digit_img

    if logger.isEnabledFor(logging.DEBUG):
        display_image(grid, "Grid")
        
    logger.info("Exiting! 👋")
    return 0

if __name__ == "__main__":
    exit(main())
