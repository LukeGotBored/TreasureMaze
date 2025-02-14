import argparse
import logging
import os
import math

import cv2
import numpy as np
from colorama import Fore, init
from logger import setup_logger
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
            logger.error("Could not find 4 corners")
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


# ---------------- Grid & Digit Extraction ----------------

def process(image: np.ndarray) -> dict:
    logger = logging.getLogger('TreasureMaze')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.error("No contours found")
        return None
    
    # Filter noise and exclude the largest contour (the grid)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[1:]
    contours = [c for c in contours if cv2.contourArea(c) > 100]

    # Compute average contour height to determine a dynamic row threshold
    heights = [cv2.boundingRect(c)[3] for c in contours]
    avg_height = sum(heights) / len(heights) if heights else 0
    row_threshold = int(avg_height * 0.5)  # Adjust multiplier as needed

    # Sort by y-coordinate
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    rows = []

    for i, c in enumerate(contours):
        x, y, w, h = cv2.boundingRect(c)
        if i == 0 or not rows or y > rows[-1][0] - row_threshold:
            rows.append([y + h, []])
        rows[-1][1].append(c)

    # Sort each row's contours from left to right
    for row in rows:
        row[1] = sorted(row[1], key=lambda c: cv2.boundingRect(c)[0])

    # Flatten the list of contours
    sorted_contours = [c for row in rows for c in row[1]]

    # Open the image to remove noise
    opened = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    rects = [cv2.boundingRect(c) for c in sorted_contours]

    # Estimate grid size
    est_rows = len(rows)
    est_cols = max(len(row[1]) for row in rows)
    logger.info(f"Estimated grid size: {est_cols}x{est_rows}")
    test_img = cv2.cvtColor(opened, cv2.COLOR_GRAY2BGR)
    test_cnt = cv2.drawContours(test_img, sorted_contours, -1, (0, 255, 0), 3)
    

    # ????????????????    
    
        
    if logger.isEnabledFor(logging.DEBUG):
        display_image(test_img, "Bounding Rectangles")

        

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
    parser.add_argument("-m", "--model", required=True, help="Path to trained model")
    args = parser.parse_args()

    logger = setup_logger()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug(f"Debug {Fore.GREEN}enabled{Fore.RESET}!")
    
    warped = standardize(args.file)
    if warped is None:
        return 1
    
    result = process(warped)
    digits = []
    
    grid = np.zeros((28 * result["est_rows"], 28 * result["est_cols"]), dtype=np.uint8)
    for i, rect in enumerate(result["digits"]):
        digit_img = extract_digit(warped, rect)
        digit_emn = cv2.bitwise_not(digit_img)
        digit_emn = cv2.flip(digit_emn, 1)
        digit_emn = cv2.rotate(digit_emn, cv2.ROTATE_90_COUNTERCLOCKWISE)
        digits.append(digit_emn)
        row, col = divmod(i, result["est_cols"])
        grid[row*28:(row+1)*28, col*28:(col+1)*28] = digit_img

    if logger.isEnabledFor(logging.DEBUG):
        display_image(grid, "Grid")
        
    
    # Now, let's load the model and predict the digits
    import tensorflow as tf
    from keras import models
    import math
        
    if(not os.path.exists(args.model)):
        logger.error(f"Model file not found: {args.model}")
        return 1
    
    logger.info(f"Loading model from {args.model}")
    model = models.load_model(args.model)
    logger.info("Model loaded successfully!")
    
    
    predictions = []
    for i, digit in enumerate(digits):
        digit = digit.reshape(1, 28, 28, 1)
        prediction = model.predict(digit)
        pred_label = np.argmax(prediction)
        predictions.append(label_map[pred_label])
        logger.info(f"PREDICTION {i+1}: {label_map[pred_label]}")
        

    logger.info(f"Predictions: {predictions}")
    
    
    
    logger.info("Exiting! 👋")
    return 0

if __name__ == "__main__":
    exit(main())
