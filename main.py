import argparse
import logging
import os
import math
import cv2
import numpy as np
from colorama import Fore, init
import colorsys

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

# ---------------- Utility Functions ----------------


def display_image(image: np.ndarray, title: str = "Image") -> None:
    cv2.imshow(title, image)
    if cv2.waitKey(0) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        exit(0)
    else:
        cv2.destroyAllWindows()


def check_blurriness(image: np.ndarray) -> float:
    # ? Source: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ---------------- Image Preprocessing Functions ----------------


def preprocess(image: np.ndarray) -> np.ndarray:
    if image is None:
        logger.error("Invalid image")
        return None
    try:
        h, w = image.shape[:2]
        logger.debug(f"Image dimensions: {w}x{h}")

        # Resize only if the image is larger or smaller than the maximum size
        if max(h, w) > MAX_SIZE or min(h, w) < MAX_SIZE:
            ratio = MAX_SIZE / float(max(h, w))
            new_size = (int(w * ratio), int(h * ratio))
            image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
            logger.debug(f"Resized image to {new_size[0]}x{new_size[1]}")

        fm = check_blurriness(image)
        logger.debug(f"Focus measure: {fm:.2f}")
        if fm < MIN_BLUR_THRESHOLD:
            logger.warning(
                f"{'Very blurry' if fm < 10 else 'Blurry'} image (FM={fm:.2f})"
            )

        # Apply image processing steps
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
    # ? Source: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
    try:
        contours, _ = cv2.findContours(
            thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
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
        w = (
            int(
                max(
                    np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])
                )
            )
            + 2 * padding
        )
        h = (
            int(
                max(
                    np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[3] - rect[0])
                )
            )
            + 2 * padding
        )

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
    logger = logging.getLogger("TreasureMaze")

    img_contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )

    grid, grid_idx = get_largest_child(hierarchy, img_contours, -1)
    grid_rect = cv2.boundingRect(grid)

    if logger.isEnabledFor(logging.DEBUG):
        newImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(newImage, img_contours, grid_idx, (0, 255, 0), 3)
        display_image(newImage)

    cells = []

    global cell_scan
    cell_scan = get_first_child(hierarchy, grid_idx)
    while cell_scan != -1:
        if cv2.contourArea(img_contours[cell_scan]) > 1000:

            new_cell = {}

            new_cell["idx"] = cell_scan
            new_cell["contour"] = img_contours[cell_scan]
            cell_rect = cv2.boundingRect(img_contours[cell_scan])

            cell_center = (
                math.floor(cell_rect[0] + cell_rect[2] / 2),
                math.floor(cell_rect[1] + cell_rect[3] / 2),
            )

            new_cell["rect"] = cell_rect
            new_cell["center"] = cell_center

            cells.append(new_cell)

        cell_scan = get_next_cnt(hierarchy, cell_scan)

    n_cells = len(cells)

    grid_rows, grid_columns = estimate_grid_size(grid_rect, cells, n_cells)
    logger.info(f"Estimated size: {grid_rows}x{grid_columns} | Found {n_cells} cells ({'match' if grid_rows * grid_columns == n_cells else 'mismatch'})")

    digits = [0 for i in range(0, n_cells)]

    for cell in cells:
        new_digit = {}
        digit_cnt, digit_idx = get_largest_child(hierarchy, img_contours, cell["idx"])

        new_digit["idx"] = digit_idx
        new_digit["contour"] = digit_cnt
        new_digit["rect"] = cv2.boundingRect(digit_cnt)

        digit_row = math.floor(
            (cell["center"][1] - grid_rect[1]) / grid_rect[3] * grid_rows
        )
        digit_column = math.floor(
            (cell["center"][0] - grid_rect[0]) / grid_rect[2] * grid_columns
        )

        digits[digit_row * grid_columns + digit_column] = new_digit

    logger.info(f"Found digits: {len(digits)}")
    
    # Convert the grayscale image to BGR once, then extract ROIs from this conversion
    newImage = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def pad_to_square(image: np.ndarray, target_size: int) -> np.ndarray:
        h, w = image.shape[:2]
        if w != h:
            border = abs(w - h) // 2
            if w > h:
                image = cv2.copyMakeBorder(
                    image, border, border, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
            else:
                image = cv2.copyMakeBorder(
                    image, 0, 0, border, border, cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
        return cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)

    digits_img = []
    for digit in digits:
        # Extract region of interest from the pre-converted image
        x, y, w, h = digit["rect"]
        digit_img = newImage[y:y + h, x:x + w]

        # Pad the image to form a square then add the fixed border
        digit_img = pad_to_square(digit_img, EMN_DIGIT_SIZE)
        digit_img = cv2.copyMakeBorder(
            digit_img,
            EMN_BORDER_SIZE,
            EMN_BORDER_SIZE,
            EMN_BORDER_SIZE,
            EMN_BORDER_SIZE,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
        
        # Mirror the digit
        digit_img = cv2.flip(digit_img, 1)

        # Invert colors if the image is predominantly light
        if np.mean(digit_img) > 127:
            digit_img = cv2.bitwise_not(digit_img)

        # Rotate for final orientation.
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
            # Convert the digit image to grayscale and resize to the expected 28x28 dimensions with a single channel
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


# ---------------- Main ----------------



def main():
    parser = argparse.ArgumentParser(description="Treasure Maze Image Processor")
    parser.add_argument("-d", "--debug", action="store_true", help="Show debug info")
    parser.add_argument("-f", "--file", required=True, help="Path to image file")
    parser.add_argument("-m", "--model", required=True, help="Path to trained model")

    args = parser.parse_args()

    # Setup logger
    logger = setup_logger()
    logger.setLevel(logging.INFO)

    if args.debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled!")

    img = get_img(args.file)
    img_standard = standardize(img)

    if img_standard is None:
        return 1

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
                if pred is None:
                    cell_output = "None"
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
