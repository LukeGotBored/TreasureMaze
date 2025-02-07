import cv2
import numpy as np
import math
import time

def process_grid(warped):
    """Processes the warped maze image to extract grid cells and their details."""
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_blur = cv2.GaussianBlur(warped_gray, (7, 7), 0)
    _, warped_bin = cv2.threshold(warped_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_contours, hierarchy = cv2.findContours(warped_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if hierarchy is None:
        print("✘ | No hierarchy found in grid contours.")
        return {"cells": [], "rows": 0, "cols": 0}
    if not img_contours:
        print("✘ | No contours found in grid.")
        return {"cells": [], "rows": 0, "cols": 0}

    grid_rect = cv2.boundingRect(img_contours[0])

    # Helper functions to navigate the contour hierarchy.
    def get_next_cnt(h, i):
        return int(h[0][i][0]) if 0 <= i < h.shape[1] else -1

    def get_first_child(h, i):
        return int(h[0][i][2]) if 0 <= i < h.shape[1] else -1

    cells = []
    cell_scan = get_first_child(hierarchy, 0)
    
    # Traverse sibling contours representing cell boundaries
    while cell_scan != -1:
        c_rect = cv2.boundingRect(img_contours[cell_scan])
        c_center = (int(c_rect[0] + c_rect[2] / 2), int(c_rect[1] + c_rect[3] / 2))
        cells.append({"idx": cell_scan, "rect": c_rect, "center": c_center})
        cell_scan = get_next_cnt(hierarchy, cell_scan)

    if not cells:
        print("✘ | No cell contours found in grid.")
        return {"cells": [], "rows": 0, "cols": 0}

    n_cells = len(cells)
    avg_w = sum(c["rect"][2] for c in cells) / n_cells
    avg_h = sum(c["rect"][3] for c in cells) / n_cells
    rows = max(1, math.floor(grid_rect[3] / avg_h))
    cols = max(1, math.floor(n_cells / rows))

    # Check each cell for an internal digit contour.
    for cell in cells:
        digit_idx = get_first_child(hierarchy, cell["idx"])
        if digit_idx != -1:
            digit_rect = cv2.boundingRect(img_contours[digit_idx])
            cell["digit_rect"] = digit_rect

    digits = [0 for _ in range(n_cells)]
    for cell in cells:
        row = math.floor((cell["center"][1] - grid_rect[1]) / grid_rect[3] * rows)
        col = math.floor((cell["center"][0] - grid_rect[0]) / grid_rect[2] * cols)
        idx = row * cols + col
        digits[idx] = cell

    return {"cells": digits, "rows": rows, "cols": cols}

def extract_digit_from_cell(cell_img):
    """Extracts and returns the digit image from a cell using adaptive thresholding."""
    gray = cell_img if len(cell_img.shape) == 2 else cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    for thresh_val in range(int(mean_val - 30), int(mean_val + 30), 5):
        _, cell_thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        cell_thresh = cv2.morphologyEx(cell_thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(cell_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            # If the area is significant compared to the cell, consider it a digit.
            if cv2.contourArea(largest) > 0.1 * (cell_img.shape[0] * cell_img.shape[1]):
                x, y, w, h = cv2.boundingRect(largest)
                digit = cell_img[y:y+h, x:x+w]
                return digit, (x, y, w, h)
    return None, None

def log(message, level="INFO"):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [{level}] {message}")