import cv2
import numpy as np
import time
from pathlib import Path
import sys
import math


DEBUG = True
DEFAULT_IMG_SIZE = 512
IMG_RESIZE = DEFAULT_IMG_SIZE


class ImageProcessor:
    def __init__(self, image_path: str):
        start = time.time()
        path = Path(image_path.strip("'\""))
        if not (path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg")):
            raise ValueError("Invalid image file.")

        self.color_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if self.color_image is None:
            raise ValueError("Could not read image.")
        self.gray_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)

        h, w = self.color_image.shape[:2]
        if h > w:
            new_h = IMG_RESIZE
            new_w = int(w * (new_h / h))
        else:
            new_w = IMG_RESIZE
            new_h = int(h * (new_w / w))
        self.color_image = cv2.resize(self.color_image, (new_w, new_h))
        self.gray_image = cv2.resize(self.gray_image, (new_w, new_h))

        print(f"* | Image loaded in {time.time() - start:.4f}s")

    def check_blurriness(self):
        start = time.time()
        blur = cv2.Laplacian(self.gray_image, cv2.CV_64F).var()
        print(f"* | Blurriness check: {time.time() - start:.4f}s")
        print(f"* | Blurriness level: {blur:.4f}")
        return blur < 100

    def process(self):
        start = time.time()

        blur = cv2.GaussianBlur(self.gray_image, (9, 9), 0)
        tresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 57, 7
        )
        _, binary = cv2.threshold(
            tresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        binary = cv2.morphologyEx(
            binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
        )

        if DEBUG:
            cv2.imshow("Binary Output", binary)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            print(f"✘ | No contours found. [{time.time() - start:.4f}s]")
            return None

        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.1 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        print(f"* | Shape detection completed in {time.time() - start:.4f}s")
        return approx, contours


def process_grid(warped):
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_blur = cv2.GaussianBlur(warped_gray, (7, 7), 0)
    _, warped_bin = cv2.threshold(
        warped_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    img_contours, hierarchy = cv2.findContours(
        warped_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if not img_contours:
        print("✘ | No contours found in grid.")
        return None

    grid_rect = cv2.boundingRect(img_contours[0])

    def get_next_cnt(h, i):
        if i < 0 or i >= h.shape[1]:
            return -1
        return int(h[0][i][0])
    def get_first_child(h, i):
        if i < 0 or i >= h.shape[1]:
            return -1
        return int(h[0][i][2])

    cells = []
    cell_scan = get_first_child(hierarchy, 0)
    while cell_scan != -1:
        c_rect = cv2.boundingRect(img_contours[cell_scan])
        c_center = (
            int(c_rect[0] + c_rect[2] / 2),
            int(c_rect[1] + c_rect[3] / 2),
        )
        cells.append({"idx": cell_scan, "rect": c_rect, "center": c_center})
        cell_scan = get_next_cnt(hierarchy, cell_scan)
    
    if not cells:
        print("✘ | No cell contours found in grid.")
        return []  # Restituisce lista vuota per evitare divisione per zero

    n_cells = len(cells)
    avg_w = sum([c["rect"][2] for c in cells]) / n_cells
    avg_h = sum([c["rect"][3] for c in cells]) / n_cells
    rows = max(1, math.floor(grid_rect[3] / avg_h))
    cols = max(1, math.floor(n_cells / rows))
    print(f"* | Estimated grid: {rows}x{cols}")

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
    return digits


def main(args=None):
    print("TreasureMaze")
    try:
        overall_start = time.time()
        image_path = (
            input("* | Insert the image path: ").strip() if args is None else args
        )

        processor = ImageProcessor(image_path)
        is_blurry = processor.check_blurriness()
        if is_blurry:
            print("Image appears blurry!")
            prompt = input("Continue anyways? (y/N): ")
            if prompt.lower() != "y":
                return
        result = processor.process()
        if not result:
            print("No contours found.")
            return

        approx, contours = result
        print(f"* | {len(contours)} contours found!")
        print(f"* | Approximated shape: {approx.shape[0]} sides")

        t_warp_start = time.time()
        src_pts = approx.reshape(4, 2).astype(np.float32)
        rect = np.zeros((4, 2), dtype=np.float32)
        s = src_pts.sum(axis=1)
        rect[0] = src_pts[np.argmin(s)]
        rect[2] = src_pts[np.argmax(s)]
        diff = np.diff(src_pts, axis=1)
        rect[1] = src_pts[np.argmin(diff)]
        rect[3] = src_pts[np.argmax(diff)]
        src_pts = rect

        padding = 15
        width = max(
            int(np.linalg.norm(src_pts[0] - src_pts[1])),
            int(np.linalg.norm(src_pts[2] - src_pts[3])),
        ) + (2 * padding)
        height = max(
            int(np.linalg.norm(src_pts[1] - src_pts[2])),
            int(np.linalg.norm(src_pts[3] - src_pts[0])),
        ) + (2 * padding)

        dst_pts = np.array(
            [
                [padding, padding],
                [width - padding, padding],
                [width - padding, height - padding],
                [padding, height - padding],
            ],
            dtype=np.float32,
        )

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(processor.color_image, matrix, (width, height))
        print(f"* | Warping completed: {time.time() - t_warp_start:.4f}s")

        t_contour_start = time.time()
        gray_wr = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur_wr = cv2.GaussianBlur(gray_wr, (7, 7), 0)
        treshold_wr = cv2.adaptiveThreshold(
            blur_wr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 57, 7
        )
        _, binary_wr = cv2.threshold(
            treshold_wr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours_wr, _ = cv2.findContours(
            binary_wr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        largest = max(contours_wr, key=cv2.contourArea)
        grid_rect = cv2.boundingRect(largest)

        def contour_inside_grid(c):
            x, y, w, h = grid_rect
            M = cv2.moments(c)
            if M["m00"] == 0:
                return False
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (x <= cx <= x + w) and (y <= cy <= y + h)

        contours_wr_filtered = [
            c
            for c in contours_wr
            if cv2.contourArea(c) < cv2.contourArea(largest)
            and cv2.contourArea(c) > 200
            and contour_inside_grid(c)
        ]

        print(f"* | Contours detection completed: {time.time() - t_contour_start:.4f}s")
        print(f"* | Total runtime: {time.time() - overall_start:.4f}s")

        grid_data = process_grid(warped)
        if not grid_data:
            print("Unable to process the grid.")
        else:
            print(f"* | Analyzed cells: {len(grid_data)}")

        if DEBUG:
            cv2.drawContours(warped, contours_wr, -1, (0, 0, 255), 2)
            cv2.drawContours(warped, contours_wr_filtered, -1, (0, 255, 0), 2)
            cv2.drawContours(warped, [largest], -1, (255, 255, 0), 2)
            for cell in grid_data:
                if not isinstance(cell, dict):
                    continue
                x, y, w, h = cell["rect"]
                cv2.rectangle(warped, (x, y), (x + w, y + h), (255, 0, 0), 2)
                if "digit_rect" in cell:
                    dx, dy, dw, dh = cell["digit_rect"]
                    cv2.rectangle(warped, (dx, dy), (dx + dw, dy + dh), (0, 255, 255), 2)
            cv2.imshow("Warped Image (Press any key to close)", warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except (ValueError, KeyboardInterrupt) as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
