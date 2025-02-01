#-- TODO List --#
# 1. Read the image
# 2. Apply processing
#   2.1 Convert to grayscale
#   2.2 Apply Gaussian blur
#   2.3 Apply Canny edge detection
# 3. Extract numbers and letters (through KEROS OCR)
# 4. Get position, label and confidence, and convert image to a matrix
# 5. Show the image and the matrix in a GUI (for now)

import cv2
import numpy as np
from pathlib import Path

def load_image():
    while True:
        file_path = input("Enter the path to your image file: ").strip()
        file_path = file_path.replace("\"", "").replace("\'", "")
        file_path = file_path.lstrip("\"\'").rstrip("\"\'")
        
        if not file_path:
            print("Error: No path provided.")
            continue
        elif not Path(file_path).is_file():
            print("Error: Invalid file path.")
            continue
        elif not file_path.lower().endswith((".png", ".jpg", ".jpeg")):
            print("Error: Invalid file format. Please use PNG, JPG, or JPEG.")
            continue

        # Read the image
        img = cv2.imread(file_path)
        if img is None:
            print("Error: Unable to read the image.")
            continue
        break

    # Image processing (May god have mercy on my soul)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    _, binarized = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    binarized = cv2.erode(binarized, kernel, iterations=1)
    binarized = cv2.dilate(binarized, kernel, iterations=1)

    # External grid detection (1/3)
    contours, _ = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    
    print("Detected {} contours in the image.".format(len(contours)))
    for i, contour in enumerate(contours):
        print("Contour {}: X={}, Y={}, W={}, H={}".format(i, *cv2.boundingRect(contour)))
    
    # Approximate to a rectangle
    epsilon = 0.1 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    print("\nApproximated rectangle x: {}, y: {}, w: {}, h: {}".format(*cv2.boundingRect(approx)))

    cv2.drawContours(img, [approx], -1, (0, 255, 0), 3)
    cv2.imshow("Image", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    print("Treasure Hunt | Starting...")
    print("Select an image (PNG, JPG, JPEG)")
    load_image()
    
if __name__ == "__main__":
    main()