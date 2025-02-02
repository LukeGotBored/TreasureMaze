import cv2
import numpy as np
from pathlib import Path
import sys

class ImageProcessor:
    def __init__(self, image_path: str):
        self.image = self._load_image(image_path) if image_path else None
        
    def _load_image(self, image_path: str) -> np.ndarray:
        """Loads an image from the given path, and checks if it's valid."""
        image_path = image_path.strip("'\"")  # Clean up quotes
        path = Path(image_path) 
        if not (path.is_file() and path.suffix.lower() in ('.png', '.jpg', '.jpeg')):
            raise ValueError("Invalid image file.")
        
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError("Could not read image")
        return image
    
    def process(self):
        """Image processing pipeline"""
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        _, binary = cv2.threshold(blur, 0, 255, 
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE)
        
        # Check if any contours found, if so, return the largest one
        if not contours:
            return None
        
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.1 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        
        return approx, contours

def display_image(image: np.ndarray, contour):
    """Display image with detected contour"""
    if contour is not None:
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
    
    cv2.imshow("Image (Press any key to close)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main(args=None):
    print("TreasureMaze v0.3")
    
    try:
        # Get image path
        image_path = input("Enter image path: ").strip() if args is None else args
        
        # Process image
        processor = ImageProcessor(image_path)
        result = processor.process()
        
        if result:
            approx, contours = result
            print(f"Found {len(contours)} contours")
            print(f"Approximated contour has {len(approx)} vertices")
            print("\nCoordinates:")
            for i, point in enumerate(approx, 1):
                x, y = point[0]
                print(f"  Point {i}: ({x}, {y})")
            
            # Perspective correction stuff
            # Partially stolen from here: https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
            
            src_pts = approx.reshape(4, 2).astype(np.float32) # Get vertices of the approximated contour
            
            # Align the points in the correct order (top-left, top-right, bottom-right, bottom-left)
            rect = np.zeros((4, 2), dtype=np.float32)
            s = src_pts.sum(axis=1)
            rect[0] = src_pts[np.argmin(s)]  # Top-left
            rect[2] = src_pts[np.argmax(s)]  # Bottom-right
            diff = np.diff(src_pts, axis=1)
            rect[1] = src_pts[np.argmin(diff)]  # Top-right
            rect[3] = src_pts[np.argmax(diff)]  # Bottom-left
            src_pts = rect

            # Calculate width and height of the new image with padding
            padding = 1
            width = max(int(np.linalg.norm(src_pts[0] - src_pts[1])),
                    int(np.linalg.norm(src_pts[2] - src_pts[3]))) + (2 * padding)
            height = max(int(np.linalg.norm(src_pts[1] - src_pts[2])),
                        int(np.linalg.norm(src_pts[3] - src_pts[0]))) + (2 * padding)
            
            # Create destination points with padding
            dst_pts = np.array([
                [padding, padding],           # top-left
                [width - padding, padding],   # top-right
                [width - padding, height - padding],  # bottom-right
                [padding, height - padding]   # bottom-left
            ], dtype=np.float32)

            # Apply perspective transformation through CV2
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(processor.image, matrix, (width, height))
            
            gray_wr = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            blur_wr = cv2.GaussianBlur(gray_wr, (7, 7), 0)
            _, binary_wr = cv2.threshold(blur_wr, 0, 255, 
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Crop external borders
            border_threshold = 0  # Renamed from TRESHOLD
            binary_wr = binary_wr[border_threshold:binary_wr.shape[0]-border_threshold, border_threshold:binary_wr.shape[1]-border_threshold] 
            warped = warped[border_threshold:warped.shape[0]-border_threshold, border_threshold:warped.shape[1]-border_threshold]
            
            # Check if the borders are still present, if so, increment the threshold until there's no external border
            while binary_wr[0, 0] == 255 or binary_wr[0, binary_wr.shape[1]-1] == 255 or binary_wr[binary_wr.shape[0]-1, 0] == 255 or binary_wr[binary_wr.shape[0]-1, binary_wr.shape[1]-1] == 255:
                border_threshold += 1
                binary_wr = binary_wr[border_threshold:binary_wr.shape[0]-border_threshold, border_threshold:binary_wr.shape[1]-border_threshold]
                warped = warped[border_threshold:warped.shape[0]-border_threshold, border_threshold:warped.shape[1]-border_threshold]
            print(f"Final threshold: {border_threshold}")
        
            cv2.imshow("Warped", warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            

        else:
            print("No contours found!")
            
    except (ValueError, KeyboardInterrupt) as e:
        print(f"Error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Check if the user has provided an image path as an argument
    main(sys.argv[1] if len(sys.argv) > 1 else None)
