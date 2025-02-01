import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, List
import sys

@dataclass
class ImageProcessor:
    image: np.ndarray
    gray: np.ndarray = None
    binary: np.ndarray = None
    contours: List = None
    
    @staticmethod
    def validate_file_path(file_path: str) -> bool:
        path = Path(file_path)
        return (path.is_file() and 
                path.suffix.lower() in ('.png', '.jpg', '.jpeg'))
    
    @classmethod
    def from_file(cls, file_path: str) -> Optional['ImageProcessor']:
        if not cls.validate_file_path(file_path):
            return None
        
        img = cv2.imread(file_path)
        return cls(img) if img is not None else None
    
    def preprocess(self) -> None:
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(self.gray, (7, 7), 0)
        kernel = np.ones((3,3), np.uint8)
        morph = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
        _, self.binary = cv2.threshold(morph, 0, 255, 
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.binary = cv2.erode(self.binary, kernel, iterations=1)
        self.binary = cv2.dilate(self.binary, kernel, iterations=1)
    
    def detect_contours(self) -> Tuple[np.ndarray, List]:
        self.contours, _ = cv2.findContours(self.binary, 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        return self.get_largest_contour()
    
    def get_largest_contour(self) -> Tuple[np.ndarray, List]:
        if not self.contours:
            return None, []
        largest = max(self.contours, key=cv2.contourArea)
        epsilon = 0.1 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        return approx, self.contours

class ProcessingError(Exception):
    """Custom exception for image processing errors"""
    pass

def get_image_path() -> str:
    while True:
        try:
            file_path = input("Enter the path to your image file (or 'q' to quit): ").strip()
            if file_path.lower() in ('q', 'quit', 'exit'):
                print("\nExiting program...")
                sys.exit(0)
                
            file_path = file_path.replace("\"", "").replace("\'", "")
            file_path = file_path.lstrip("\"\'").rstrip("\"\'")
            
            if file_path:
                return file_path
            print("Error: No path provided.")
        except KeyboardInterrupt:
            print("\nExiting program...")
            sys.exit(0)

def display_contour_info(contours: List, approx: np.ndarray) -> None:
    print(f"\nDetected {len(contours)} contours in the image.")
    for i, contour in enumerate(contours):
        print(f"Contour {i}: X={cv2.boundingRect(contour)}")
    
    if approx is not None:
        print(f"\nApproximated rectangle: {cv2.boundingRect(approx)}")

def display_results(image: np.ndarray, approx: np.ndarray) -> None:
    """Display the processed image with contours"""
    cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
    cv2.imshow("Image (Press 'q' to close, 's' to save)", image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            try:
                cv2.imwrite("processed_image.png", image)
                print("Image saved as 'processed_image.png'")
            except Exception as e:
                print(f"Error saving image: {e}")
    
    cv2.destroyAllWindows()

def process_image(file_path: str) -> None:
    """Main image processing pipeline"""
    try:
        processor = ImageProcessor.from_file(file_path)
        if processor is None:
            raise ProcessingError("Invalid file or unable to read image")
            
        print("\nProcessing image...")
        processor.preprocess()
        approx, contours = processor.detect_contours()
        display_contour_info(contours, approx)
        
        if approx is not None:
            display_results(processor.image, approx)
    except ProcessingError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def main():
    print("=== Treasure Hunt Image Processor ===")
    print("Supported formats: PNG, JPG, JPEG")
    print("Commands: 'q' to quit, 's' to save when viewing image\n")
    
    while True:
        try:
            file_path = get_image_path()
            process_image(file_path)
            
            choice = input("\nProcess another image? (y/n): ").lower()
            if choice not in ['y', 'yes']:
                print("\nExiting program...")
                break
                
        except KeyboardInterrupt:
            print("\nExiting program...")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            break

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFatal error: {e}")
    finally:
        cv2.destroyAllWindows()
        sys.exit(0)