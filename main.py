import cv2
import numpy as np
import time
from pathlib import Path
import sys


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
		print(f"* | Image loading [{time.time() - start:.4f}s]")
		
	def check_blurriness(self):
		start = time.time()
		blur = cv2.Laplacian(self.gray_image, cv2.CV_64F).var()

		print(f"* | Blurriness check [{time.time() - start:.4f}s]")
		print(f"* | Blurriness: {blur:.4f}")
		return blur < 100

	
	def process(self):
		start = time.time()
		blur = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
		_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
		binary = cv2.morphologyEx(
			binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
		)
		contours, _ = cv2.findContours(
			binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
		)

		if not contours:
			print(f"Processing took {time.time() - start:.4f} seconds")
			return None

		largest = max(contours, key=cv2.contourArea)
		epsilon = 0.1 * cv2.arcLength(largest, True)
		approx = cv2.approxPolyDP(largest, epsilon, True)
		print(f"* | Processing [{time.time() - start:.4f}s]")
		return approx, contours

def main(args=None):
	print("TreasureMaze")
	try:
		overall_start = time.time()
		image_path = input("* | Enter image path: ").strip() if args is None else args
		
		processor = ImageProcessor(image_path)
		is_blurry = processor.check_blurriness()
		if is_blurry:
			print("x | Image is too blurry!")
			prompt = input("Continue? (y/n): ")
			if prompt.lower() != "y":
				return
		result = processor.process()

		if not result:
			print("x | No contours found!")
			return

		approx, contours = result
		print(f"* | Found {len(contours)} contours")
		print(f"* | Approximated contour has {len(approx)} vertices")

		t_warp_start = time.time()
		src_pts = approx.reshape(-1, 2).astype(np.float32)
		s = src_pts.sum(axis=1)
		diff = np.diff(src_pts, axis=1)
		rect = np.float32(
			[
				src_pts[np.argmin(s)],
				src_pts[np.argmin(diff)],
				src_pts[np.argmax(s)],
				src_pts[np.argmax(diff)],
			]
		)

		padding = 1
		width = (
			int(
				max(
					np.linalg.norm(rect[0] - rect[1]), np.linalg.norm(rect[2] - rect[3])
				)
			)
			+ 2 * padding
		)
		height = (
			int(
				max(
					np.linalg.norm(rect[1] - rect[2]), np.linalg.norm(rect[3] - rect[0])
				)
			)
			+ 2 * padding
		)
		dst_pts = np.float32(
			[
				[padding, padding],
				[width - padding, padding],
				[width - padding, height - padding],
				[padding, height - padding],
			]
		)

		matrix = cv2.getPerspectiveTransform(rect, dst_pts)
		warped = cv2.warpPerspective(processor.color_image, matrix, (width, height))
		print(f"* | Warping [{time.time() - t_warp_start:.4f}s]")

		t_contour_start = time.time()
		gray_wr = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
		blur_wr = cv2.GaussianBlur(gray_wr, (5, 5), 0)
		_, binary_wr = cv2.threshold(
			blur_wr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
		)
		contours_wr, _ = cv2.findContours(
			binary_wr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
		)
		largest = max(contours_wr, key=cv2.contourArea)
		contours_wr_filtered = [
			c for c in contours_wr if cv2.contourArea(c) < cv2.contourArea(largest)
		]
		cv2.drawContours(warped, contours_wr, -1, (0, 0, 255), 2)
		cv2.drawContours(warped, contours_wr_filtered, -1, (0, 255, 0), 2)
		print(f"* | Contour detection [{time.time() - t_contour_start:.4f}s]")

		print(f"* | Overall runtime [{time.time() - overall_start:.4f}s]")
		cv2.imshow("Warped Image (Press any key to close)", warped)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
	except (ValueError, KeyboardInterrupt) as e:
		print(f"Error: {e}")
	finally:
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else None)