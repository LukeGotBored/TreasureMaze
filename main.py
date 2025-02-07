import cv2
import numpy as np
import time
from pathlib import Path
import sys
import argparse
from utils import process_grid, extract_digit_from_cell, log

# --- CONFIG --- #

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
parser.add_argument(
	"-s", "--size", type=int, default=1024, help="The maximum size of an input image"
)
parser.add_argument(
	"--output-size", type=int, default=28, help="The output size of the digits"
)
parser.add_argument("--output-padding", type=int, default=10, help="Output padding")
parser.add_argument(
	"--img-resize",
	type=int,
	default=None,
	help="Image resize constant (overrides default image size)",
)


args, unknown = parser.parse_known_args()

DEBUG = args.debug
DEFAULT_IMG_SIZE = args.size
OUTPUT_SIZE = args.output_size
OUTPUT_PADDING = args.output_padding
IMG_RESIZE = args.img_resize if args.img_resize is not None else DEFAULT_IMG_SIZE


class ImageProcessor:
	"""Processa un'immagine per estrarre bordi e forme."""

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

		log(f"Image loaded in {time.time() - start:.4f}s")

	def check_blurriness(self) -> bool:
		"""Verifica se l'immagine è sfocata."""
		start = time.time()
		blur = cv2.Laplacian(self.gray_image, cv2.CV_64F).var()
		log(f"Blurriness check took {time.time() - start:.4f}s", "INFO")
		log(f"Blurriness level: {blur:.4f} ({'[!] Very Blurry' if blur < 5 else 'Blurry' if blur < 100 else 'Not Blurry'})")
		return blur < 100

	def process(self):
		"""Elabora l'immagine per rilevare contorni e approssimare la forma."""
		start = time.time()
		# Image Processing
		blur = cv2.GaussianBlur(self.gray_image, (9, 9), 0)
		thresh = cv2.adaptiveThreshold(
			blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 57, 7
		)
		_, binary = cv2.threshold(
			thresh, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
		)
		binary = cv2.morphologyEx(
			binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
		)

		if DEBUG:
			cv2.imshow("Binary Output", binary)
			cv2.waitKey(0)
			cv2.destroyWindow("Binary Output")

		contours, _ = cv2.findContours(
			binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
		)
		if not contours:
			log(f"No contours found. [{time.time() - start:.4f}s]", "ERROR")
			return None

		largest = max(contours, key=cv2.contourArea)
		epsilon = 0.1 * cv2.arcLength(largest, True)
		approx = cv2.approxPolyDP(largest, epsilon, True)
		log(f"Shape detection completed in {time.time() - start:.4f}s", "INFO")
		return approx, contours


def main(args=None):
	"""Funzione principale di esecuzione per TreasureMaze."""
	log("TreasureMaze | Initialising...")

	try:
		overall_start = time.time()
		image_path = (
			input("* | Insert the image path: ").strip() if args is None else args
		)

		processor = ImageProcessor(image_path)
		if processor.check_blurriness():
			log("Image appears blurry!", "INFO")
			if input("Continue anyways? (y/N): ").strip().lower() != "y":
				return

		result = processor.process()
		if not result:
			log("No contours found.", "ERROR")
			return

		approx, contours = result
		log(f"{len(contours)} contours found!", "INFO")
		log(f"Approximated shape: {approx.shape[0]} sides", "INFO")

		if approx.shape[0] != 4:
			log("Error: Shape approximation did not result in 4 points.", "ERROR")
			return

		# Image warping
		try:
			src_pts = approx.reshape(4, 2).astype(np.float32)
			rect = np.zeros((4, 2), dtype=np.float32)
			s = src_pts.sum(axis=1)
			rect[0] = src_pts[np.argmin(s)]
			rect[2] = src_pts[np.argmax(s)]
			diff = np.diff(src_pts, axis=1)
			rect[1] = src_pts[np.argmin(diff)]
			rect[3] = src_pts[np.argmax(diff)]
			src_pts = rect
		except Exception as e:
			log(f"Error while computing perspective transform points: {e}", "ERROR")
			return

		padding = 15
		width = max(
			int(np.linalg.norm(src_pts[0] - src_pts[1])),
			int(np.linalg.norm(src_pts[2] - src_pts[3])),
		) + (2 * padding)
		height = max(
			int(np.linalg.norm(src_pts[1] - src_pts[2])),
			int(np.linalg.norm(src_pts[3] - src_pts[0])),
		) + (2 * padding)

		try:
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
		except Exception as e:
			log(f"Error during perspective transformation: {e}", "ERROR")
			return

		log(f"Warping completed: {time.time() - overall_start:.4f}s", "INFO")

		# if DEBUG is active, run the grid detection and show the results (TODO - remove, it's just bloating the output)
		if DEBUG:
			gray_wr = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
			blur_wr = cv2.GaussianBlur(gray_wr, (7, 7), 0)
			thresh_wr = cv2.adaptiveThreshold(
				blur_wr, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 57, 7
			)
			_, binary_wr = cv2.threshold(
				thresh_wr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
			)
			contours_wr, _ = cv2.findContours(
				binary_wr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
			)
			if contours_wr:
				largest = max(contours_wr, key=cv2.contourArea)
				grid_rect = cv2.boundingRect(largest)

				def contour_inside_grid(c):
					x, y, w, h = grid_rect
					return any(
						x <= point[0][0] <= x + w and y <= point[0][1] <= y + h
						for point in c
					)

				contours_wr_filtered = [
					c
					for c in contours_wr
					if cv2.contourArea(c) < cv2.contourArea(largest)
					and cv2.contourArea(c) > 200
					and contour_inside_grid(c)
				]
				log(f"Grid debug: {len(contours_wr_filtered)} filtered contours.", "INFO")

		# Utilizza la funzione process_grid da utils.py per processare la griglia
		grid_data = process_grid(warped)
		if not grid_data["cells"]:
			log("Unable to process the grid.", "ERROR")
		else:
			log(f"Analyzed cells: {len(grid_data['cells'])}", "INFO")
			log(
				f"Estimated grid size: {grid_data['rows']}x{grid_data['cols']}", "INFO"
			)

		# Salva ogni digit in un canvas 28x28 e li elabora in batch
		padding = 15
		digits = []
		for cell in grid_data["cells"]:
			if "digit_rect" in cell:
				x, y, w, h = cell["digit_rect"]
				roi = warped[max(y, 0) : y + h, max(x, 0) : x + w]
				roi = cv2.copyMakeBorder(
					roi,
					padding,
					padding,
					padding,
					padding,
					cv2.BORDER_CONSTANT,
					value=(255, 255, 255),
				)
				roi = cv2.resize(roi, (28, 28))
				roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
				_, roi_thresh = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)

				# blur ever so slightly to make the digit look natural
				roi_thresh = cv2.GaussianBlur(roi_thresh, (3, 3), 0)
				digits.append(roi_thresh)

		if DEBUG:
			# Draw a grid of digits in a single image
			canvas = np.zeros(
				(28 * grid_data["rows"], 28 * grid_data["cols"]), dtype=np.uint8
			)
			for i, digit in enumerate(digits):
				row = i // grid_data["cols"]
				col = i % grid_data["cols"]
				canvas[row * 28 : row * 28 + 28, col * 28 : col * 28 + 28] = digit

			cv2.imshow("Digits Grid", canvas)
			cv2.waitKey(0)
			cv2.destroyWindow("Digits Grid")

		if DEBUG:
			cv2.drawContours(warped, contours_wr, -1, (0, 0, 255), 2)
			cv2.drawContours(warped, contours_wr_filtered, -1, (0, 255, 0), 2)
			cv2.drawContours(warped, [largest], -1, (255, 255, 0), 2)
			for cell in grid_data["cells"]:
				if "digit_rect" in cell:
					cv2.rectangle(
						warped,
						(cell["digit_rect"][0], cell["digit_rect"][1]),
						(
							cell["digit_rect"][0] + cell["digit_rect"][2],
							cell["digit_rect"][1] + cell["digit_rect"][3],
						),
						(255, 0, 255),
						2,
					)
			cv2.imshow("Warped Image (Press any key to close)", warped)
			cv2.waitKey(0)
			cv2.destroyWindow("Warped Image (Press any key to close)")

	except (ValueError, KeyboardInterrupt) as e:
		log(f"Error: {e}", "ERROR")
	finally:
		cv2.destroyAllWindows()


if __name__ == "__main__":
	main(sys.argv[1] if len(sys.argv) > 1 else None)
