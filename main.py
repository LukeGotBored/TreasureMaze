import cv2
import numpy as np
import time
from pathlib import Path
import sys

DEBUG = False


class ImageProcessor:
    def __init__(self, image_path: str):
        start = time.time()
        path = Path(image_path.strip("'\""))
        if not (path.is_file() and path.suffix.lower() in (".png", ".jpg", ".jpeg")):
            raise ValueError("✘ | Invalid image file.")

        self.color_image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if self.color_image is None:
            raise ValueError("Could not read image.")
        self.gray_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)
        print(f"* | Loading image [{time.time() - start:.4f}s]")

    def check_blurriness(self):
        start = time.time()
        blur = cv2.Laplacian(self.gray_image, cv2.CV_64F).var()
        print(f"* | Checking for blurriness [{time.time() - start:.4f}s]")
        print(f"* | Blur level: {blur:.4f}")
        return blur < 5

    def process(self):
        start = time.time()
        MAX = 512

        # Ottimizzazione: ridimensionare immagine mantenendo il rapporto
        h, w = self.color_image.shape[:2]
        if h > MAX or w > MAX:
            scale = MAX / max(h, w)
            new_dim = (int(w * scale), int(h * scale))
            self.color_image = cv2.resize(
                self.color_image, new_dim, interpolation=cv2.INTER_AREA
            )
            self.gray_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2GRAY)

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
        print(f"* | Processing completed [{time.time() - start:.4f}s]")
        return approx, contours


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
            print("✘ | Immagine troppo sfocata!")
            prompt = input("Continue anyways? (y/N): ")
            if prompt.lower() != "y":
                return
        result = processor.process()
        if not result:
            print("x | No contours found.")
            return

        approx, contours = result
        print(f"* | {len(contours)} contours found!")
        print(f"* | Approximated shape: {approx.shape[0]} sides")

        # Gestione robusta per avere 4 punti per il perspective transform
        if approx.shape[0] != 4:
            print(f"✘ | Approximated shape has {approx.shape[0]} sides")

            # Tenta di riapprossimare usando il contorno più grande
            largest = max(contours, key=cv2.contourArea)
            new_approx = cv2.approxPolyDP(
                largest, 0.02 * cv2.arcLength(largest, True), True
            )
            if new_approx.shape[0] == 4:
                approx = new_approx
                print("* | Re-approximated to 4 points.")
            else:
                print("✘ | Failed to obtain 4 points for perspective transform.")
                return

        # Perspective correction
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
        print(f"* | Warping completed in [{time.time() - t_warp_start:.4f}s]")

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
        if not contours_wr:
            print("✘ | No contours found during warped image processing.")
            return

        if DEBUG:
            cv2.imshow("Warped Output", warped)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        largest = max(contours_wr, key=cv2.contourArea)
        contours_wr_filtered = [
            c
            for c in contours_wr
            if cv2.contourArea(c) < cv2.contourArea(largest)
            and cv2.contourArea(c) > 100
        ]

        cv2.drawContours(warped, contours_wr, -1, (0, 0, 255), 2)
        cv2.drawContours(warped, contours_wr_filtered, -1, (0, 255, 0), 2)
        print(
            f"* | Contourn detection completed in [{time.time() - t_contour_start:.4f}s]"
        )
        print(f"* | Total runtime: {time.time() - overall_start:.4f}s")

        cv2.imshow("Warped Output", warped)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except (ValueError, KeyboardInterrupt) as e:
        print(f"✘ | Something went wrong: {e}")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
