import cv2
import math
import numpy as np

def show_image(img):
   cv2.imshow("CV2", img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()

def get_next_cnt(h, i):
   return int(h[0][i][0])

def get_first_child(h, i):
   return int(h[0][i][2])

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

img = cv2.imread('samples/sample13.png')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (7, 7), 0)
_, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 

show_image(binary)

img_contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

grid = img_contours[0]
grid_rect = cv2.boundingRect(grid)

print(grid_rect)

cells = []

global cell_scan
cell_scan = get_first_child(hierarchy, 0)
while cell_scan != -1:
   
   new_cell = {}
   
   new_cell["idx"] = cell_scan
   new_cell["contour"] = img_contours[cell_scan]

   cell_rect = cv2.boundingRect(img_contours[cell_scan])
   cell_center = (math.floor(cell_rect[0] + cell_rect[2]/2), math.floor(cell_rect[1] + cell_rect[3]/2))

   new_cell["rect"] = cell_rect
   new_cell["center"] = cell_center

   cells.append(new_cell)
   
   cell_scan = get_next_cnt(hierarchy, cell_scan)

n_cells = len(cells)
print(n_cells)

grid_rows, grid_columns = estimate_grid_size(grid_rect, cells, n_cells)
print(grid_rows, grid_columns)

digits = [0 for i in range(0, n_cells)]

for cell in cells:
   
   new_digit = {}

   new_digit_idx = get_first_child(hierarchy, cell["idx"])

   new_digit["idx"] = new_digit_idx
   new_digit["contour"] = img_contours[new_digit_idx]
   new_digit["rect"] = cv2.boundingRect(img_contours[new_digit_idx])

   digit_row = math.floor((cell["center"][1] - grid_rect[1]) / grid_rect[3] * grid_rows) 
   digit_column = math.floor((cell["center"][0] - grid_rect[0]) / grid_rect[2] * grid_columns)

   digits[digit_row * grid_columns + digit_column] = new_digit

print(len(digits))

for digit in digits:
   newImage = img.copy()  
   cv2.drawContours(newImage, img_contours, digit["idx"], (0, 255, 0), 3)
   #cv2.circle(newImage, cell["center"], 3, (0, 0, 255), 3)
   show_image(newImage)
