# region Imports
import sys
import os
import logging
from typing import List, Tuple, Optional

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt, QSize, QThreadPool, QRunnable, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QColor, QAction, QPixmap, QFontDatabase
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QStatusBar,
    QProgressBar,
    QFrame,
    QGraphicsView,
    QGraphicsScene,
    QMenuBar,
    QSizePolicy,
    QMenu,
)

try:
    from main import get_img, standardize, extract_digits, predict_digit, pathfind
except ImportError as e:
    logging.error("Error importing image processing functions: %s", e)
    raise

# Region: Constants and Styles
CELL_SIZE = 50
CONFIDENCE_HIGH = 0.9
CONFIDENCE_MEDIUM = 0.7
DEFAULT_WINDOW_SIZE = (1000, 700)
IMAGE_PREVIEW_SIZE = (400, 400)
BORDER_RADIUS = 6
BUTTON_PADDING = "8px 16px"
FONT_FAMILY = QFontDatabase.systemFont(QFontDatabase.SystemFont.GeneralFont).family()
MONOSPACE_FONT = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont).family() # TODO fix this

logger = logging.getLogger("TreasureMazeGUI")
logger.setLevel(logging.INFO)

# region Style Definitions
DARK_MODE = {
    "name": "dark",
    "primary": "#2b2b2b",
    "secondary": "#1e1e1e",
    "text": "#e0e0e0",
    "border": "#454545",
    "button": """
        QPushButton {
            background-color: #3a3a3a;
            color: #e0e0e0;
            border: 1px solid #454545;
            border-radius: """
    + str(BORDER_RADIUS)
    + """px;
            padding: """
    + BUTTON_PADDING
    + """;
        }
        QPushButton:hover {
            background-color: #454545;
            border-color: #505050;
        }
        QPushButton:pressed {
            background-color: #2e2e2e;
        }
        QPushButton:disabled {
            background-color: #2a2a2a;
            color: #6a6a6a;
        }
    """,
    "console": """
        QPlainTextEdit {
            background-color: #1e1e1e;
            color: #e0e0e0;
            border: 1px solid #3a3a3a;
            border-radius: """
    + str(BORDER_RADIUS)
    + """px;
            font-family: """
    + MONOSPACE_FONT
    + """;
            font-size: 13px;
            padding: 5px;
        }
    """,
    "progress": """
        QProgressBar {
            border: none;
            background: transparent;
            height: 20px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4a9eff;
            border-radius: """
    + str(BORDER_RADIUS)
    + """px;
        }
    """,
    "panel": """
        QFrame {
            background-color: #1e1e1e;
            border: none;
            border-radius: """
    + str(BORDER_RADIUS)
    + """px;
        }
    """,
    "grid_border": "none",
    "grid_bg": "#1e1e1e",
    "confidence_high": "#388E3C",
    "confidence_medium": "#F57C00",
    "confidence_low": "#D32F2F",
}

LIGHT_MODE = {
    "name": "light",
    "primary": "#f0f0f0",
    "secondary": "#ffffff",
    "text": "#2b2b2b",
    "border": "#d0d0d0",
    "button": """
        QPushButton {
            background-color: #ffffff;
            color: #2b2b2b;
            border: 1px solid #d0d0d0;
            border-radius: """
    + str(BORDER_RADIUS)
    + """px;
            padding: """
    + BUTTON_PADDING
    + """;
        }
        QPushButton:hover {
            background-color: #f8f8f8;
            border-color: #c0c0c0;
        }
        QPushButton:pressed {
            background-color: #e8e8e8;
        }
        QPushButton:disabled {
            background-color: #f5f5f5;
            color: #a0a0a0;
        }
    """,
    "console": """
        QPlainTextEdit {
            background-color: #ffffff;
            color: #2b2b2b;
            border: 1px solid #d0d0d0;
            border-radius: """
    + str(BORDER_RADIUS)
    + """px;
            font-family: """
    + MONOSPACE_FONT
    + """;
            font-size: 13px;
            padding: 5px;
        }
    """,
    "progress": """
        QProgressBar {
            border: none;
            background: transparent;
            height: 20px;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #4a9eff;
        }
    """,
    "panel": """
        QFrame {
            background-color: #ffffff;
            border: none;
            border-radius: """
    + str(BORDER_RADIUS)
    + """px;
        }
    """,
    "grid_border": "none",
    "grid_bg": "#ffffff",
    "confidence_high": "#81C784",
    "confidence_medium": "#FFB74D",
    "confidence_low": "#E57373",
}


# region Worker and Processing Class Definitions
class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    result = pyqtSignal(list, int, int)
    error = pyqtSignal(str)
    finished = pyqtSignal()


class ImageProcessingWorker(QRunnable):
    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path
        self.signals = WorkerSignals()
        self._is_running = True

    def run(self):
        try:
            self.signals.progress.emit(10)
            input_img = get_img(self.image_path)
            if input_img is None:
                raise ValueError("Failed to load image")
            self.signals.progress.emit(30)
            standardized = standardize(input_img)
            if standardized is None:
                raise ValueError("Failed to standardize image")
            self.signals.progress.emit(50)
            extracted = extract_digits(standardized)
            if not extracted:
                raise ValueError("No digits extracted")
            self.signals.progress.emit(70)
            model_path = os.path.join(os.getcwd(), "model", "out", "cnn.keras")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            predictions = predict_digit(extracted["digits"], model_path)
            self.signals.progress.emit(90)
            self.signals.result.emit(
                predictions,
                extracted["grid"].get("rows", 0),
                extracted["grid"].get("columns", 0),
            )
        except Exception as e:
            self.signals.error.emit(f"Processing error: {str(e)}")
        finally:
            self.signals.finished.emit()
            self.signals.progress.emit(100)

    def cancel(self):
        self._is_running = False


# region Grid Visualizer Class
class GridVisualizer(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.current_style = DARK_MODE
        self.setRenderHints(QtGui.QPainter.RenderHint.Antialiasing)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.placeholder_text = None
        self._init_style()  # Renamed from apply_style() for clarity
        self.show_no_grid_message()

    def _init_style(self):  # Initial style application
        self.setStyleSheet(f"""
            QGraphicsView {{
                background-color: {self.current_style['grid_bg']};
                border: none;
            }}
        """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_placeholder_position()

    def _update_placeholder_position(self):
        if self.placeholder_text:
            view_rect = self.viewport().rect()
            t_rect = self.placeholder_text.boundingRect()
            center_x = (view_rect.width() - t_rect.width()) / 2
            center_y = (view_rect.height() - t_rect.height()) / 2
            self.placeholder_text.setPos(
                self.mapToScene(int(center_x), int(center_y))
            )

    def show_no_grid_message(self):
        self.scene.clear()
        self.placeholder_text = self.scene.addText("No grid loaded")
        self.placeholder_text.setDefaultTextColor(QColor(self.current_style["text"]))
        self.placeholder_text.setFont(QFont(FONT_FAMILY, 14))
        self._update_placeholder_position()

    def update_grid(self, predictions: List[Tuple[str, float]], rows: int, cols: int):
        self.scene.clear()
        self.placeholder_text = None
        
        if not predictions or rows == 0 or cols == 0:
            self.show_no_grid_message()
            return

        # region Adding Cells to Grid
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(predictions):
                    continue
                pred, confidence = predictions[idx]
                text = pred if pred else "?"
                self._add_cell(j, i, text, confidence)
        self.setSceneRect(0, 0, cols * CELL_SIZE, rows * CELL_SIZE)

    def _add_cell(self, x: int, y: int, text: str, confidence: float):
        # color = self._get_confidence_color(confidence) 
        color = self.current_style["grid_bg"]
        rect = self.scene.addRect(
            x * CELL_SIZE,
            y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE,
            pen=QtGui.QPen(QColor(self.current_style["text"]), 0.5),
            brush=QtGui.QBrush(QColor(color)),
        )
        rect.setZValue(-1)
        text_item = self.scene.addText(text)
        text_item.setDefaultTextColor(QColor(self.current_style["text"]))
        text_item.setFont(QFont(FONT_FAMILY, 14, QFont.Weight.Bold))
        t_rect = text_item.boundingRect()
        text_item.setPos(
            x * CELL_SIZE + (CELL_SIZE - t_rect.width()) / 2,
            y * CELL_SIZE + (CELL_SIZE - t_rect.height()) / 2,
        )

    def _get_confidence_color(self, confidence: float) -> str:
        if confidence >= CONFIDENCE_HIGH:
            return self.current_style["confidence_high"]
        if confidence >= CONFIDENCE_MEDIUM:
            return self.current_style["confidence_medium"]
        return self.current_style["confidence_low"]

    def set_theme(self, style: dict):
        self.current_style = style
        self._init_style()  # Use the same method for consistency
        if self.placeholder_text:
            self.placeholder_text.setDefaultTextColor(QColor(self.current_style["text"]))
            self._update_placeholder_position()

        if self.scene.items():
            self.update_grid([], 0, 0) #! This is really hacky, it'll delete the grid when changing themes, but to be fair it's 11pm and I'm tired    
            


# region Main GUI Class
class TreasureMazeGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_style = DARK_MODE
        self.image_path = ""
        self.thread_pool = QThreadPool()
        self.active_worker: Optional[ImageProcessingWorker] = None
        self.selected_algorithm = "BFS"
        self.latest_predictions = []
        self.latest_rows = 0
        self.latest_cols = 0
        self.solution_positions = []   # List of (col, row) on grid for the solution path
        self.current_step_idx = 0      # Index into solution_positions for step-by-step display
        self.solution_overlay_items = []  # Overlay graphical items for the displayed solution
        self.btn_prev = None  # Initialize btn_prev
        self.btn_next = None  # Initialize btn_next
        self.init_ui()
        self.apply_styles()
        self.setAcceptDrops(True)

    def init_ui(self):
        self.setWindowTitle("Treasure Maze Analyzer")
        self.setMinimumSize(*DEFAULT_WINDOW_SIZE)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        left_panel = self._create_image_panel()
        main_layout.addWidget(left_panel, 35)
        right_panel = self._create_results_panel()
        main_layout.addWidget(right_panel, 65)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.progress_bar = QProgressBar()
        self.progress_bar.hide()
        self.status_bar.addPermanentWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)
        self.status_bar.setContentsMargins(10, 5, 10, 5)
        self._create_menu()

    def _create_image_panel(self):
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        self.image_label = QLabel("Drop image here or click 'Open Image'")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(*IMAGE_PREVIEW_SIZE)
        layout.addWidget(self.image_label)
        return panel

    def _create_results_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        # region Grid Visualizer Area
        grid_container = QFrame()
        grid_container_layout = QVBoxLayout(grid_container)
        grid_container_layout.setContentsMargins(0, 0, 0, 0)
        grid_container_layout.setSpacing(0)
        self.grid_visualizer = GridVisualizer()
        self.grid_visualizer.setMinimumSize(400, 300)
        grid_container_layout.addWidget(self.grid_visualizer)
        btn_panel = QWidget()
        hbox = QHBoxLayout(btn_panel)
        hbox.setContentsMargins(5, 5, 5, 5)
        hbox.setSpacing(10)

        # Add algorithm settings button
        self.btn_algo = QPushButton("⚙️")
        self.btn_algo.setToolTip("Select Algorithm")
        self.btn_algo.setFixedSize(32, 32)
        # change the style to remove the padding
        self.btn_algo.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.current_style['primary']};
                color: {self.current_style['text']};
                border: 1px solid {self.current_style['border']};
                border-radius: {BORDER_RADIUS}px;
                padding: 0;
            }}
            QPushButton:hover {{
                background-color: {self.current_style['secondary']};
            }}
        """)
        self.btn_algo.clicked.connect(self.select_algorithm)
        hbox.addWidget(self.btn_algo)

        # Add play button
        self.btn_play = QPushButton("▶️")
        self.btn_play.setToolTip("Run Pathfinding")
        self.btn_play.setFixedSize(32, 32)
        self.btn_play.clicked.connect(self.run_pathfinding)
        self.btn_play.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.current_style['primary']};
                color: {self.current_style['text']};
                border: 1px solid {self.current_style['border']};
                border-radius: {BORDER_RADIUS}px;
                padding: 0;
            }}
            QPushButton:hover {{
                background-color: {self.current_style['secondary']};
            }}
        """)
        self.btn_play.setEnabled(False)
        
        hbox.addWidget(self.btn_play)

        btn_prev = QPushButton("⬅️")
        btn_prev.setToolTip("Previous move")
        btn_prev.setEnabled(False)
        btn_prev.clicked.connect(self.previous_move)
        hbox.addWidget(btn_prev)
        self.btn_prev = btn_prev # Assign to the class attribute

        btn_next = QPushButton("➡️")
        btn_next.setToolTip("Next move")
        btn_next.setEnabled(False)
        btn_next.clicked.connect(self.next_move)
        hbox.addWidget(btn_next)
        self.btn_next = btn_next # Assign to the class attribute

        self.btn_skip = QPushButton("⏭️")
        self.btn_skip.setToolTip("Go to last move")
        self.btn_skip.setFixedSize(32, 32)
        self.btn_skip.clicked.connect(self.goto_last_move)

        self.btn_skip.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.current_style['primary']};
                color: {self.current_style['text']};
                border: 1px solid {self.current_style['border']};
                border-radius: {BORDER_RADIUS}px;
                padding: 0;
            }}
            QPushButton:hover {{
                background-color: {self.current_style['secondary']};
            }}
            QPushButton:disabled {{
                background-color: {self.current_style['primary']};
                color: {self.current_style['text']};
            }}
        """)
        self.btn_skip.setEnabled(False)

        hbox.addWidget(self.btn_skip)


        grid_container_layout.addWidget(btn_panel)
        layout.addWidget(grid_container, 60)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setPlaceholderText("Processing output will appear here...")
        layout.addWidget(self.console, 25)
        control_layout = QHBoxLayout()
        self.btn_open = QPushButton("📂 Open Image")
        self.btn_open.clicked.connect(self.open_file_dialog)
        self.btn_process = QPushButton("🔎 Analyze")
        self.btn_process.clicked.connect(self.start_processing)
        self.btn_process.setEnabled(False)
        control_layout.addWidget(self.btn_open)
        control_layout.addWidget(self.btn_process)
        layout.addLayout(control_layout)
        return panel

    def previous_move(self):
        if self.solution_positions and self.current_step_idx > 0:
            self._update_navigation_buttons()
            self.current_step_idx -= 1
            self.update_solution_overlay()
        else:
            print("Already at the first step")
        self._update_navigation_buttons()

    def next_move(self):
        if self.solution_positions and self.current_step_idx < len(self.solution_positions) - 1:
            self.current_step_idx += 1
            self.update_solution_overlay()
        else:
            print("Already at the final step")
        self._update_navigation_buttons()

    def update_solution_overlay(self):
        # Remove previous overlay items
        for item in self.solution_overlay_items:
            self.grid_visualizer.scene.removeItem(item)
        self.solution_overlay_items = []
        if not self.solution_positions:
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.btn_skip.setEnabled(False)
            return
        # Draw path from start up to current_step_idx as red lines
        pen = QtGui.QPen(QtGui.QColor("red"), 2)
        # Compute cell center positions (using CELL_SIZE)
        centers = []
        for pos in self.solution_positions[:self.current_step_idx + 1]:
            x = pos[0] * CELL_SIZE + CELL_SIZE / 2
            y = pos[1] * CELL_SIZE + CELL_SIZE / 2
            centers.append((x, y))
        # Draw lines between consecutive centers
        for i in range(len(centers) - 1):
            line = QtWidgets.QGraphicsLineItem(centers[i][0], centers[i][1], centers[i+1][0], centers[i+1][1])
            line.setPen(pen)
            self.grid_visualizer.scene.addItem(line)
            self.solution_overlay_items.append(line)
        self._update_navigation_buttons()

    def _update_navigation_buttons(self):
        if not self.solution_positions:
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.btn_skip.setEnabled(False)
        else:
            self.btn_prev.setEnabled(self.current_step_idx > 0)
            self.btn_next.setEnabled(self.current_step_idx < len(self.solution_positions) - 1)
            self.btn_skip.setEnabled(self.current_step_idx < len(self.solution_positions) - 1)

    def goto_last_move(self):
        if self.solution_positions:
            self.current_step_idx = len(self.solution_positions) - 1
            self.update_solution_overlay()
            self._update_navigation_buttons()

    def _create_menu(self):
        menu_bar = QMenuBar()
        file_menu = menu_bar.addMenu("&File")
        open_action = QAction("Open...", self)
        open_action.triggered.connect(self.open_file_dialog)
        file_menu.addAction(open_action)
        theme_menu = menu_bar.addMenu("&Theme")
        dark_action = QAction("Dark Mode", self)
        dark_action.triggered.connect(lambda: self.set_theme(DARK_MODE))
        light_action = QAction("Light Mode", self)
        light_action.triggered.connect(lambda: self.set_theme(LIGHT_MODE))
        theme_menu.addAction(dark_action)
        theme_menu.addAction(light_action)
        self.setMenuBar(menu_bar)

    def apply_styles(self):
        base_style = f"""
            QMainWindow {{
                background-color: {self.current_style['primary']};
                color: {self.current_style['text']};
            }}
            {self.current_style['button']}
            {self.current_style['console']}
            {self.current_style['panel']}
            {self.current_style['progress']}
            QLabel {{
                font: 14px {FONT_FAMILY};
                color: {self.current_style['text']};
                background: transparent;
            }}
            QStatusBar {{
                background-color: {self.current_style['secondary']};
                border: none;
            }}
        """
        self.setStyleSheet(base_style)
        self.grid_visualizer.set_theme(self.current_style)
        self.progress_bar.setStyleSheet(self.current_style["progress"])

    def set_theme(self, style: dict):
        self.current_style = style
        self.apply_styles()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            self.load_image(urls[0].toLocalFile())

    def load_image(self, path: str):
        if not os.path.exists(path):
            self.show_error("File not found")
            return
        self.image_path = path
        pixmap = QPixmap(path).scaled(
            *IMAGE_PREVIEW_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)
        self.btn_process.setEnabled(True)
        self.status_label.setText(f"Loaded: {os.path.basename(path)}")

    def open_file_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", os.getcwd(), "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if path:
            self.load_image(path)

    def start_processing(self):
        if not self.image_path:
            self.show_error("No image selected")
            return
        self.console.clear()
        self.progress_bar.show()
        self.btn_process.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.status_label.setText("Processing...")

        # Clear solution-related attributes
        self.solution_positions = []
        self.current_step_idx = 0
        self.solution_overlay_items = []
        self._update_navigation_buttons()

        self.active_worker = ImageProcessingWorker(self.image_path)
        self.active_worker.signals.progress.connect(self.progress_bar.setValue)
        self.active_worker.signals.result.connect(self.handle_results)
        self.active_worker.signals.error.connect(self.show_error)
        self.active_worker.signals.finished.connect(self.processing_finished)
        self.thread_pool.start(self.active_worker)

    def handle_results(self, predictions: list, rows: int, cols: int):
        self.latest_predictions = predictions
        self.latest_rows = rows
        self.latest_cols = cols
        self.grid_visualizer.update_grid(predictions, rows, cols)
        self.console.appendPlainText(
            f"Processed {rows}x{cols} grid\n"
            f"Identified {len(predictions)} numbers\n"
            f"Confidence rate: {len(predictions)/(rows*cols)*100:.1f}%"
        )
        self.status_label.setText("Analysis complete")
        self.btn_play.setEnabled(bool(predictions) and bool(self.selected_algorithm))

    def run_pathfinding(self):
        if not self.latest_predictions:
            self.show_error("No digit predictions available. Please analyze an image first.")
            return
        self.status_label.setText(f"Calculating path using {self.selected_algorithm}")
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        QApplication.processEvents()
        try:
            result = pathfind(
                self.selected_algorithm,
                self.latest_rows,
                self.latest_cols,
                self.latest_predictions
            )
            sol = result.get("solution", [])
            self.console.appendPlainText(f"Pathfinding result:\nSteps: {sol}\nCost: {result.get('cost')}\nLength: {result.get('length')}")
            if sol:
                self.solution_positions = self.compute_solution_positions(sol)
                self.current_step_idx = 0
                self.update_solution_overlay()
                self.btn_prev.setEnabled(True)  # Enable the "previous" button
                self.btn_next.setEnabled(True)  # Enable the "next" button
            else:
                self.console.appendPlainText("No solution found.")
        except Exception as e:
            self.show_error(str(e))
        self.progress_bar.setValue(100)
        self.status_label.setText("Ready")
        self.progress_bar.hide()
        self._update_navigation_buttons()

    def compute_solution_positions(self, moves: list) -> list:
        # Find the starting cell by looking for the first "S" in predictions.
        start = (0, 0)
        for idx, (label, _) in enumerate(self.latest_predictions):
            if label == "S":
                start = (idx % self.latest_cols, idx // self.latest_cols)
                break
        positions = [start]
        curr = start
        delta = {"UP": (0, -1), "DOWN": (0, 1), "LEFT": (-1, 0), "RIGHT": (1, 0)}
        for act in moves:
            d = delta.get(act)
            if d:
                curr = (curr[0] + d[0], curr[1] + d[1])
                positions.append(curr)
        return positions

    def update_solution_overlay(self):
        # Remove previous overlay items
        for item in self.solution_overlay_items:
            self.grid_visualizer.scene.removeItem(item)
        self.solution_overlay_items = []
        if not self.solution_positions:
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.btn_skip.setEnabled(False)
            return
        # Draw path from start up to current_step_idx as red lines
        pen = QtGui.QPen(QtGui.QColor("red"), 2)
        # Compute cell center positions (using CELL_SIZE)
        centers = []
        for pos in self.solution_positions[:self.current_step_idx + 1]:
            x = pos[0] * CELL_SIZE + CELL_SIZE / 2
            y = pos[1] * CELL_SIZE + CELL_SIZE / 2
            centers.append((x, y))
        # Draw lines between consecutive centers
        for i in range(len(centers) - 1):
            line = QtWidgets.QGraphicsLineItem(centers[i][0], centers[i][1], centers[i+1][0], centers[i+1][1])
            line.setPen(pen)
            self.grid_visualizer.scene.addItem(line)
            self.solution_overlay_items.append(line)
        self._update_navigation_buttons()

    def _update_navigation_buttons(self):
        if not self.solution_positions:
            self.btn_prev.setEnabled(False)
            self.btn_next.setEnabled(False)
            self.btn_skip.setEnabled(False)
        else:
            self.btn_prev.setEnabled(self.current_step_idx > 0)
            self.btn_next.setEnabled(self.current_step_idx < len(self.solution_positions) - 1)
            self.btn_skip.setEnabled(self.current_step_idx < len(self.solution_positions) - 1)

    def goto_last_move(self):
        if self.solution_positions:
            self.current_step_idx = len(self.solution_positions) - 1
            self.update_solution_overlay()
            self._update_navigation_buttons()

    def select_algorithm(self):
        menu = QMenu(self)
        algos = ["BFS", "DFS", "UCS", "Best First Graph", "A*"]
        for algo in algos:
            action = QAction(algo, self)
            if algo == self.selected_algorithm:
                action.setCheckable(True)
                action.setChecked(True)
            action.triggered.connect(lambda checked, a=algo: self.set_algorithm(a))
            menu.addAction(action)
        menu.exec(QtGui.QCursor.pos())

    def set_algorithm(self, algo: str):
        self.selected_algorithm = algo
        self.status_label.setText(f"Selected Algorithm: {algo}")
        self.btn_play.setEnabled(bool(self.latest_predictions) and bool(algo))

    def processing_finished(self):
        self.progress_bar.hide()
        self.btn_process.setEnabled(True)
        self.btn_open.setEnabled(True)
        self.active_worker = None

    def show_error(self, message: str):
        self.console.appendPlainText(f"[ERROR] {message}")
        self.status_bar.showMessage(message, 5000)
        self.status_label.setText("Error")
        self.processing_finished()

    def closeEvent(self, event):
        if self.active_worker:
            self.active_worker.cancel()
        event.accept()


def main():
    app = QApplication(sys.argv)
    font = QFont(FONT_FAMILY)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    window = TreasureMazeGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
