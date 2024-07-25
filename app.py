import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget, QMessageBox, QHBoxLayout, QTextEdit
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os

class SteelDefectApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.original_pixmap = None
        self.processed_pixmap = None
        self.is_processed = False
        self.defect_info = ""
        self.temp_image_file = 'temp_image.jpg'
        self.processed_image_file = None
        self.original_image_path = None

        self.class_mapping = {
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5'
        }

    def initUI(self):
        self.setWindowTitle('Steel Defect Detector')
        self.setGeometry(100, 100, 900, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: 2px solid black;")  # Black border for image
        layout.addWidget(self.label)

        self.info_label = QTextEdit(self)
        self.info_label.setFixedHeight(100)  # Fixed height to ensure it's small
        self.info_label.setReadOnly(True)
        self.info_label.setStyleSheet("font-size: 12px;")  # Smaller font size
        layout.addWidget(self.info_label)

        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.exit_btn = QPushButton('Exit', self)
        self.exit_btn.clicked.connect(self.close)
        button_layout.addWidget(self.exit_btn)

        self.load_btn = QPushButton('Load Image', self)
        self.load_btn.clicked.connect(self.load_image)
        button_layout.addWidget(self.load_btn)

        self.process_btn = QPushButton('Process Image', self)
        self.process_btn.clicked.connect(self.process_image)
        button_layout.addWidget(self.process_btn)

        self.clear_btn = QPushButton('Clear Image', self)
        self.clear_btn.clicked.connect(self.clear_image)
        button_layout.addWidget(self.clear_btn)

        self.return_btn = QPushButton('Return', self)
        self.return_btn.clicked.connect(self.return_image)
        self.return_btn.setEnabled(False)
        button_layout.addWidget(self.return_btn)

        self.setFixedSize(self.size())

        self.show()

    def load_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Images (*.png *.jpg *.jpeg)', options=options)
        if fileName:
            self.original_pixmap = QPixmap(fileName)
            self.display_image(self.original_pixmap)
            self.processed_pixmap = None
            self.is_processed = False
            self.defect_info = ""
            self.info_label.setText(self.defect_info)
            self.return_btn.setEnabled(False)
            self.original_image_path = fileName

    def process_image(self):
        if not self.original_pixmap:
            QMessageBox.warning(self, 'No Image Loaded', 'Please load an image before processing.')
            return

        self.original_pixmap.save(self.temp_image_file)

        model = YOLO('best_mod.pt')
        results = model(self.temp_image_file)

        img = results[0].plot()

        self.defect_info = ""
        for result in results[0].boxes:
            x1, y1, x2, y2 = result.xyxy[0].tolist()
            confidence = result.conf[0].item()
            label_index = int(result.cls[0].item()) + 1  
            class_name = self.class_mapping.get(label_index, 'Unknown') 
            self.defect_info += f"Координаты: ({x1:.2f}, {y1:.2f}), ({x2:.2f}, {y2:.2f}) // Класс дефекта: {class_name} // Вероятность: {confidence:.2f}\n"

            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        if self.original_image_path:
            base_name = os.path.splitext(os.path.basename(self.original_image_path))[0]
            directory = os.path.dirname(self.original_image_path)
            self.processed_image_file = os.path.join(directory, f"{base_name}_result.jpg")
            result_image = Image.fromarray(img)
            result_image.save(self.processed_image_file)
            self.processed_pixmap = QPixmap(self.processed_image_file)

        if self.is_processed:
            self.display_image(self.original_pixmap)
            self.is_processed = False
        else:
            self.display_image(self.processed_pixmap)
            self.is_processed = True

        self.info_label.setText(self.defect_info)

        self.return_btn.setEnabled(True)

    def return_image(self):
        if self.is_processed:
            self.display_image(self.original_pixmap)
            self.is_processed = False
        else:
            self.display_image(self.processed_pixmap)
            self.is_processed = True

    def display_image(self, pixmap):
        self.label.setPixmap(pixmap.scaled(self.label.size(), Qt.KeepAspectRatio))

    def clear_image(self):
        self.label.clear()
        self.info_label.clear()
        self.original_pixmap = None
        self.processed_pixmap = None
        self.is_processed = False
        self.defect_info = ""
        self.return_btn.setEnabled(False) 

    def closeEvent(self, event):
        if os.path.exists(self.temp_image_file):
            os.remove(self.temp_image_file)
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = SteelDefectApp()
    sys.exit(app.exec_())
