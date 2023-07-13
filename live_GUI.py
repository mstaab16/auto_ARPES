import sys
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from pyqtgraph import GraphicsView, ImageItem
import pyqtgraph as pg
import numpy as np
import imagezmq


class ImageDisplayGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Display GUI")
        self.resize(250, 250)
        # Create a central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create a GraphicsView for the scatter plot
        self.scatter_view = pg.GraphicsView()
        self.scatter_plot = pg.PlotItem()
        self.scatter_view.setCentralItem(self.scatter_plot)
        layout.addWidget(self.scatter_view)

        # Create a GraphicsView for the image display
        self.image_view = GraphicsView()
        layout.addWidget(self.image_view)

        # Create an ImageItem to display the image
        self.image_item = ImageItem()
        self.image_view.addItem(self.image_item)

        # Create a button to start and stop the image updates
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        self.start_button.clicked.connect(self.start_image_updates)
        self.stop_button.clicked.connect(self.stop_image_updates)

        self.image_hub = imagezmq.ImageHub(open_port='tcp://*:5551')

        # Create a QTimer to trigger the image updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_image)

        # Create a ScatterPlotItem for the scatter plot
        self.scatter_plot_item = pg.ScatterPlotItem()
        self.scatter_plot.addItem(self.scatter_plot_item)

        
    def start_image_updates(self):
        # Start the QTimer to trigger image updates
        self.timer.start(1000)  # 10 milliseconds interval

        # Update button states
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_image_updates(self):
        # Stop the QTimer from triggering image updates
        self.timer.stop()

        # Update button states
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def display_image(self):
        img_type, data = self.image_hub.recv_image()
        if img_type=='ARPES':
            self.image_item.setImage(np.fliplr(data.T))
            self.resize(data.shape[0], data.shape[1]*2)
            self.image_hub.send_reply(b'OK')
        elif img_type=='position':
            print(data)
            self.scatter_plot_item.setData(pos=data)
            self.image_hub.send_reply(b'OK')
        else:
            self.stop_image_updates()
            self.image_hub.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    gui = ImageDisplayGUI()
    gui.show()
    sys.exit(app.exec())

