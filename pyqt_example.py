import sys
from PyQt5.QtWidgets import QMainWindow, QApplication

import numpy as np

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')


class Plotter(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        FigureCanvas.__init__(self, fig)
        self.setParent(parent)
        self.plot()

    def plot(self):
        x = np.linspace(0, 2*np.pi, 100)
        self.ax.plot(x, np.sin(x))

class Gui(QMainWindow):
    def __init__(self, top=300, left=300, width=1100, height=700):
        super().__init__()
        self.setWindowTitle('Fourier Filter')
        self.setGeometry(top, left, width, height)
        self.plotter = Plotter(self, width/100, height/100, dpi=100)


app = QApplication(sys.argv)
window = Gui()
window.show()
app.exec()
