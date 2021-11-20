import sys
import numpy as np
import matplotlib.figure as mpl_fig
import matplotlib.animation as anim
import random

# from PyQt5.QtWidgets import QHBoxLayout, QMainWindow, QApplication, QPushButton, \
# QLineEdit, QGridLayout, QLabel, QCheckBox, QVBoxLayout, QWidget, QSlider, QMessageBox
# from PyQt5.QtGui import QPixmap, QDoubleValidator, QPalette, QColor
# from PyQt5.QtCore import Qt, QLocale
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from math import pi
from scipy.fft import rfft, irfft, rfftfreq


class FourierFilter:
    def __init__(self, ndots, noise_amp = 1, signal = None):
        self.ndots = ndots
        self.noise_amp = noise_amp
        if signal is None:
            self.random()
        else:
            self.signals_params = signal
        self.init_signal()
        self.filter_signal()

    def init_signal(self):    
        self.global_time = 1
        self.time = np.linspace(0, 1, num=self.ndots)
        self.signal = np.zeros(shape=(self.ndots))
        for params in self.signals_params:
            self.signal += params[0] * np.sin(2*pi*params[1]*self.time + params[2])
        self.noisy_signal = self.signal + (np.random.rand(self.ndots)-0.5)*self.noise_amp*2

    def filter_signal(self):
        self.transformed = rfft(self.noisy_signal)
        self.freqs = rfftfreq(self.transformed.size, d=1./self.ndots)
        self.power = np.abs(self.transformed)
        self.mask = np.ones(len(self.transformed), dtype=bool)
        tmp_mask = np.zeros(len(self.transformed), dtype=bool)
        n = 1
        while (not(tmp_mask == self.mask).all()):
            median = np.median(self.power[self.mask])
            std = np.std(self.power[self.mask])
            tmp_mask = self.mask
            self.mask = abs(self.power - median) <= n*std
            n+=1
        self.transformed[self.mask] = 0
        self.recovered = irfft(self.transformed)


    def time_step(self):
        new_value = 0
        self.global_time += 1/self.ndots
        for params in self.signals_params:
            new_value += params[0] * np.sin(2*pi*params[1]*self.global_time + params[2])
        self.signal = np.append(self.signal[1:], new_value)
        new_noisy_value = new_value + (np.random.rand()-0.5)*self.noise_amp*2
        self.noisy_signal = np.append(self.noisy_signal[1:], new_noisy_value)
        old_recovered = self.recovered
        self.filter_signal()
        self.recovered = np.append(old_recovered[1:], self.recovered[-1:])

    def random(self):
        self.signals_params = [ [[], [], []] for _ in range(3)]
        for i in range(3):
            self.signals_params[i][0] = random.randint(0, 10)
            self.signals_params[i][1] = random.randint(0, 10)
            self.signals_params[i][2] = random.random() * np.pi
            self.noise_amp = random.randint(0, 10)

    def set(self): #for tests
        self.filter_signal()


class FFPlotter(FigureCanvas, anim.FuncAnimation):
    def __init__(self, ff:FourierFilter):
        self.ff = ff
        FigureCanvas.__init__(self, mpl_fig.Figure())
        self.plot = self.figure.subplots(nrows=3)
        self.figure.tight_layout() 
        self.active = False
        self.need_update = True
        self.redraw = False
        self.init_graphs()
        self.set_limits()
        anim.FuncAnimation.__init__(self, self.figure, self._update_canvas_, interval=20, blit=True)

    def init_graphs(self):
        self.plot_signal()
        self.plot_freqs()
        self.plot_recover()
        self.add_grid()

    def plot_signal(self):
        x = np.linspace(0, 1, num=self.ff.ndots)
        self.signal_line, = self.plot[0].plot(x, self.ff.signal, 'tab:orange', label='Signal')
        self.noisy_signal_line, = self.plot[0].plot(x, self.ff.noisy_signal, 'tab:blue', label='Noisy signal')
        self.plot[0].set_xlabel('time, s')

    def plot_freqs(self):
        x1, x2, y1, y2 = self.get_freqs_data()
        self.freqs_line1 = self.plot[1].scatter(x1, y1, color='tab:orange', label='Filtered out')
        self.freqs_line2 = self.plot[1].scatter(x2, y2, color='tab:blue')
        self.plot[1].set_xlabel('freqs, Hz')

    def plot_recover(self):
        x = np.linspace(0, 1, num=self.ff.ndots)
        self.signal_line2, = self.plot[2].plot(x, self.ff.signal, 'tab:orange', label='Signal')
        self.recovered_line, = self.plot[2].plot(x, self.ff.recovered, label='Recovered signal')
        self.plot[2].set_xlabel('time, s')
    
    def add_grid(self):
        for i in range(3):
            self.plot[i].grid(b=True, which='major', color='#666666', linestyle='-')
            self.plot[i].minorticks_on()
            self.plot[i].grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
            self.plot[i].legend(loc = 2)
        
    def set_limits(self):
        signal_limit = self.ff.noise_amp
        for signal in self.ff.signals_params:
            signal_limit += signal[0]
        self.plot[0].set_ylim(-signal_limit, signal_limit)
        self.plot[2].set_ylim(-signal_limit, signal_limit)
        self.plot[1].set_ylim(0, 1.5*max(self.ff.power))
        self.plot[1].set_xlim(0, len(self.ff.power))
        if self.redraw:
            self.draw()

    def _update_canvas_(self, i):
        if self.active or self.need_update:
            if self.need_update:
                self.set_limits()
                self.need_update = False
            self.ff.time_step()
            x = np.linspace(0, 1, self.ff.ndots)
            self.signal_line.set_data(x, self.ff.signal)
            self.noisy_signal_line.set_data(x, self.ff.noisy_signal)
            self.recovered_line.set_data(x, self.ff.recovered)
            self.signal_line2.set_data(x, self.ff.signal)
            x1, x2, y1, y2 = self.get_freqs_data()
            self.freqs_line1.set_offsets (np.stack((x1, y1), axis=-1))
            self.freqs_line2.set_offsets (np.stack((x2, y2), axis=-1))
        return self.signal_line, self.noisy_signal_line, self.recovered_line, self.signal_line2, self.freqs_line1, self.freqs_line2

    def get_freqs_data(self):
        x = np.linspace(0, int(self.ff.ndots/2), num=int(self.ff.ndots/2)+1)
        x1 = x[self.ff.mask==True]
        y1 = self.ff.power[self.ff.mask==True]
        x2 = x[self.ff.mask==False]
        y2 = self.ff.power[self.ff.mask==False]
        return x1, x2, y1, y2
    
    def set_redraw(self, state):
        self.redraw = state
    

# class Gui(QMainWindow):
#     def __init__(self, top=300, left=300, width=1100, height=700, ff:FourierFilter=None):
#         super().__init__()
#         self.ff = ff
#         self.validator = QDoubleValidator()
#         self.validator.setLocale(QLocale("en_US"))
#         layout = self.init_widgets(width, height)
#         widget = QWidget()
#         widget.setLayout(layout)
#         self.setCentralWidget(widget)

#     def init_widgets(self, width, height):
#         layout = QHBoxLayout()
#         right_layout = QVBoxLayout()
#         grid = QGridLayout()
#         grid.setSpacing(15)
#         self.signal_widgets = []
#         for i, name in zip(range(3), ["Amplitude", "Frequency", "Phase pi "]):
#             lable = QLabel(self, text=name)
#             lable.setFixedHeight(22)
#             grid.addWidget(lable, 0, i+1)
#         for i in range(1, 4):
#             grid.addWidget(QLabel(f"Signal №{i}"), i, 0)
#             self.signal_widgets.append([])
#             for j in range(1, 3):
#                 area = QLineEdit(str(self.ff.signals_params[i-1][j-1]), self)
#                 grid.addWidget(area, i, j)
#                 area.setValidator(self.validator)
#                 area.textEdited.connect(self.update_signal_props)
#                 self.signal_widgets[i-1].append(area)
#             phase_widget = QSlider(Qt.Horizontal, self)
#             phase_widget.setMaximum(100)
#             phase_widget.setSingleStep(1)
#             phase_widget.setValue(int(self.ff.signals_params[i-1][2] / np.pi * 100))
#             phase_widget.setMinimumSize(100, 10)
#             grid.addWidget(phase_widget, i, 3)
#             phase_widget.sliderMoved.connect(self.update_signal_props)
#             self.signal_widgets[i-1].append(phase_widget)

#         grid.addWidget(QLabel("Noise amplitude"), 4, 0, 1, 2, alignment=Qt.AlignRight)
#         self.noise = QLineEdit(str(self.ff.noise_amp), self)
#         self.noise.setValidator(self.validator)
#         self.noise.textEdited.connect(self.update_signal_props)
#         grid.addWidget(self.noise, 4, 2, 1, 2, alignment=Qt.AlignLeft )

#         grid.addWidget(QLabel("Measurements number"), 5, 0, 1, 2, alignment=Qt.AlignRight)
#         self.measurements = QSlider(Qt.Horizontal, self)
#         self.measurements.setMinimum(1)
#         self.measurements.setMaximum(40)
#         self.measurements.setValue(int(self.ff.ndots/10))
#         self.measurements.setTickInterval(10)
#         self.measurements.setSingleStep(10)
#         self.measurements.setPageStep(10)
#         self.measurements.setTickPosition(QSlider.TicksBelow)
#         self.measurements.valueChanged.connect(self.update_signal_props)
#         grid.addWidget(self.measurements, 5, 2)
#         self.measurements_number = QLabel(str(self.ff.ndots))
#         grid.addWidget(self.measurements_number, 5, 3, alignment=Qt.AlignCenter)

#         random_button = QPushButton("Generate random signal", self)
#         random_button.clicked.connect(self.generate_random)
#         grid.addWidget(random_button, 6, 1, alignment=Qt.AlignCenter)

#         self.animation_button = QPushButton("Toggle animation", self)
#         self.animation_button.setCheckable(True)
#         self.animation_button.clicked.connect(self.toggle_animation)
#         grid.addWidget(self.animation_button, 6, 2, alignment=Qt.AlignCenter)

#         grid.addWidget(QLabel("Redraw ticks labels (slow)"), 7, 1)
#         self.check = QCheckBox(self)
#         self.check.clicked.connect(lambda state: self.plotter.set_redraw(state))
#         grid.addWidget(self.check, 7, 2)

#         save_button = QPushButton("Save to file", self)
#         save_button.clicked.connect(self.save)
#         grid.addWidget(save_button, 8, 1, 1, 2)
 
#         self.plotter = FFPlotter(ff=self.ff)
#         self.plotter.setFixedHeight(height)
#         layout.addWidget(self.plotter, 2, Qt.AlignLeft)
#         right_layout.setSpacing(20)

#         right_layout.addLayout(grid)
#         right_layout.addWidget(QLabel(self))
#         layout.addLayout(right_layout)
#         return layout

#     def toggle_animation(self):
#         if self.animation_button.isChecked():
#             self.animation_button.setStyleSheet("background-color : lightblue")
#             self.plotter.active = True
#         else:
#             self.plotter.active = False
#             self.animation_button.setStyleSheet("background-color : lightgrey")

#     def update_signal_props(self):
#         new_signal = []
#         try:
#             for i in range(3):
#                 new_signal.append([])
#                 for j in range (2):
#                     new_signal[i].append(float(self.signal_widgets[i][j].text()))
#                 new_signal[i].append(float(self.signal_widgets[i][2].value()*np.pi/100))
#             noise = float(self.noise.text())
#             measurements = int(self.measurements.value()) * 10
#             self.measurements_number.setText(str(measurements))
#         except ValueError:
#             return
#         self.ff.signals_params = new_signal
#         self.ff.noise_amp = noise
#         self.ff.ndots = measurements
#         self.ff.init_signal()
#         self.ff.filter_signal()
#         self.plotter.need_update = True

#     def generate_random(self):
#         self.ff.random()
#         for i in range(3):
#                 for j in range (2):
#                     self.signal_widgets[i][j].setText(str(self.ff.signals_params[i][j]))
#                 self.signal_widgets[i][2].setValue(int(self.ff.signals_params[i][2]/np.pi*100))
#         self.noise.setText(str(self.ff.noise_amp))
#         self.update_signal_props()

#     def save(self):
#         pix = QPixmap(self.size())
#         self.render(pix)
#         pix.save('ffilter.png', 'png')
#         QMessageBox.about(self, "Saved", "Saved to file 'ffilter.png'")


# def main():
#     ff = FourierFilter(200)
#     ff.filter_signal()
#     app = QApplication(sys.argv)
#     app.setStyle('Fusion')
#     app.setStyleSheet("QLabel{font-size: 13pt;} QLineEdit{font-size: 13pt;} QPushButton{font-size: 13pt;}")
#     window = Gui(ff=ff)
#     window.show()
#     app.exec()


# if __name__ == '__main__':
#     main()
