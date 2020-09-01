import tkinter

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, figure, master):
        self.canvas = FigureCanvasTkAgg(figure, master=master)
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        self.ax = master.ax
        self.plot()

    def plot(self):
        x = np.linspace(0, 2*np.pi, 100)
        self.ax.plot(x, np.sin(x))


class Gui(tkinter.Tk):
    def __init__(self, width=11, height=7, dpi=100):
        super().__init__()
        self.title('Fourier Filter')
        self.geometry(f'{width*dpi}x{height*dpi}')
        self.resizable(0, 0)

        fig, self.ax = plt.subplots(figsize=(width, height), dpi=dpi)
        self.canvas = Plotter(fig, self)


window = Gui()
tkinter.mainloop()
