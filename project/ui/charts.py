from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class HistoryChart(QWidget):
    def __init__(self):
        super().__init__()

        self.fig = Figure(figsize=(5, 3))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.ax.set_title("Histórico do Melhor Valor")
        self.ax.set_xlabel("Iteração")
        self.ax.set_ylabel("Melhor")

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)

        self.data = []

    def add_point(self, value):
        self.data.append(value)
        self.ax.clear()
        self.ax.plot(self.data)
        self.ax.set_title("Histórico do Melhor Valor")
        self.ax.set_xlabel("Iteração")
        self.ax.set_ylabel("Melhor")
        self.canvas.draw()

    def clear(self):
        self.data = []
        self.ax.clear()
        self.ax.set_title("Histórico do Melhor Valor")
        self.ax.set_xlabel("Iteração")
        self.ax.set_ylabel("Melhor")
        self.canvas.draw()

    def set_series(self, values):
        self.data = list(values)
        self.ax.clear()
        self.ax.plot(self.data)
        self.ax.set_title("Histórico do Melhor Valor")
        self.ax.set_xlabel("Iteração")
        self.ax.set_ylabel("Melhor")
        self.canvas.draw()
