import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class FunctionPlot(QWidget):
    def __init__(self, fitness_func, lower=-100, upper=100, resolution=100):
        super().__init__()

        self.fitness_func = fitness_func
        self.lower = lower
        self.upper = upper
        self.resolution = resolution

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Matplotlib Figure
        self.fig = Figure(figsize=(5, 5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111)

        # Precomputar grid
        self._gen_grid()
        self._plot_function()

        self.best_point_plot = None
        self.points_scatter = None

    def _gen_grid(self):
        x = np.linspace(self.lower, self.upper, self.resolution)
        y = np.linspace(self.lower, self.upper, self.resolution)
        self.X, self.Y = np.meshgrid(x, y)

        pos = np.vstack([self.X.ravel(), self.Y.ravel()]).T
        Z = self.fitness_func(pos)
        self.Z = Z.reshape(self.X.shape)

    def _plot_function(self):
        self.ax.clear()
        self.ax.set_title("Função de Fitness")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.map = self.ax.imshow(
            self.Z,
            extent=[self.lower, self.upper, self.lower, self.upper],
            origin="lower",
            cmap="viridis",
            alpha=0.85
        )

        self.fig.colorbar(self.map, ax=self.ax)
        self.canvas.draw()

    def update_best_point(self, pos):
        x, y = pos

        if self.best_point_plot is None:
            self.best_point_plot = self.ax.plot(x, y, 'ro', markersize=8)[0]
        else:
            self.best_point_plot.set_data([x], [y])

        self.canvas.draw()

    def update_points(self, points):
        """Atualiza nuvem de pontos (partículas ou indivíduos)."""
        if len(points) == 0:
            if self.points_scatter is not None:
                self.points_scatter.remove()
                self.points_scatter = None
            self.canvas.draw()
            return
        pts = np.asarray(points)
        if self.points_scatter is None:
            self.points_scatter = self.ax.scatter(pts[:, 0], pts[:, 1], c='white', s=20, edgecolors='black', linewidths=0.5)
        else:
            self.points_scatter.set_offsets(pts[:, 0:2])
        self.canvas.draw()

    def clear_points(self):
        if self.points_scatter is not None:
            self.points_scatter.remove()
            self.points_scatter = None
        if self.best_point_plot is not None:
            self.best_point_plot.remove()
            self.best_point_plot = None
        self.canvas.draw()
