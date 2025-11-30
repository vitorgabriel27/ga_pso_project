import numpy as np
from PyQt6.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # necessário para 3D


class FunctionPlot3D(QWidget):
    def __init__(self, fitness_func, lower=-100, upper=100, resolution=60):
        super().__init__()

        self.fitness_func = fitness_func
        self.lower = lower
        self.upper = upper
        self.resolution = resolution

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Figura matplotlib
        self.fig = Figure(figsize=(6, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas)

        self.ax = self.fig.add_subplot(111, projection="3d")

        self._gen_grid()
        self._plot_surface()

        self.best_point_plot = None
        self.points_scatter = None

    def _gen_grid(self):
        x = np.linspace(self.lower, self.upper, self.resolution)
        y = np.linspace(self.lower, self.upper, self.resolution)
        self.X, self.Y = np.meshgrid(x, y)

        pos = np.vstack([self.X.ravel(), self.Y.ravel()]).T
        Z = self.fitness_func(pos)
        self.Z = Z.reshape(self.X.shape)

    def _plot_surface(self):
        self.ax.clear()
        self.ax.set_title("Função (3D Surface)")

        self.ax.plot_surface(
            self.X, self.Y, self.Z,
            cmap="viridis",
            edgecolor="none",
            alpha=0.9
        )

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("f(x, y)")

        self.canvas.draw()

    def update_best_point(self, pos):
        x, y = pos
        z = self.fitness_func(np.array([[x, y]]))[0]

        if self.best_point_plot is None:
            self.best_point_plot = self.ax.scatter(
                x, y, z,
                color="red",
                s=50
            )
        else:
            self.best_point_plot.remove()
            self.best_point_plot = self.ax.scatter(
                x, y, z,
                color="red",
                s=50
            )

        self.canvas.draw()

    def update_points(self, points):
        """Atualiza nuvem 3D das posições atuais."""
        if len(points) == 0:
            if self.points_scatter is not None:
                self.points_scatter.remove()
                self.points_scatter = None
            self.canvas.draw()
            return
        pts = np.asarray(points)
        z = self.fitness_func(pts)
        # Matplotlib 3D scatter doesn't support direct coordinate array update; recreate
        if self.points_scatter is not None:
            self.points_scatter.remove()
        self.points_scatter = self.ax.scatter(pts[:, 0], pts[:, 1], z, c='white', s=15, depthshade=True, edgecolors='black', linewidths=0.3)
        self.canvas.draw()

    def clear_points(self):
        if self.points_scatter is not None:
            self.points_scatter.remove()
            self.points_scatter = None
        if self.best_point_plot is not None:
            self.best_point_plot.remove()
            self.best_point_plot = None
        self.canvas.draw()
