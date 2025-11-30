from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np
from core.algorithms.particle_swarm import ParticleSwarm


class PSOWorker(QObject):
    update_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(dict)

    def __init__(
        self,
        fitness_func,
        num_particles=40,
        num_iterations=200,
        dimension=2,
        inertia=0.7,
        cognitive=1.5,
        social=1.5,
        lower_bound=-100,
        upper_bound=100,
        rng_seed=42,
    ):
        super().__init__()
        self.fitness_func = fitness_func
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.dimension = dimension
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.rng_seed = rng_seed

    def run(self):
        optimizer = ParticleSwarm(
            fitness_func=self.fitness_func,
            num_particles=self.num_particles,
            num_iterations=self.num_iterations,
            dimension=self.dimension,
            inertia=self.inertia,
            cognitive=self.cognitive,
            social=self.social,
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            rng=np.random.default_rng(self.rng_seed)
        )

        history_best = []
        history_best_pos = []
        positions_history = []

        def callback(state):
            # buffer somente
            history_best.append(float(state["best_fitness"]))
            history_best_pos.append(state["best_pos"].copy())
            positions_history.append(state["positions"].copy())

        best_pos, best_val = optimizer.optimize(on_update=callback)

        self.finished_signal.emit({
            "best": float(best_val),
            "best_pos": best_pos.tolist(),
            "history_best": history_best,
            "history_best_pos": [p.tolist() for p in history_best_pos],
            "positions_history": [p.tolist() for p in positions_history]
        })
