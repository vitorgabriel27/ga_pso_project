import numpy as np
from .base_optimizer import BaseOptimizer

class ParticleSwarm(BaseOptimizer):
    def __init__(
        self,
        fitness_func,
        num_particles,
        num_iterations,
        dimension,
        inertia,
        cognitive,
        social,
        lower_bound,
        upper_bound,
        rng=None
    ):
        self.fitness_func = fitness_func
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.dimension = dimension
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.rng = rng or np.random.default_rng()

        self.positions = self.rng.uniform(lower_bound, upper_bound, (num_particles, dimension))
        self.velocities = self.rng.uniform(-1, 1, (num_particles, dimension))
        self.personal_best = np.copy(self.positions)
        self.personal_best_fitness = self.fitness_func(self.personal_best)
        self.global_best = self.personal_best[np.argmax(self.personal_best_fitness)]

    def optimize(self, on_update=None):
        for iteration in range(self.num_iterations):
            fitness = self.fitness_func(self.positions)

            # Update personal best
            better = fitness > self.personal_best_fitness
            self.personal_best[better] = self.positions[better]
            self.personal_best_fitness[better] = fitness[better]

            # Update global best
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > self.fitness_func(np.array([self.global_best])):
                self.global_best = self.positions[best_idx]

            # Emit state to UI
            if on_update is not None:
                on_update({
                    "iteration": iteration,
                    "positions": self.positions.copy(),
                    "velocities": self.velocities.copy(),
                    "fitness": fitness.copy(),
                    "best_pos": self.global_best.copy(),
                    "best_fitness": float(fitness[best_idx])
                })

            # Random factors
            r1 = self.rng.random((self.num_particles, self.dimension))
            r2 = self.rng.random((self.num_particles, self.dimension))

            # Velocity update
            self.velocities = (
                self.inertia * self.velocities +
                self.cognitive * r1 * (self.personal_best - self.positions) +
                self.social * r2 * (self.global_best - self.positions)
            )

            # Position update
            self.positions += self.velocities

            # Bounds
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

        final_fitness = self.fitness_func(self.positions)
        best_idx = np.argmax(final_fitness)
        return self.positions[best_idx], final_fitness[best_idx]
