from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np
from core.algorithms.genetic_algorithm import GeneticAlgorithm
from core.operators.selection import roulette_selection
from core.operators.crossover import one_point_crossover
from core.operators.mutation import uniform_mutation


class GAWorker(QObject):
    update_signal = pyqtSignal(dict)
    finished_signal = pyqtSignal(dict)

    def __init__(
        self,
        fitness_func,
        population_size=200,
        generations=40,
        chromosome_length=2,
        mutation_rate=0.1,
        crossover_rate=0.7,
        rng_seed=42,
        selection_op=roulette_selection,
        crossover_op=one_point_crossover,
        mutation_op=uniform_mutation,
    ):
        super().__init__()
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.generations = generations
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.rng_seed = rng_seed
        self.selection_op = selection_op
        self.crossover_op = crossover_op
        self.mutation_op = mutation_op

    def run(self):
        ga = GeneticAlgorithm(
            fitness_func=self.fitness_func,
            population_size=self.population_size,
            num_generations=self.generations,
            crossover_rate=self.crossover_rate,
            mutation_rate=self.mutation_rate,
            chromosome_length=self.chromosome_length,
            selection_op=self.selection_op,
            crossover_op=self.crossover_op,
            mutation_op=self.mutation_op,
            rng=np.random.default_rng(self.rng_seed),
        )

        history_best = []
        history_best_pos = []
        population_history = []

        def on_update(payload):
            # Buffer apenas
            history_best.append(float(payload["best"]))
            history_best_pos.append(payload["best_individual"].copy())
            population_history.append(payload["population"].copy())

        best_individual, best_fitness = ga.optimize(on_update=on_update)

        self.finished_signal.emit({
            "best": float(best_fitness),
            "best_pos": best_individual.tolist(),
            "history_best": history_best,
            "history_best_pos": [p.tolist() for p in history_best_pos],
            "population_history": [p.tolist() for p in population_history]
        })
