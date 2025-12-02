import numpy as np
from .base_optimizer import BaseOptimizer


class GeneticAlgorithm(BaseOptimizer):
    def __init__(
        self,
        fitness_func,
        population_size,
        num_generations,
        crossover_rate,
        mutation_rate,
        chromosome_length,
        selection_op,
        crossover_op,
        mutation_op,
        elitism_count: int = 2,
        rng=None,
    ):
        self.fitness_func = fitness_func
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.chromosome_length = chromosome_length
        self.selection_op = selection_op
        self.crossover_op = crossover_op
        self.mutation_op = mutation_op
        self.elitism_count = max(0, int(elitism_count))
        self.rng = rng or np.random.default_rng()

        # população inicial entre 0 e 1
        self.population = self.rng.random((population_size, chromosome_length))

    def optimize(self, on_update=None):
        for gen in range(self.num_generations):
            fitness = self.fitness_func(self.population)

            if on_update is not None:
                on_update({
                    "generation": gen,
                    "population": self.population.copy(),
                    "fitness": fitness.copy(),
                    "best": float(np.max(fitness)),
                    "best_individual": self.population[np.argmax(fitness)].copy()
                })

            # Seleção
            selected = self.selection_op(self.population, fitness, self.rng)

            # Crossover
            offspring = self.crossover_op(selected, self.crossover_rate, self.rng)

            # Mutação
            mutated = self.mutation_op(offspring, self.mutation_rate, self.rng)

            # Elitismo
            if self.elitism_count > 0:
                elite_idx = np.argsort(fitness)[-self.elitism_count:]
                elites = self.population[elite_idx]
                new_fit = self.fitness_func(mutated)
                worst_idx = np.argsort(new_fit)[:self.elitism_count]
                mutated[worst_idx] = elites

            self.population = mutated

        # Resultado final
        final_fitness = self.fitness_func(self.population)
        best_idx = np.argmax(final_fitness)

        return self.population[best_idx], final_fitness[best_idx]