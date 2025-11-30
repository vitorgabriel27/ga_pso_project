import numpy as np

def one_point_crossover(pop, rate, rng):
    offspring = np.copy(pop)
    for i in range(0, len(pop), 2):
        if rng.random() < rate:
            point = rng.integers(1, pop.shape[1])
            offspring[i, point:], offspring[i+1, point:] = (
                offspring[i+1, point:].copy(),
                offspring[i, point:].copy(),
            )
    return offspring
