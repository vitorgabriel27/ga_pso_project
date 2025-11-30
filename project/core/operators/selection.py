import numpy as np

def roulette_selection(population, fitness, rng):
    probs = fitness / fitness.sum()
    picks = rng.random(len(population))
    indexes = np.searchsorted(np.cumsum(probs), picks)
    return population[indexes]
