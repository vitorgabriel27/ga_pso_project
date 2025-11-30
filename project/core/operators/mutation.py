import numpy as np

def uniform_mutation(pop, rate, rng):
    mask = rng.random(pop.shape) < rate
    pop[mask] = rng.random(np.count_nonzero(mask))
    return pop
