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

def blx_alpha_crossover(pop, rate, alpha, rng, lower_bounds=None, upper_bounds=None):
    """
    For each pair of parents (p1, p2) and each gene j, sample child values
    uniformly from [min_j - α·d_j, max_j + α·d_j], where d_j = |p1_j - p2_j|.

    Parameters
    - pop: ndarray (n_individuals, n_genes)
    - rate: float in [0,1], probability to apply crossover to a pair
    - alpha: float >= 0, extension factor beyond parents' range
    - rng: numpy Generator
    - lower_bounds/upper_bounds: optional arrays or scalars for clamping

    Returns
    - offspring: ndarray same shape as pop
    """
    offspring = np.copy(pop)
    n_genes = pop.shape[1]
    for i in range(0, len(pop), 2):
        if rng.random() < rate:
            p1 = offspring[i].copy()
            p2 = offspring[i+1].copy()
            lo = np.minimum(p1, p2)
            hi = np.maximum(p1, p2)
            d = hi - lo
            a_lo = lo - alpha * d
            a_hi = hi + alpha * d
            c1 = rng.uniform(a_lo, a_hi, size=n_genes)
            c2 = rng.uniform(a_lo, a_hi, size=n_genes)
            # Optional clamping
            if lower_bounds is not None:
                c1 = np.maximum(c1, lower_bounds)
                c2 = np.maximum(c2, lower_bounds)
            if upper_bounds is not None:
                c1 = np.minimum(c1, upper_bounds)
                c2 = np.minimum(c2, upper_bounds)
            offspring[i] = c1
            offspring[i+1] = c2
    return offspring
