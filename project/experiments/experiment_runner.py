def run_experiment(optimizer):
    best_solution, best_fitness = optimizer.optimize()
    return {
        "solution": best_solution,
        "fitness": best_fitness
    }
