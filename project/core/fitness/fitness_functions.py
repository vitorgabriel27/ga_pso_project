import numpy as np 

def objective_function(pos):
    x = pos[:, 0]
    y = pos[:, 1]
    z = -x * np.sin(np.sqrt(np.abs(x))) - y * np.sin(np.sqrt(np.abs(y)))
    x_norm = x / 250
    y_norm = y / 250
    r = 100 * (y_norm - x_norm * 2) ** 2 + (1 - x_norm) ** 2
    return r - z