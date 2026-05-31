#!/usr/bin/env python3
def optimize(self, iterations=100):
    """
    Optimizes the black-box function

    Args:
        iterations: maximum number of iterations

    Returns:
        X_opt, Y_opt
    """
    for i in range(iterations):
        X_next, _ = self.acquisition()

        if np.any(np.isclose(self.gp.X.flatten(), X_next[0])):
            break

        Y_next = self.f(X_next)

        self.gp.update(X_next, Y_next)

    if self.minimize:
        idx = np.argmin(self.gp.Y)
    else:
        idx = np.argmax(self.gp.Y)

    X_opt = self.gp.X[idx]
    Y_opt = self.gp.Y[idx]

    return X_opt, Y_opt
