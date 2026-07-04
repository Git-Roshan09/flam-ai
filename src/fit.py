import numpy as np
import pandas as pd
from scipy.optimize import least_squares

DATA = "data/xy_data.csv"

def load(path=DATA):
    df = pd.read_csv(path)
    return df["x"].to_numpy(float), df["y"].to_numpy(float)

def residuals(p, x, y):
    """Un-rotate/un-shift each point, then compare v to the base-curve law."""
    th, M, X = p
    ax, ay = x - X, y - 42.0
    c, s = np.cos(th), np.sin(th)
    u =  ax * c + ay * s          # recovered t
    v = -ax * s + ay * c
    return v - np.exp(M * u) * np.sin(0.3 * u)

def fit(x, y):
    lo = [0.0,            -0.05,   0.0]
    hi = [np.radians(50),  0.05, 100.0]
    best = None
    for th0 in np.arange(2, 50, 3):          # deg
        for X0 in np.arange(5, 100, 5):
            p0 = [np.radians(th0), 0.0, X0]
            sol = least_squares(
                residuals, p0, args=(x, y), method="trf",
                bounds=(lo, hi), x_scale=[0.1, 0.01, 10.0], max_nfev=2000,
            )
            cost = float(np.sum(sol.fun ** 2))
            if best is None or cost < best[0]:
                best = (cost, sol.x.copy())
    return best[1]

if __name__ == "__main__":
    x, y = load()
    th, M, X = fit(x, y)
    r = residuals((th, M, X), x, y)
    print(f"theta = {np.degrees(th):.6f} deg  ({th:.6f} rad)")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")
    print(f"RMSE  = {np.sqrt(np.mean(r**2)):.3e}")
