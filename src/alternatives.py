"""Alternative fitting approaches, benchmarked against the primary method in fit.py.

Three families from the curve-fitting / registration literature:

1. Differential evolution  — population-based global optimizer, no starting grid,
   no gradients (scipy.optimize.differential_evolution).
2. Basin hopping           — random perturbation + local minimization, accepts/rejects
   basins Metropolis-style (scipy.optimize.basinhopping).
3. Geometric (ICP-style) point-distance minimization — no inverse transform at all:
   densely sample the candidate curve, match every data point to its nearest curve
   point (KD-tree), minimize the summed squared geometric distance. This is the
   correspondence-based approach used when the model frame cannot be recovered
   in closed form (PDM/SDM, iterative closest point).

All three optimize the same 3 unknowns (theta, M, X) under the assignment bounds.
Run:  python src/alternatives.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.optimize import basinhopping, differential_evolution, least_squares
from scipy.spatial import cKDTree

from fit import load, residuals

BOUNDS = [(0.0, np.radians(50)), (-0.05, 0.05), (0.0, 100.0)]


def sse(p, x, y):
    """Scalar objective for the global optimizers: same base-frame residual as fit.py."""
    return float(np.sum(residuals(p, x, y) ** 2))


def fit_differential_evolution(x, y, seed=0):
    res = differential_evolution(
        sse, BOUNDS, args=(x, y), seed=seed, tol=1e-12, polish=True,
    )
    return res.x, res.fun


def fit_basinhopping(x, y, seed=0):
    lo, hi = zip(*BOUNDS)
    p0 = [np.radians(25.0), 0.0, 50.0]
    res = basinhopping(
        sse, p0, niter=100, seed=seed,
        minimizer_kwargs={
            "args": (x, y), "method": "L-BFGS-B", "bounds": BOUNDS,
        },
        stepsize=0.5,
    )
    p = np.clip(res.x, lo, hi)
    return p, sse(p, x, y)


def curve_samples(p, n=6000):
    """Densely sample the forward model over the stated domain t in (6, 60)."""
    th, M, X = p
    t = np.linspace(6.0, 60.0, n)
    v = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    cx = t * np.cos(th) - v * np.sin(th) + X
    cy = t * np.sin(th) + v * np.cos(th) + 42.0
    return np.column_stack([cx, cy])


def geometric_residuals(p, pts):
    """Distance from each data point to its nearest sampled curve point (ICP-style)."""
    tree = cKDTree(curve_samples(p))
    d, _ = tree.query(pts, k=1)
    return d


def fit_geometric(x, y):
    pts = np.column_stack([x, y])
    lo, hi = zip(*BOUNDS)
    best = None
    for th0 in np.arange(5, 50, 9):           # deg — coarser grid: each eval is costlier
        for X0 in np.arange(10, 100, 15):
            p0 = [np.radians(th0), 0.0, X0]
            sol = least_squares(
                geometric_residuals, p0, args=(pts,), method="trf",
                bounds=(lo, hi), x_scale=[0.1, 0.01, 10.0],
                diff_step=1e-4, max_nfev=200,
            )
            cost = float(np.sum(sol.fun ** 2))
            if best is None or cost < best[0]:
                best = (cost, sol.x.copy())
    return best[1], best[0]


if __name__ == "__main__":
    x, y = load()
    methods = [
        ("differential evolution", lambda: fit_differential_evolution(x, y)),
        ("basin hopping", lambda: fit_basinhopping(x, y)),
        ("geometric / ICP-style", lambda: fit_geometric(x, y)),
    ]
    print(f"{'method':<24} {'theta (deg)':>12} {'M':>10} {'X':>10} "
          f"{'base-frame RMSE':>16} {'time (s)':>9}")
    for name, run in methods:
        t0 = time.perf_counter()
        p, _ = run()
        dt = time.perf_counter() - t0
        rmse = np.sqrt(np.mean(residuals(p, x, y) ** 2))
        print(f"{name:<24} {np.degrees(p[0]):>12.6f} {p[1]:>10.6f} {p[2]:>10.4f} "
              f"{rmse:>16.3e} {dt:>9.2f}")
