# CLAUDE.md

Project context and build guide for the **R&D / AI parametric curve-fitting assignment**.
This file is the single source of truth: it contains the problem, the full methodology,
the reference implementation, the repo layout, the run commands, and the verified answer.
An agent (or a human) should be able to reproduce the entire solution from this file alone.

---

## 1. Goal

Given a CSV of unordered `(x, y)` points that lie on a parametric curve, recover the three
unknown constants **θ, M, X** that were baked into the curve's equations.

**Verified answer (already solved — use for regression testing):**

| Unknown | Value | Radians / exact | Allowed range | In range |
|---------|-------|-----------------|---------------|----------|
| θ | **30°** | 0.5236 rad (π/6) | 0°–50° | ✓ |
| M | **0.03** | — | −0.05–0.05 | ✓ |
| X | **55** | — | 0–100 | ✓ |

Fit quality on the provided data: mean L1 error per point ≈ 2×10⁻⁵, max abs error ≈ 3.5×10⁻⁵
(i.e. at the CSV's rounding-precision floor — effectively an exact recovery).

**Desmos / LaTeX submission string (the required deliverable):**

```
\left(t*\cos(0.5236)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,42+t*\sin(0.5236)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```

Domain: `6 ≤ t ≤ 60`.

---

## 2. The problem

Parametric equations of the curve (θ, M, X unknown; everything else fixed):

```
x(t) = t·cos(θ) − e^(M·|t|)·sin(0.3t)·sin(θ) + X
y(t) = 42 + t·sin(θ) + e^(M·|t|)·sin(0.3t)·cos(θ)
```

Constraints:
- `0° < θ < 50°`
- `−0.05 < M < 0.05`
- `0 < X < 100`
- parameter `6 < t < 60`  (so `t > 0` ⇒ `|t| = t`)

Data: `data/xy_data.csv`, columns `x,y`, ~1500 rows, **shuffled** (no `t` column, no ordering).

Scoring (from the brief):
1. L1 distance between uniformly sampled points on expected vs. predicted curve — max 100
2. Explanation of process/steps — max 80
3. Code / GitHub repo — max 50
4. Partial credit for reasoning even if the numbers are off.

---

## 3. Methodology (the core idea)

### 3.1 Recognize the structure: rotation + translation

Define a **base curve** in its own frame:

```
u(t) = t
v(t) = e^(M·t)·sin(0.3t)          # a sine wave with exponentially scaled amplitude
```

Then the given equations are exactly a **2-D rotation by θ** of `(u, v)`, followed by a
**translation** by `(X, 42)`:

```
[ x − X  ]   [ cos θ   −sin θ ] [ u ]
[ y − 42 ] = [ sin θ    cos θ ] [ v ]
```

So the "true shape" is just a wavy line; θ tilts it, `(X, 42)` slides it. The frequency `0.3`
and the y-offset `42` are known, which is why only θ, M, X remain free.

### 3.2 The key trick: invert the transform, no `t` needed

A rotation is invertible. For **any** candidate `(θ, X)`, un-rotate and un-shift every data
point back into base-frame coordinates:

```
u_i =  (x_i − X)·cos θ + (y_i − 42)·sin θ      # this recovers t_i
v_i = −(x_i − X)·sin θ + (y_i − 42)·cos θ
```

At the **correct** parameters, every recovered point must satisfy the base-curve law:

```
v_i = e^(M·u_i)·sin(0.3·u_i)
```

This is what makes the shuffled/unlabeled data a non-issue — we never need to know which `t`
produced which point.

### 3.3 Objective function

Minimize the sum of squared residuals over the three unknowns:

```
minimize_(θ, M, X)   Σ_i [ v_i − e^(M·u_i)·sin(0.3·u_i) ]²
```

where `u_i, v_i` are the inverse-transformed coordinates above.

### 3.4 Solver: multi-start bounded least squares

The objective is non-convex in `(θ, X)`, so a single local solve can land in a wrong basin.
Use a **grid of starting points** over θ ∈ [0°, 50°] and X ∈ [0, 100] (M starts at 0), run
bounded Levenberg–Marquardt / trust-region least squares
(`scipy.optimize.least_squares`, `method='trf'`) from each, and keep the lowest-cost result.
`x_scale` is set per-parameter because θ (~O(1) rad), M (~O(0.01)), and X (~O(10)) live on
very different scales.

### 3.5 Why the answer is trustworthy (identifiability + checks)

- 1500 points vs. 3 parameters, and the curve shape is distinctive → a unique global minimum.
- **Residual check:** RMSE collapses to ~1e-6 (the CSV rounding floor), not a vague minimum.
- **Range check:** recovered `u = t` values land in ~[6.0, 60.0], matching the stated domain
  — independent confirmation that the un-rotation is correct.
- **Round numbers:** θ=30°, M=0.03, X=55 are exact, i.e. the true generating constants.
- **Forward L1 check:** re-simulate the curve at the recovered `t` and compare to the raw
  data with the assignment's own L1 metric → ~2e-5 (see `verify.py`).

---

## 4. Repository layout

```
.
├── CLAUDE.md              # this file
├── README.md             # human-facing writeup (Section 8) + Desmos link + answer
├── requirements.txt      # numpy, scipy, pandas, matplotlib
├── data/
│   └── xy_data.csv       # provided points (x,y)
├── src/
│   ├── fit.py            # recovers θ, M, X  (Section 5)
│   └── verify.py         # L1 metric + forward-model check (Section 6)
└── figures/
    └── fit_overlay.png   # data vs. fitted curve (generated by verify.py)
```

---

## 5. Reference implementation — `src/fit.py`

```python
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
```

---

## 6. Verification — `src/verify.py`

```python
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Fitted answer (from fit.py)
TH, M, X = np.radians(30.0), 0.03, 55.0

def model(t, th=TH, M=M, X=X):
    u = t
    v = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = u*np.cos(th) - v*np.sin(th) + X
    y = u*np.sin(th) + v*np.cos(th) + 42.0
    return x, y

if __name__ == "__main__":
    df = pd.read_csv("data/xy_data.csv")
    x, y = df["x"].to_numpy(float), df["y"].to_numpy(float)

    # recover t per point, forward-simulate, measure L1
    ax, ay = x - X, y - 42.0
    c, s = np.cos(TH), np.sin(TH)
    t = ax*c + ay*s
    xp, yp = model(t)
    L1 = np.mean(np.abs(xp - x) + np.abs(yp - y))
    print(f"recovered t range: [{t.min():.3f}, {t.max():.3f}]")
    print(f"mean L1 per point: {L1:.3e}")

    # overlay plot
    tt = np.linspace(6, 60, 2000)
    cx, cy = model(tt)
    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=4, alpha=0.3, label="data")
    plt.plot(cx, cy, lw=1.5, color="crimson", label="fitted curve")
    plt.gca().set_aspect("equal"); plt.legend(); plt.tight_layout()
    plt.savefig("figures/fit_overlay.png", dpi=140)
    print("wrote figures/fit_overlay.png")
```

---

## 7. Setup & run

```bash
pip install -r requirements.txt          # numpy scipy pandas matplotlib
python src/fit.py                        # prints θ, M, X and RMSE
python src/verify.py                     # prints L1, t-range; writes figures/fit_overlay.png
```

Expected `fit.py` output: `theta ≈ 30`, `M ≈ 0.03`, `X ≈ 55`, `RMSE ≈ 3e-6`.

`requirements.txt`:
```
numpy
scipy
pandas
matplotlib
```

---

## 8. README.md contents (human-facing)

The README should contain, in order:
1. One-line problem statement and the final answer table (θ=30°, M=0.03, X=55).
2. The Desmos submission string (Section 1) and the calculator link:
   `https://www.desmos.com/calculator/zkfrnxiudo`
3. The methodology narrative from Section 3 — lead with the rotation/translation insight,
   then the inverse-transform trick, the objective, and the solver.
4. The `figures/fit_overlay.png` image showing data vs. fitted curve.
5. How to reproduce (Section 7 commands).

Keep it in the writer's own words; do not paste the assignment brief verbatim.

---

## 9. Assessment-criteria mapping

| Criterion (max) | Where it's addressed |
|---|---|
| L1 distance (100) | Near-zero L1 (~2e-5); reported by `verify.py`; answer is the exact generating values |
| Explanation (80) | Section 3 methodology, reproduced in README |
| Code / repo (50) | `src/fit.py`, `src/verify.py`, this CLAUDE.md, README, reproducible commands |
| "Additional code/maths to extract vars" (bonus) | The inverse-rotation derivation + multi-start least-squares pipeline |

---

## 10. Conventions & notes

- `t > 0` throughout the domain, so `|t| = t`; the code uses `abs(t)` anyway for safety.
- The `42` and the `0.3` frequency are fixed constants — never fit them.
- Angles: solver works in **radians**; report/submit both degrees and radians.
- Determinism: the multi-start grid is fixed, so results are reproducible (no random seed needed).
- If a future dataset spanned negative `t`, replace `np.exp(M*u)` with `np.exp(M*np.abs(u))`
  in `residuals` (the `|t|` term); for this assignment it is irrelevant since `t ∈ (6, 60)`.
- Sanity gate for "did it converge to the right basin": residual RMSE should be ≲1e-4 **and**
  recovered `t` range should be ≈[6, 60]. If not, widen the multi-start grid.
```
