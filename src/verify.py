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
