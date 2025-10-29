import numpy as np
import matplotlib.pyplot as plt

def plot_eval_curve(npz_path="logs/evaluations.npz", out_png="reward_curve.png"):
    data = np.load(npz_path)
    ts = data["timesteps"]
    mean_r = data["results"].mean(axis=1)
    std_r = data["results"].std(axis=1)

    plt.figure(figsize=(6,4))
    plt.plot(ts, mean_r, label="Mean reward (5 eps)")
    plt.fill_between(ts, mean_r-std_r, mean_r+std_r, alpha=0.2, label="±1 std")
    plt.xlabel("Timesteps")
    plt.ylabel("Mean reward")
    plt.title("SAC on Panda Pick-and-Place (Eval)")
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    print("Saved curve to", out_png)

if __name__ == "__main__":
    plot_eval_curve()
