import numpy as np
import matplotlib.pyplot as plt


def harmonic_oscillator(k=1, m=1, time=1, timesteps=1000):
    dt = time/timesteps
    x = 1
    v = (-k/m) * x * (dt/2)

    Lx, Lv = [x], [v]

    for _ in range(timesteps):
        x += v * dt
        v += (-k/m) * x * dt

        Lx.append(x)
        Lv.append(v)

    return Lx, Lv


def visualise_harmonic_oscillator(Lt, dict_x, dict_v):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # Flatten the axes for correct image rendering
    axes = axes.flatten()
    
    for k in dict_x.keys():
        axes[0].plot(Lt, dict_x[k], label=f"k={k}")
        axes[1].plot(Lt, dict_v[k], label=f"k={k}")
    
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("displacement")
    axes[0].set_title(f"Displacement of 1D Harmonic Oscillator")
    axes[0].legend(loc="lower right")

    axes[1].set_xlabel("time")
    axes[1].set_ylabel("velocity")
    axes[1].set_title(f"Velocity of 1D Harmonic Oscillator")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(f"results/leapfrog_harmosc_{[k for k in dict_x.keys()]}.pdf")
