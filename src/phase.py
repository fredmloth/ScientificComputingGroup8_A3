import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def leapfrog_oscillator(k=1.0, m=1.0, time=10.0, timesteps=1000, driving_amplitude=0.0, driving_frequency=0.0, x0=1.0, v0=0.0):

    dt = time/timesteps

    t_values = np.linspace(0, time, timesteps+1)
    x_values = np.zeros(timesteps+1)
    v_values = np.zeros(timesteps+1)
    

    x_values[0] = x0 #initial condition for x anything except 0 otherwise system will be in equilibrum 
    
    # start v at t = -dt/2 (half step back)
    # if we have an approximate for v0, we can use v(-dt/2) = v0 - F(x0)/(2m) * dt

    v_half_back = v0 - 0.5 * dt * (-k * x0 / m + driving_amplitude * np.sin(driving_frequency * 0) / m)
    

    for i in range(timesteps):
        # full step update for x
        x_values[i+1] = x_values[i] + v_half_back * dt

        t = t_values[i+1]

        spring_force = -k * x_values[i+1]
        driving_force = driving_amplitude * np.sin(driving_frequency * t)
        total_force = spring_force + driving_force
        
        # full step velocity update
        v_half_forward = v_half_back + dt * (total_force / m)
        
        # at n take average of half back and half forward 
        v_values[i] = (v_half_back + v_half_forward) / 2 if i > 0 else v0

        v_half_back = v_half_forward
    
    v_values[-1] = v_half_forward

    potential_energy = 0.5 * k * x_values**2
    kinetic_energy = 0.5 * m * v_values**2
    energy = potential_energy + kinetic_energy
    
    return t_values, x_values, v_values, energy


def analyze_resonance(k=1.0, m=1.0, driving_amplitude=0.2, time=10.0, timesteps=1000):
    """
    Function to analyze the system behavior at different driving frequencies around resonance
    """
   
    natural_freq = np.sqrt(k/m)

    freq_ratios = [0.5, 1.0, 1.5]

    frequencies = [ratio * natural_freq for ratio in freq_ratios]
    
    results = {}
    for freq in frequencies:
        t, x, v, e = leapfrog_oscillator(k=k, m=m, time=time, timesteps=timesteps,driving_amplitude=driving_amplitude,driving_frequency=freq)
        results[f"{freq/natural_freq:.1f}"] = (t, x, v, e)
    
    return results, natural_freq


def visualize_phase_plots(resonance_results):
    n_freqs = len(resonance_results)
    fig, axes = plt.subplots(1, n_freqs, figsize=(15, 4))
    
    if n_freqs == 1:
        axes = [axes]
    
    for i, (freq_ratio, (_, x, v, _)) in enumerate(resonance_results.items()):
  
        axes[i].plot(x, v)
        axes[i].set_xlabel("Position", fontsize= 18)
        axes[i].set_ylabel("Velocity", fontsize= 18)
        axes[i].set_title(f"Phase Plot (ω/ω₀ = {freq_ratio})", fontsize = 20)
        axes[i].tick_params(axis='both', which='major', labelsize=16)
    fig.tight_layout()
    fig.savefig("driven_oscillator_phase_plots.pdf")
    
    return fig


def visualize_position_time_plots(resonance_results):

    n_freqs = len(resonance_results)
    fig, axes = plt.subplots(1, n_freqs, figsize=(15, 4))

    if n_freqs == 1:
        axes = [axes]
    
    for i, (freq_ratio, (t, x, v, e)) in enumerate(resonance_results.items()):

        axes[i].plot(t, x)
        axes[i].set_xlabel("Time", fontsize = 18)
        axes[i].set_ylabel("Position", fontsize = 18)
        axes[i].set_title(f"Position vs Time (ω/ω₀ = {freq_ratio})", fontsize = 20)
        axes[i].tick_params(axis='both', which='major', labelsize=16)
    
    fig.tight_layout()
    fig.savefig("driven_oscillator_position_plots.pdf")
    
    return fig
