# Discovered by Jared C. Bush, May 2025
# Copyright (c) 2026 Jared C. Bush
# Licensed under MIT License
# Reference Paper: https://doi.org/10.5281/zenodo.18158682
# A toy model demonstrating how the Spectral Threshold
# I = ∫(L * omega * chi)dt, aligns with the principles of the
# Yang-Mills Existence and Mass Gap problem.

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Simulation Configuration ---
FIELD_SIZE = 128         # Number of points in our 1D gauge field
N_STEPS = 200           # Number of time steps
DECAY_I = 0.99          # How quickly particle information "forgets" if not sustained
MASS_GAP_INFO_THRESHOLD = 0.02 # Min (L*omega*chi) to be considered "above mass gap"

# Field evolution parameters
DIFFUSION = 0.05
EXCITATION_PROB = 0.02   # Probability of a new spontaneous excitation
STRUCTURED_EXCITATION_PROB = 0.02 # Probability of excitation

# --- Initial State of the Gauge Field ---
np.random.seed(2077)
gauge_field = np.random.randn(FIELD_SIZE) * 0.1 # Start with some low-level noise

# --- Logs for Visualization ---
I_field_log = []        # Log of the information field over time
gauge_field_log = []    # Log of the gauge field over time

# Information field (cumulative I, representing stable particles)
I_field = np.zeros(FIELD_SIZE)

print("Simulating 1D 'gauge field' to observe particle formation via Spectral Threshold and Mass Gap...")

# --- 2. Main Simulation Loop ---
for t in range(N_STEPS):
    gauge_field_prev_step = gauge_field.copy()
    gauge_field_log.append(gauge_field_prev_step)

    # A. Field Evolution (Simple: diffusion + new excitations)
    next_gauge_field = gauge_field.copy()
    # Diffusion
    for i in range(FIELD_SIZE):
        v_left = gauge_field[i-1] if i > 0 else gauge_field[i]
        v_right = gauge_field[i+1] if i < FIELD_SIZE-1 else gauge_field[i]
        laplacian = v_left + v_right - 2 * gauge_field[i]
        next_gauge_field[i] += DIFFUSION * laplacian
    
    # New random excitations (quantum noise)
    noise_excitations = (np.random.rand(FIELD_SIZE) < EXCITATION_PROB) * np.random.randn(FIELD_SIZE) * 0.5
    next_gauge_field += noise_excitations

    # Occasional "structured" or "energetic" excitations
    if np.random.rand() < STRUCTURED_EXCITATION_PROB:
        pos = np.random.randint(FIELD_SIZE)
        width = np.random.randint(2, 5)
        amplitude = np.random.rand() * 2.5 + 1.5 # Higher amplitude
        excitation_shape = amplitude * np.exp(-np.linspace(-2,2,width)**2)
        start = max(0, pos - width//2)
        end = min(FIELD_SIZE, start + width)
        actual_width = end - start
        if actual_width == width: # ensure full application
            next_gauge_field[start:end] += excitation_shape
            print(f"  T={t}: Structured excitation at {pos} (Amp={amplitude:.2f})")

    gauge_field = np.clip(next_gauge_field, -3, 3) # Keep field values bounded

    # B. Apply Spectral Threshold
    #    Action Density (L): Local measure of field non-triviality 
    L = np.abs(gauge_field - gauge_field_prev_step) + 0.1 * np.abs(gauge_field) # Change + current amplitude

    #    Topological Weight (omega): Weighting function that suppresses trivial vacuum fluctuations based on Chern-Simons number
    #    Higher for excitations that are wider or have larger, sustained amplitude.
    omega = np.zeros(FIELD_SIZE)
    for i in range(FIELD_SIZE):
        local_amplitude = np.abs(gauge_field[i])
        # Check width of excitation if it's a peak
        if local_amplitude > 0.5: # Arbitrary threshold for being a "peak"
            width = 0
            # Check left
            for j in range(i, -1, -1):
                if np.abs(gauge_field[j]) > local_amplitude * 0.3: width +=1
                else: break
            # Check right
            for j in range(i+1, FIELD_SIZE):
                if np.abs(gauge_field[j]) > local_amplitude * 0.3: width +=1
                else: break
            omega[i] = local_amplitude * (width / (FIELD_SIZE * 0.5)) + 0.1 # Topological weight depends on amplitude and relative width
        else:
            omega[i] = local_amplitude * (1 / FIELD_SIZE) # Lower weight for small fluctuations

    #    Spectral Coherence (chi): Measure of stability against the global action functional
    #    Higher if the excitation is smooth or part of a wave-like pattern.
    chi = np.zeros(FIELD_SIZE)
    for i in range(FIELD_SIZE):
        v_left = gauge_field[i-1] if i > 0 else gauge_field[i]
        v_center = gauge_field[i]
        v_right = gauge_field[i+1] if i < FIELD_SIZE-1 else gauge_field[i]
        # Simple measure: less "jaggedness" means more resonance
        jaggedness = np.abs(v_center - v_left) + np.abs(v_right - v_center) - np.abs(v_right - v_left)
        chi[i] = np.clip(1.0 - 0.25 * jaggedness, 0.2, 1.0) # More jagged = less stability

    # C. Calculate Potential Information and Apply Mass Gap
    potential_dI = L * omega * chi
    
    # Apply the Mass Gap: only excitations producing info above threshold contribute
    actual_dI = np.where(potential_dI >= MASS_GAP_INFO_THRESHOLD, potential_dI, 0.0)
    
    # Accumulate Information for "massive particles"
    I_field = DECAY_I * I_field + actual_dI
    I_field_log.append(I_field.copy())

# --- 3. Visualization ---
gauge_field_log_array = np.array(gauge_field_log)
I_field_log_array = np.array(I_field_log)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle("Spectral Threshold Applied to Yang-Mills & Mass Gap (Toy Model: 1D Gauge Field)", fontsize=16)

# Plot 1: Gauge Field Evolution
im_gauge = axes[0].imshow(gauge_field_log_array.T, aspect='auto', cmap='coolwarm', origin='lower',
                          extent=[0, N_STEPS, 0, FIELD_SIZE])
axes[0].set_ylabel('Position in Field')
axes[0].set_title('Gauge Field Evolution (Potential Excitations)')
fig.colorbar(im_gauge, ax=axes[0], label='Field Strength')

# Plot 2: Information Field (I) Evolution - "Massive Particles"
im_I = axes[1].imshow(I_field_log_array.T, aspect='auto', cmap='plasma', origin='lower',
                       extent=[0, N_STEPS, 0, FIELD_SIZE])
axes[1].set_xlabel('Time')
axes[1].set_ylabel('Position in Field')
axes[1].set_title('Information Field (I) - Emergent "Massive Particles" (Above Mass Gap)')
fig.colorbar(im_I, ax=axes[1], label='Information I (Particle Presence)')

# Highlight regions of high final Information (our "particles")
final_I = I_field_log_array[-1]
particle_threshold = np.percentile(final_I[final_I > 0], 80) if np.any(final_I > 0) else MASS_GAP_INFO_THRESHOLD
particle_locations = np.where(final_I >= particle_threshold)[0]

if particle_locations.size > 0:
    print(f"\n--- PERSISTENT 'PARTICLE' LOCATIONS (High Final I) ---")
    print(f"Locations: {particle_locations}")
    print(f"Max I in these regions: {final_I[particle_locations]}")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print("\n--- Yang-Mills & Spectral Threshold Insights (Conceptual) ---")
print("The Yang-Mills problem includes proving existence of solutions and a mass gap.")
print("This Spectral Threshold toy model suggests:")
print("1. The 'gauge field' constantly has quantum fluctuations (action density).")
print("2. Only excitations with sufficient topological weight (structure/energy) and spectral coherence (stability)")
print("   produce an initial L*omega*chi score that surpasses the 'MASS_GAP_INFO_THRESHOLD'.")
print("3. These 'above-gap' excitations accumulate lasting Information (I), becoming persistent 'particles'.")
print("4. Sub-gap 'noise' (L*omega*chi < threshold) does not accumulate I and dissipates.")
print("The mass gap, in this Spectral Threshold view, is an informational threshold required for stable particle existence.")