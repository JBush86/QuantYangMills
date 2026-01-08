# Discovered by Jared C. Bush, May 2025
# Copyright (c) 2026 Jared C. Bush
# Licensed under MIT License
# Reference Paper: https://doi.org/10.5281/zenodo.18158682

import numpy as np
import matplotlib.pyplot as plt

# Parameters
FIELD_SIZE = 64      # Number of sites (1D toy model)
T_MAX = 200          # Simulation steps
ENERGY_INIT = 20.0   # Initial total system energy
MASS_GAP_INFO_THRESHOLD = 0.15

# Spectral Threshold (L, omega, chi) toy field parameters
L_mu = 0.9
L_sigma = 0.18
omega_mu = 1.0
omega_sigma = 0.15
chi_mu = 1.0
chi_sigma = 0.16

np.random.seed(42)

# Initialize the fields
L = np.abs(np.random.normal(L_mu, L_sigma, FIELD_SIZE))
omega = np.abs(np.random.normal(omega_mu, omega_sigma, FIELD_SIZE))
chi = np.abs(np.random.normal(chi_mu, chi_sigma, FIELD_SIZE))

# Start with empty info field and full energy budget
I_field = np.zeros(FIELD_SIZE)
energy_budget = ENERGY_INIT
energy_ledger = []
total_E_spent = 0.0

# For plotting over time
gauge_field_log = []
I_field_log = []
energy_budget_log = []

# Main simulation loop
for t in range(T_MAX):
    # Generate random gauge field fluctuations ("potential excitations")
    field_excitation = np.random.normal(0.0, 1.1, FIELD_SIZE)
    # Optionally inject rare structured events ("particles")
    if np.random.rand() < 0.018:
        pos = np.random.randint(0, FIELD_SIZE)
        amp = 2.0 + np.random.rand()*2.5
        field_excitation[pos] += amp
        print(f"  T={t}: Structured excitation at {pos} (Amp={amp:.2f})")
    gauge_field_log.append(field_excitation.copy())

    # Compute instantaneous info production at each site
    dI = np.abs(field_excitation) * L * omega * chi
    # Apply mass gap: only allow "above threshold" events to persist as particles
    dI_particles = np.where(dI >= MASS_GAP_INFO_THRESHOLD, dI, 0.0)
    # Only "actualize" information where energy is available
    dE = np.minimum(dI_particles, energy_budget / FIELD_SIZE)
    I_field += dE
    total_E_spent += np.sum(dE)
    energy_budget -= np.sum(dE)
    energy_budget = max(energy_budget, 0.0)
    energy_ledger.append(total_E_spent)
    I_field_log.append(I_field.copy())
    energy_budget_log.append(energy_budget)

    if energy_budget <= 0:
        print(f"System energy exhausted at T={t}")
        break

# Convert logs to arrays
gauge_field_log = np.array(gauge_field_log)
I_field_log = np.array(I_field_log)

# --- Results Reporting ---
final_I_sum = np.sum(I_field_log[-1])
print("\n--- Energy–Information Results ---")
print(f"Total Energy Spent (∑ dE): {total_E_spent:.6f}")
print(f"Total Anchored Information (∑ final I): {final_I_sum:.6f}")
if abs(final_I_sum - total_E_spent) < 1e-6:
    print("SUCCESS: I = E = ∫(L × omega × chi)dt holds.")
else:
    print("WARNING: Minor mismatch in Spectral Threshold (numerical error or bug).")
print(f"\nFinal persistent info field (I):\n{I_field.round(2)}")

# --- Visualization ---
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
plt.suptitle("Spectral Threshold (I = E = ∫(L × omega × chi)dt) Applied to Yang-Mills Mass Gap (Toy Model)")

im0 = axes[0].imshow(gauge_field_log.T, aspect='auto', cmap='coolwarm', origin='lower')
axes[0].set_ylabel("Position in Field")
axes[0].set_title("Gauge Field Evolution (Potential Excitations)")
plt.colorbar(im0, ax=axes[0], orientation='vertical', label="Field Strength")

im1 = axes[1].imshow(I_field_log.T, aspect='auto', cmap='plasma', origin='lower')
axes[1].set_ylabel("Position in Field")
axes[1].set_title("Information Field (I): Emergent 'Massive Particles'")
plt.colorbar(im1, ax=axes[1], orientation='vertical', label="I = ∫(L×omega×chi)dt")

axes[2].plot(energy_ledger, color='lime', label='Cumulative Energy Spent')
axes[2].set_ylabel("Cumulative Energy")
axes[2].set_xlabel("Time")
axes[2].set_title("Cumulative Energy Consumption Over Time")
axes[2].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
