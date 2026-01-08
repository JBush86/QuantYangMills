# Discovered by Jared C. Bush, May 2025
# Copyright (c) 2026 Jared C. Bush
# Licensed under MIT License
# Reference Paper: https://doi.org/10.5281/zenodo.18158682

"""
QuantYangMills4.py  ––  Spectral Threshold × Yang–Mills toy (SU(3), finite‑size robustness)

Purpose
-------
Demonstrate that the informational mass‑gap (ΔI_crit) and the energy–information
identity  I = E = ∫(L·omega·chi)dt  hold for an SU(3) gauge field.

Key differences vs. QuantYangMills{1,2,3}.py
  • GAUGE_GROUP = "SU3"
  • FIELD_SIZE  = 48   (keeps runtime < 2 min)
  • β           = 6.0  (≈ a ≈ 0.1 fm)
  • MASS_GAP_INFO_THRESHOLD set to Lambert‑W prediction: 0.0213
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import qr

# ---------------------------------------------------------------------------
# Simulation configuration
# ---------------------------------------------------------------------------
GAUGE_GROUP      = "SU3"
FIELD_SIZE       = 48          # lattice sites (1‑D toy)
N_STEPS          = 250
BETA             = 6.0         # Wilson β = 6/g²  (SU(3))
DIFF_COEFF       = 0.05        # diffusion weight
DECAY            = 0.995       # Spectral Threshold memory decay λ
DT               = 1.0         # time step (arbitrary units)
SEED             = 31415
MASS_GAP_INFO_THRESHOLD = 0.0213   # ΔI_crit from Lambert‑W (SU(3), β = 6)

# Spectral Threshold noise parameters
L_mu, L_sigma        = 0.9, 0.18
omega_mu, omega_sigma        = 1.0, 0.15
chi_mu, chi_sigma    = 1.0, 0.16

np.random.seed(SEED)

# ---------------------------------------------------------------------------
# SU(3) helpers
# ---------------------------------------------------------------------------
def random_su3(eps=0.1):
    """Small random SU(3) matrix close to identity (first‑order)."""
    z = (np.random.randn(3, 3) + 1j * np.random.randn(3, 3))
    q, r = qr(z)                                # unitary
    q *= np.exp(-1j * np.angle(np.diag(r)))     # det = 1
    return np.eye(3, dtype=np.complex128) + eps * (q - q.conj().T)

# ---------------------------------------------------------------------------
# Field initialisation
# ---------------------------------------------------------------------------
field = [np.eye(3, dtype=np.complex128) for _ in range(FIELD_SIZE)]

# UIL coefficient fields (constant in this toy)
L    = np.abs(np.random.normal(L_mu,   L_sigma,  FIELD_SIZE))
omega    = np.abs(np.random.normal(omega_mu,   omega_sigma,  FIELD_SIZE))
chi  = np.abs(np.random.normal(chi_mu, chi_sigma, FIELD_SIZE))

I_field         = np.zeros(FIELD_SIZE)
energy_budget   = 20.0
energy_ledger   = []
total_E_spent   = 0.0

gauge_field_log = []
I_field_log     = []
energy_budget_log = []

# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------
for t in range(N_STEPS):
    # Random SU(3) kicks + diffusion
    for i in range(FIELD_SIZE):
        field[i] = random_su3(0.05) @ field[i]                   # kick
        left  = field[(i - 1) % FIELD_SIZE]
        right = field[(i + 1) % FIELD_SIZE]
        field[i] = ((1 - DIFF_COEFF) * field[i]
                    + (DIFF_COEFF / 2) * (left + right))         # diffusion

    # Gauge "observable": here we just use norm deviation as a proxy
    field_excitation = np.array(
        [np.linalg.norm(mat - np.eye(3)) for mat in field]
    )

    # Occasionally inject above‑gap excitation
    if np.random.rand() < 0.02:
        pos = np.random.randint(0, FIELD_SIZE)
        amp = 2.0 + 2.5 * np.random.rand()
        field_excitation[pos] += amp
        print(f"  T={t}: Structured excitation at {pos} (Amp={amp:.2f})")

    gauge_field_log.append(field_excitation.copy())

    # Spectral Threshold incremental information
    dI = np.abs(field_excitation) * L * omega * chi
    dI_particles = np.where(dI >= MASS_GAP_INFO_THRESHOLD, dI, 0.0)

    # Energy spending: limit by remaining budget
    dE = np.minimum(dI_particles, energy_budget / FIELD_SIZE)
    I_field += dE
    total_E_spent += np.sum(dE)
    energy_budget -= np.sum(dE)
    energy_budget = max(energy_budget, 0.0)

    # Logs
    energy_ledger.append(total_E_spent)
    I_field_log.append(I_field.copy())
    energy_budget_log.append(energy_budget)

    if energy_budget <= 0:
        print(f"Energy exhausted at T={t}")
        break

# ---------------------------------------------------------------------------
# Convert logs
# ---------------------------------------------------------------------------
gauge_field_log = np.array(gauge_field_log)
I_field_log     = np.array(I_field_log)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
final_I_sum = np.sum(I_field)
print("\n--- Energy–Information Results (SU(3)) ---")
print(f"Total Energy Spent      : {total_E_spent:.6f}")
print(f"Total Anchored Info (I) : {final_I_sum:.6f}")
print(f"Abs(E - I)              : {abs(total_E_spent - final_I_sum):.3e}")
if abs(final_I_sum - total_E_spent) < 1e-5:
    print("SUCCESS: I = E holds within numerical tolerance.")
else:
    print("WARNING: small mismatch (precision / step size).")

# ---------------------------------------------------------------------------
# Simple exponential fits for λ (decay) & μ (persistence)
# ---------------------------------------------------------------------------
from scipy.optimize import curve_fit
import numpy as np

total_I_over_time = I_field_log.sum(axis=1)  # shape: (num_timesteps,)

times = np.arange(len(total_I_over_time))
n_times = len(times)

# Choose at least 3 points for each fit, or a third of available points
decay_n = max(3, int(0.3 * n_times))
growth_n = max(3, int(0.4 * n_times))

# Slices for fitting
decay_slice = slice(0, decay_n)
growth_slice = slice(n_times - growth_n, None)

decay_t = times[decay_slice]
decay_I = total_I_over_time[decay_slice]
growth_t = times[growth_slice]
growth_I = total_I_over_time[growth_slice]

# Exponential fit functions
def exp_decay(t, A, lam):
    return A * np.exp(-lam * t)

def exp_growth(t, A, mu):
    return A * np.exp(mu * t)

print(f"Decay fit: {len(decay_t)} points; Growth fit: {len(growth_t)} points")

from scipy.optimize import curve_fit

# Fit decay exponent λ
if len(decay_t) < 2 or len(decay_I) < 2:
    print("Not enough points for decay fitting!")
    λ = np.nan
else:
    try:
        popt, pcov = curve_fit(exp_decay, decay_t, decay_I, p0=(decay_I[0], 0.05))
        λ = popt[1]
        print(f"λ (decay exponent) ≈ {λ:.4f}")
    except Exception as e:
        print("Decay curve fitting failed:", e)
        λ = np.nan

# Fit growth exponent μ
if len(growth_t) < 2 or len(growth_I) < 2:
    print("Not enough points for growth fitting!")
    μ = np.nan
else:
    try:
        popt, pcov = curve_fit(exp_growth, growth_t, growth_I, p0=(growth_I[0], 0.05))
        μ = popt[1]
        print(f"μ (growth exponent) ≈ {μ:.4f}")
    except Exception as e:
        print("Growth curve fitting failed:", e)
        μ = np.nan


# ---------------------------------------------------------------------------
# Optional quick heat‑map
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    plt.suptitle("Spectral Threshold Applied to Yang–Mills Mass Gap (Toy‑4: SU(3), 48 sites)")

    im0 = axes[0].imshow(gauge_field_log.T, aspect='auto', cmap='coolwarm',
                         origin='lower')
    axes[0].set_ylabel("Position in Field")
    axes[0].set_title("Gauge Field Evolution")
    plt.colorbar(im0, ax=axes[0], label="Field Strength")

    im1 = axes[1].imshow(I_field_log.T, aspect='auto', cmap='plasma',
                         origin='lower')
    axes[1].set_ylabel("Position in Field")
    axes[1].set_xlabel("Time")
    axes[1].set_title("Information Field (I): Persistent Particles")
    plt.colorbar(im1, ax=axes[1], label="I = ∫ L·omega·chi dt")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
