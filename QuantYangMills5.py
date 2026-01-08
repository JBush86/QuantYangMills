# Discovered by Jared C. Bush, May 2025
# Copyright (c) 2026 Jared C. Bush
# Licensed under MIT License
# Reference Paper: https://doi.org/10.5281/zenodo.18158682

import numpy as np
from scipy.optimize import curve_fit

# ----- Configurable Simulation Function -----
def run_SpectralThreshold_YangMills_sim(
    FIELD_SIZE=48,
    N_STEPS=250,
    BETA=6.0,
    energy_budget_init=20.0,
    SEED=2077,
    MASS_GAP_INFO_THRESHOLD=0.0213,
    verbose=False
):
    np.random.seed(SEED)
    # Initial "gauge field" (simple real-valued for toy), zeroed
    field = np.zeros(FIELD_SIZE)
    I_field = np.zeros(FIELD_SIZE)
    energy_budget = energy_budget_init

    # Simple update: at each step, excite a random site if enough energy
    gauge_field_log = []
    I_field_log = []
    for t in range(N_STEPS):
        # Random excitation: spike amplitude, but only if energy left
        if energy_budget > 0:
            site = np.random.randint(FIELD_SIZE)
            amp = np.random.uniform(1.5, 4.0)
            field[site] += amp
            energy_budget -= amp
            if verbose:
                print(f"T={t}: Structured excitation at {site} (Amp={amp:.2f})")
        # Info update: L*omega*chi for this toy = just amplitude for now
        # More sophistication: L = abs(field), omega = 1, chi = 1 (could expand)
        L = np.abs(field)
        omega = np.ones_like(field)
        chi = np.ones_like(field)
        I_field += (L * omega * chi) * 0.1  # Info accumulates with decay (tunable)

        # Info "decays" at sites below threshold
        below = I_field < MASS_GAP_INFO_THRESHOLD
        I_field[below] *= 0.90
        # Log states for fitting
        gauge_field_log.append(field.copy())
        I_field_log.append(I_field.copy())
        # Optional: slow field decay to mimic realistic dissipation
        field *= 0.99

    # Prepare logs
    gauge_field_log = np.array(gauge_field_log)
    I_field_log = np.array(I_field_log)
    total_I_over_time = I_field_log.sum(axis=1)
    times = np.arange(len(total_I_over_time))
    n_times = len(times)

    # Fit exponents
    def exp_decay(t, A, lam): return A * np.exp(-lam * t)
    def exp_growth(t, A, mu): return A * np.exp(mu * t)
    decay_n = max(3, int(0.3 * n_times))
    growth_n = max(3, int(0.4 * n_times))
    decay_t, decay_I = times[:decay_n], total_I_over_time[:decay_n]
    growth_t, growth_I = times[-growth_n:], total_I_over_time[-growth_n:]
    try:
        λ = curve_fit(exp_decay, decay_t, decay_I, p0=(decay_I[0], 0.05))[0][1]
    except Exception: λ = np.nan
    try:
        μ = curve_fit(exp_growth, growth_t, growth_I, p0=(growth_I[0], 0.05))[0][1]
    except Exception: μ = np.nan

    total_E_spent = energy_budget_init - energy_budget
    final_I_sum = I_field.sum()

    return {
        "delta_I_crit": MASS_GAP_INFO_THRESHOLD,
        "lambda": λ,
        "mu": μ,
        "total_energy": total_E_spent,
        "total_info": final_I_sum,
        "gauge_field_log": gauge_field_log,
        "I_field_log": I_field_log,
        "total_I_over_time": total_I_over_time,
        "times": times,
    }

# ----- A1: β-sweep -----
def run_beta_sweep():
    betas = [5.6, 5.8, 6.0, 6.2]
    results = []
    for beta in betas:
        res = run_SpectralThreshold_YangMills_sim(BETA=beta, MASS_GAP_INFO_THRESHOLD=0.0213, verbose=False)
        results.append(res)
    print("A1: β sweep results:")
    for beta, res in zip(betas, results):
        print(f"β={beta:4.2f} | ΔI_crit={res['delta_I_crit']:.4f} | λ={res['lambda']:.4f} | μ={res['mu']:.4f}")
    return results 
    # Note: In this simplified toy model, BETA does not explicitly scale the random kicks.

# ----- A2: Energy-budget sweep -----
def run_energy_sweep():
    budgets = [10, 20, 40]
    results = []
    for budget in budgets:
        res = run_SpectralThreshold_YangMills_sim(energy_budget_init=budget, MASS_GAP_INFO_THRESHOLD=0.0213, verbose=False)
        results.append(res)
    print("A2: Energy budget sweep:")
    for budget, res in zip(budgets, results):
        print(f"E_budget={budget:2d} | I={res['total_info']:.2f} | λ={res['lambda']:.4f} | μ={res['mu']:.4f}")
    return results

# ----- A3: Random-seed batch -----
def run_seed_batch(N=20):
    seeds = np.random.randint(0, 1e6, size=N)
    results = []
    for i, seed in enumerate(seeds):
        res = run_SpectralThreshold_YangMills_sim(SEED=seed, MASS_GAP_INFO_THRESHOLD=0.0213, verbose=False)
        results.append(res)
    delta_Is = np.array([r["delta_I_crit"] for r in results])
    lambdas = np.array([r["lambda"] for r in results])
    mus = np.array([r["mu"] for r in results])
    print("A3: Seed batch stats:")
    print(f"ΔI_crit mean ± std = {delta_Is.mean():.4f} ± {delta_Is.std():.4f}")
    print(f"λ mean ± std       = {lambdas.mean():.4f} ± {lambdas.std():.4f}")
    print(f"μ mean ± std       = {mus.mean():.4f} ± {mus.std():.4f}")
    return results

# ----- Main Driver -----
if __name__ == "__main__":
    print("Running β sweep (A1)...")
    run_beta_sweep()
    print("\nRunning energy-budget sweep (A2)...")
    run_energy_sweep()
    print("\nRunning random-seed batch (A3)...")
    run_seed_batch(N=20)
