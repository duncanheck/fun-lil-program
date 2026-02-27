"""
peclet_calculator.py

Centralized module for calculating all Péclet number (Pe) variants used in the
all-inclusive spray drying simulation program.

Covers the full workflow:
- Training runs (feature engineering on complete historical dataset)
- Particle evolution (time-series tracking during d²-law shrinkage)
- Morphology prediction (ML features: effective_pe, max_pe, integrated_pe)
- Surface composition analysis (compound-specific enrichment via Pe)

Provides:
- Global (primary solute) Pe timeline and derived metrics
- Compound-specific effective_pe_X and max_pe_X (drug, moni, stabilizer, etc.)
- Integrated and time-averaged (effective) Pe

Author: Integrated from existing simulation logic (December 2025)
Dependencies: numpy
"""

import numpy as np
from typing import Dict, Optional, List

# ────────────────────────────────────────────────
# Absolute imports (fixed for root-level execution and __init__.py loading)
# ────────────────────────────────────────────────
from spraydrier.core.constants import R_AIR, R_N2, R_v, C_p_water, C_p_vapor, R_UNIVERSAL, M_AIR, M_N2
from spraydrier.core.properties import (
    calculate_psat_tetens,
    calculate_wet_bulb_temperature,
    fetch_gas_properties_from_table,
    classify_compound,
    calculate_mixed_solution_density
)
from spraydrier.core.diffusion_coefficient import calculate_diffusion_for_compounds
from spraydrier.core.temperature_evolution import calculate_realistic_droplet_temperature_evolution
from spraydrier.core.tg_calculator import detect_glass_transition, calculate_mixture_tg_gordon_taylor, calculate_shell_and_tg
from spraydrier.core.darcy_pressure import calculate_complete_darcy_analysis


def calculate_all_peclet_metrics(
    time_history_s: np.ndarray,
    radius_history_um: np.ndarray,
    v_evap_history_m_s: Optional[np.ndarray] = None,
    D_primary_m2_s: float = 1e-10,
    D_compounds_m2_s: Optional[Dict[str, float]] = None
) -> Dict[str, object]:
    """
    Compute comprehensive Péclet number metrics from simulation histories.

    Parameters
    ----------
    time_history_s : np.ndarray
        Time array in seconds.
    radius_history_um : np.ndarray
        Droplet/particle radius history in micrometers.
    v_evap_history_m_s : np.ndarray, optional
        Surface recession velocity history (m/s). If None, computed from radius.
    D_primary_m2_s : float
        Diffusion coefficient of primary solute (m²/s) for global Pe.
    D_compounds_m2_s : dict, optional
        Compound-specific diffusion coefficients, e.g. {'drug': 5e-12, 'moni': 1e-11}.

    Returns
    -------
    dict
        Contains global and compound-specific Pe metrics.
    """
    if len(time_history_s) != len(radius_history_um):
        raise ValueError("time_history_s and radius_history_um must have same length")

    # Convert radius to meters
    radius_m = radius_history_um * 1e-6

    # Compute v_evap if not provided (positive = evaporation)
    if v_evap_history_m_s is None:
        dr_dt = np.diff(radius_m) / np.diff(time_history_s)
        v_evap = np.concatenate([[np.abs(dr_dt[0])], np.abs(dr_dt)])
    else:
        v_evap = np.array(v_evap_history_m_s)
        if len(v_evap) != len(time_history_s):
            raise ValueError("v_evap_history_m_s must match time_history length")

    v_evap = np.maximum(v_evap, 0.0)  # Ensure non-negative

    # Global Pe (primary solute)
    pe_global = v_evap * radius_m / D_primary_m2_s
    pe_global = np.maximum(pe_global, 0.0)

    # Derived global metrics
    dt = np.diff(time_history_s)
    dt = np.concatenate([dt, [dt[-1]]])  # Pad for integration
    integrated_pe = np.trapz(pe_global, time_history_s)
    total_time = time_history_s[-1] - time_history_s[0]
    effective_pe = integrated_pe / total_time if total_time > 0 else 0.0
    max_pe = np.max(pe_global)

    results = {
        'pe_global_timeline': pe_global.tolist(),
        'max_pe': float(max_pe),
        'effective_pe': float(effective_pe),
        'integrated_pe': float(integrated_pe),
        'total_drying_time_s': float(total_time)
    }

    # Compound-specific metrics
    if D_compounds_m2_s:
        for comp, D in D_compounds_m2_s.items():
            if D <= 0:
                continue
            pe_comp = v_evap * radius_m / D
            pe_comp = np.maximum(pe_comp, 0.0)

            integrated_comp = np.trapz(pe_comp, time_history_s)
            effective_comp = integrated_comp / total_time if total_time > 0 else 0.0
            max_comp = np.max(pe_comp)

            results[f'effective_pe_{comp}'] = float(effective_comp)
            results[f'max_pe_{comp}'] = float(max_comp)
            results[f'integrated_pe_{comp}'] = float(integrated_comp)
            results[f'pe_timeline_{comp}'] = pe_comp.tolist()

    return results


if __name__ == "__main__":
    # Demo / validation
    import matplotlib.pyplot as plt

    t = np.linspace(0, 0.1, 200)  # 100 ms
    R_um = 10 * (1 - t/0.1)**0.5  # d²-law shrinkage
    v_evap = np.full_like(t, 1e-6)  # ~1 µm/s

    metrics = calculate_all_peclet_metrics(
        time_history_s=t,
        radius_history_um=R_um,
        v_evap_history_m_s=v_evap,
        D_primary_m2_s=5e-12,
        D_compounds_m2_s={'drug': 5e-12, 'moni': 1e-11}
    )

    print("=== Pe Metrics Demo ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.3f}")
        elif 'timeline' in k:
            print(f"{k}: array of length {len(v)}")

    plt.figure(figsize=(8, 5))
    plt.plot(t * 1000, metrics['pe_global_timeline'], label='Global Pe')
    plt.xlabel('Time (ms)')
    plt.ylabel('Péclet Number')
    plt.title('Pe Evolution During Droplet Drying')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()