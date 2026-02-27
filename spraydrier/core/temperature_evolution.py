"""
Temperature Evolution Module
============================

Physically accurate droplet/particle temperature evolution.
- Starts at wet-bulb temperature (calculated from inlet T & RH)
- Transitions to outlet temperature with exponential decay
- Optional delta_T for custom cooling rate control

Author: Doug Hecker (updated with Grok assistance)
Date: February 2026
"""

import numpy as np
import math
from typing import Optional, Tuple, Dict

# Relative imports from core/
try:
    from .properties import calculate_wet_bulb_temperature, calculate_psat_tetens
    from .constants import C_p_water, C_p_vapor
    MAIN_IMPORTS_AVAILABLE = True
except ImportError:
    MAIN_IMPORTS_AVAILABLE = False
    # Fallback inline implementations
    def calculate_psat_tetens(T_C):
        if T_C >= 50:
            A, B, C = 8.07131, 1730.63, 233.426
            Psat_mmHg = 10 ** (A - B / (T_C + C))
            return Psat_mmHg * 133.322
        return 610.78 * math.exp((17.27 * T_C) / (T_C + 237.3)) * 1000

    def calculate_wet_bulb_temperature(T_dry_C, RH, C_p_gas=1005, h_vap=2260e3):
        P_v = (RH / 100) * calculate_psat_tetens(T_dry_C)
        def residual(T_wb_C):
            P_sat_wb = calculate_psat_tetens(max(T_wb_C, 0))
            return (P_sat_wb - P_v) - (C_p_gas * (T_dry_C - T_wb_C)) / (h_vap * 0.622)
        try:
            T_wb_C = brentq(residual, -10, T_dry_C, xtol=1e-5)
        except ValueError:
            T_wb_C = (T_dry_C + 0) / 2
        return T_wb_C

    C_p_water = 4186
    C_p_vapor = 1840


def calculate_realistic_droplet_temperature_evolution(
    T_dry_C: float,
    RH: float,
    t_eval: np.ndarray,
    delta_T: Optional[float] = None,
    time_constant: float = 0.02,  # seconds - adjust for drying speed
    debug: bool = False
) -> np.ndarray:
    """
    Calculate realistic droplet temperature evolution.
    
    Physics:
    - Starts at wet-bulb temperature (evaporative cooling)
    - Transitions toward outlet temperature with exponential decay
    - Rate controlled by delta_T (if provided) or estimated
    
    Parameters:
    -----------
    T_dry_C : float
        Inlet dry-bulb temperature (°C)
    RH : float
        Inlet relative humidity (%)
    t_eval : np.ndarray
        Time points (seconds)
    delta_T : float, optional
        Temperature rise from wet-bulb to outlet (T_outlet - T_wb)
        If None, estimated as 70°C (typical spray drying)
    time_constant : float
        Exponential decay time constant (seconds)
    debug : bool
        Print debug info
        
    Returns:
    --------
    temperature_array : np.ndarray
        Temperature at each time point (°C)
    """
    # Calculate actual wet-bulb temperature
    T_wb = calculate_wet_bulb_temperature(T_dry_C, RH)

    # Estimate or use provided delta_T
    if delta_T is None:
        # Typical drying delta (inlet - wet-bulb ≈ outlet rise)
        delta_T = 70.0  # fallback - realistic for most spray drying
        if debug:
            print(f"[Temperature] Using estimated delta_T = {delta_T:.1f}°C")
    else:
        if debug:
            print(f"[Temperature] Using provided delta_T = {delta_T:.1f}°C")

    # Exponential decay from wet-bulb toward outlet-like temperature
    # T(t) = T_wb + delta_T * (1 - exp(-t / tau))
    temperature_array = T_wb + delta_T * (1 - np.exp(-t_eval / time_constant))

    # Safety clip: never exceed inlet or go below reasonable bounds
    temperature_array = np.clip(temperature_array, max(T_wb - 5, 0), T_dry_C)

    if debug:
        print(f"[Temperature] Wet-bulb: {T_wb:.2f}°C")
        print(f"[Temperature] Initial: {temperature_array[0]:.2f}°C")
        print(f"[Temperature] Final: {temperature_array[-1]:.2f}°C")
        print(f"[Temperature] Delta achieved: {temperature_array[-1] - T_wb:.2f}°C")

    return temperature_array


if __name__ == "__main__":
    """Quick test / demo"""
    print("="*60)
    print("TEMPERATURE EVOLUTION TEST")
    print("="*60)

    T_inlet = 37.0
    RH_inlet = 55.0
    t_eval = np.linspace(0, 0.3, 100)  # 300 ms typical chamber time

    # With default delta_T
    T_array_default = calculate_realistic_droplet_temperature_evolution(
        T_inlet, RH_inlet, t_eval, debug=True
    )

    # With custom delta_T
    T_array_custom = calculate_realistic_droplet_temperature_evolution(
        T_inlet, RH_inlet, t_eval, delta_T=45.0, debug=True
    )

    print("\nTest complete.")