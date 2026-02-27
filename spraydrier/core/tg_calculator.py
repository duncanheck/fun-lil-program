
#!/usr/bin/env python3
"""
Glass Transition Temperature (Tg) Calculations for Spray Drying
================================================================

Pure functions for calculating mixture Tg using Gordon-Taylor equation,
tracking Tg evolution during drying, detecting glass transitions,
and estimating shell formation time.

All functions are independent and can be called by any module.

Author: Doug Hecker (updated with Grok assistance)
Date: February 2026
"""

import numpy as np
import json
from pathlib import Path
import math
from typing import Dict, Tuple, Optional, List

# Relative import from core/
try:
    from .properties import classify_compound
except ImportError:
    print("Warning: classify_compound not available from properties.py - using fallback")
    def classify_compound(name, mw, conc_mg_ml=0.0, is_drug=False):
        return {'class': 'unknown', 'tg': None, 'k': 0.25, 'specific_volume': 0.7, 'mw': mw}

# ────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────
TG_WATER = -135.0  # Glass transition temperature of water in °C
KELVIN_OFFSET = 273.15  # Conversion from °C to Kelvin

# ────────────────────────────────────────────────
# Loading Tg data
# ────────────────────────────────────────────────
def load_component_tg_from_json(json_path: str = 'compound_props.json') -> Tuple[Dict, Dict]:
    """
    Load Tg values and Gordon-Taylor k constants from JSON.
    Tries multiple locations if file not found.
    
    Returns:
    --------
    component_tg : Dict[str, float]
        {component_name_lower: Tg_in_Celsius}
    component_k : Dict[str, float]
        {component_name_lower: Gordon_Taylor_constant}
    """
    tg_dict = {}
    k_dict = {}
    
    json_path = Path(json_path)
    
    # Try alternate locations
    if not json_path.exists():
        alternate_paths = [
            Path(__file__).parent / 'compound_props.json',
            Path(__file__).parent.parent / 'data' / 'compound_props.json',
            Path('./compound_props.json')
        ]
        for alt_path in alternate_paths:
            if alt_path.exists():
                json_path = alt_path
                break
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for name, props in data.items():
            name_lower = str(name).lower()
            tg_val = props.get('tg') or props.get('Tg') or props.get('Tg_C')
            if tg_val is not None:
                try:
                    tg_dict[name_lower] = float(tg_val)
                except (ValueError, TypeError):
                    pass
            
            k_val = props.get('k_water') or props.get('k_gordon_taylor') or props.get('k')
            if k_val is not None:
                try:
                    k_dict[name_lower] = float(k_val)
                except (ValueError, TypeError):
                    pass
    
    except Exception as e:
        print(f"Warning: Could not load Tg data from {json_path}: {e}")
    
    # Always add water (critical plasticizer)
    tg_dict['water'] = TG_WATER
    k_dict['water'] = 1.0
    
    # Default k for missing components
    default_k = {
        'igg': 0.20, 'protein': 0.20, 'mab': 0.20,
        'moni': 0.22,
        'trehalose': 0.20, 'sucrose': 0.18, 'lactose': 0.19,
        'ps80': 0.30, 'polysorbate80': 0.30
    }
    for comp in tg_dict.keys():
        if comp not in k_dict:
            k_dict[comp] = default_k.get(comp, 0.20)
    
    return tg_dict, k_dict

# ────────────────────────────────────────────────
# Core Tg calculation functions
# ────────────────────────────────────────────────
def calculate_mass_fractions_from_moisture(
    moisture_fraction: float,
    solids_composition_dry_basis: Dict[str, float]
) -> Dict[str, float]:
    """
    Convert moisture fraction to full component mass fractions.
    
    Parameters
    ----------
    moisture_fraction : float
        Water mass fraction (0-1)
    solids_composition_dry_basis : dict
        {component: fraction_of_dry_solids} summing to 1.0
    
    Returns
    -------
    mass_fractions : dict
        {component: mass_fraction} including water, sums to 1.0
    """
    mass_fracs = {'water': moisture_fraction}
    solids_fraction = 1.0 - moisture_fraction
    
    for comp, dry_frac in solids_composition_dry_basis.items():
        mass_fracs[str(comp).lower()] = dry_frac * solids_fraction
    
    # Normalize
    total = sum(mass_fracs.values())
    if total > 0:
        mass_fracs = {k: v / total for k, v in mass_fracs.items()}
    
    return mass_fracs

def calculate_mixture_tg_gordon_taylor(
    mass_fractions: Dict[str, float],
    component_tg_dict: Dict[str, float],
    k_values: Dict[str, float]
) -> float:
    """
    Calculate mixture Tg using Gordon-Taylor equation for multi-component system.
    
    Tg_mix = Σ(wi × Tgi × ki) / Σ(wi × ki)
    """
    numerator = 0.0
    denominator = 0.0
    
    for comp_lower, w in mass_fractions.items():
        if w <= 0:
            continue
        
        Tg_C = component_tg_dict.get(comp_lower)
        if Tg_C is None:
            continue
        Tg_K = Tg_C + KELVIN_OFFSET
        
        k = k_values.get(comp_lower, 0.20)
        
        numerator += w * Tg_K * k
        denominator += w * k
    
    if denominator == 0:
        return 25.0  # fallback
    
    Tg_mix_K = numerator / denominator
    return Tg_mix_K - KELVIN_OFFSET

def detect_glass_transition(
    time_array: np.ndarray,
    T_droplet_array: np.ndarray,
    Tg_mix_array: np.ndarray
) -> Tuple[Optional[float], Optional[int], Dict]:
    """
    Detect when droplet temperature crosses below mixture Tg.
    Returns time/index of transition and details.
    """
    for i in range(len(time_array)):
        if T_droplet_array[i] < Tg_mix_array[i]:
            info = {
                'transition_detected': True,
                'time_s': float(time_array[i]),
                'time_ms': float(time_array[i] * 1000),
                'index': int(i),
                'T_droplet_C': float(T_droplet_array[i]),
                'Tg_mixture_C': float(Tg_mix_array[i]),
                'delta_T_C': float(T_droplet_array[i] - Tg_mix_array[i]),
                'message': f'Glass transition at t={time_array[i]*1000:.1f} ms: T_droplet={T_droplet_array[i]:.1f}°C < Tg={Tg_mix_array[i]:.1f}°C'
            }
            return time_array[i], i, info
    
    info = {
        'transition_detected': False,
        'reason': 'T_droplet remained above Tg throughout drying',
        'T_droplet_min': float(np.min(T_droplet_array)),
        'Tg_mixture_max': float(np.max(Tg_mix_array)),
        'message': 'No glass transition detected - particle remained rubbery/liquid'
    }
    return None, None, info

def tg_evolution_during_drying(
    moisture_array: np.ndarray,
    solids_composition_dry_basis: Dict[str, float],
    component_tg_dict: Dict[str, float],
    k_values: Dict[str, float]
) -> np.ndarray:
    """
    Calculate Tg trajectory during drying as moisture decreases.
    """
    tg_trajectory = []
    
    for moisture in moisture_array:
        mass_fracs = calculate_mass_fractions_from_moisture(moisture, solids_composition_dry_basis)
        Tg_mix = calculate_mixture_tg_gordon_taylor(mass_fracs, component_tg_dict, k_values)
        tg_trajectory.append(Tg_mix)
    
    return np.array(tg_trajectory)

# ────────────────────────────────────────────────
# NEW: Shell formation time + Tg array (what simulation.py was missing)
# ────────────────────────────────────────────────
def calculate_shell_and_tg(
    t_dry: float,
    T_inlet_C: float,
    T_outlet_C: float,
    t_eval: np.ndarray,
    moisture_array: Optional[np.ndarray] = None,
    solids_composition: Optional[Dict[str, float]] = None,
    debug: bool = False
) -> Tuple[float, np.ndarray]:
    """
    Estimate shell formation time and Tg evolution array.
    
    Shell formation time: when T_droplet first drops below Tg_mix.
    If moisture_array not provided, uses simple exponential decay.
    
    Returns:
    --------
    shell_formation_time : float
        Time of shell formation (s)
    Tg_array : np.ndarray
        Mixture Tg at each t_eval point (°C)
    """
    # Load Tg data if not provided
    component_tg, component_k = load_component_tg_from_json()
    
    # Default solids composition if not provided
    if solids_composition is None:
        solids_composition = {'drug': 0.7, 'moni': 0.2, 'sugar': 0.1}
    
    # Default moisture evolution if not provided
    if moisture_array is None:
        # Exponential drying from 0.9 to 0.05
        moisture_array = 0.05 + (0.9 - 0.05) * np.exp(-t_eval / (t_dry / 4))
    
    # Calculate Tg evolution
    Tg_array = tg_evolution_during_drying(
        moisture_array, solids_composition, component_tg, component_k
    )
    
    # Estimate droplet temperature trajectory (simple linear for now)
    T_droplet_array = np.linspace(T_inlet_C, T_outlet_C, len(t_eval))
    
    # Detect glass transition
    shell_time, _, info = detect_glass_transition(t_eval, T_droplet_array, Tg_array)
    
    if info['transition_detected']:
        shell_formation_time = info['time_s']
        if debug:
            print(f"[Tg] Shell formation detected at {shell_formation_time*1000:.1f} ms")
    else:
        shell_formation_time = t_dry * 0.35  # fallback: ~35% of drying time
        if debug:
            print(f"[Tg] No transition detected - using fallback shell time: {shell_formation_time*1000:.1f} ms")
    
    return shell_formation_time, Tg_array

# ────────────────────────────────────────────────
# Test / Demo when run directly
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*70)
    print("TG CALCULATOR MODULE - TEST MODE")
    print("="*70)
    
    print("\n1. Loading Tg data...")
    component_tg, component_k = load_component_tg_from_json()
    print(f"Loaded {len(component_tg)} components with Tg data")
    
    print("\n2. Testing Tg calculation at different moistures...")
    solids_dry_basis = {'igg': 0.75, 'moni': 0.15, 'trehalose': 0.10}
    
    print(f"{'Moisture (%)':>15} {'Tg mixture (°C)':>20}")
    print("-" * 40)
    for moisture_pct in [95, 80, 60, 40, 20, 10, 5, 3]:
        moisture_frac = moisture_pct / 100.0
        mass_fracs = calculate_mass_fractions_from_moisture(moisture_frac, solids_dry_basis)
        Tg_mix = calculate_mixture_tg_gordon_taylor(mass_fracs, component_tg, component_k)
        print(f"{moisture_pct:>15.0f} {Tg_mix:>20.1f}")
    
    print("\n3. Testing shell formation & Tg array...")
    t_dry = 0.3  # 300 ms drying time
    t_eval = np.linspace(0, t_dry, 100)
    shell_time, Tg_array = calculate_shell_and_tg(
        t_dry, T_inlet_C=37.0, T_outlet_C=30.0, t_eval=t_eval, debug=True
    )
    
    print(f"\nShell formation time: {shell_time*1000:.1f} ms")
    print(f"Tg at start: {Tg_array[0]:.1f}°C")
    print(f"Tg at end: {Tg_array[-1]:.1f}°C")
    
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70)