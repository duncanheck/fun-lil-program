"""
diffusion_coefficient.py

Calculate compound-specific diffusion coefficients in multi-component spray drying feed solutions.
Uses Stokes-Einstein with crowding correction and class-based adjustments.
Integrates with properties.py for compound classification and known values.

Author: Doug Hecker (updated with Grok assistance)
Date: February 2026
"""

import math
import json
from pathlib import Path
import numpy as np
from typing import List, Dict, Optional

# Core import from your folder structure
try:
    from spraydrier.core.properties import classify_compound, load_compound_database
except ImportError:
    print("Warning: properties.py not found - using basic classification")
    def classify_compound(name, mw, conc_mg_ml=0.0, is_drug=False):
        return {'class': 'other', 'tg': None, 'k': 0.5, 'specific_volume': 0.7, 'mw': mw}
    COMPOUND_DB = {}
else:
    COMPOUND_DB = load_compound_database()

# Physical constants
k_B = 1.380649e-23  # Boltzmann constant (J/K)
T_REF = 298.15      # Reference temperature (K = 25°C)

# Default specific volumes (cm³/g) - can be overridden by compound_props.json
DEFAULT_SPECIFIC_VOLUMES = {
    'protein': 0.73,
    'polymer': 0.8,
    'sugar': 0.6,
    'amino_acid': 0.7,
    'salt': 0.5,
    'surfactant': 0.93,
    'other': 0.7
}

# Known diffusion coefficients for small molecules/ions (m²/s at 25°C in water)
KNOWN_D = {
    'hcl': 3.5e-9,
    'nacl': 1.6e-9,
    'kcl': 2.0e-9,
    'tris': 8.0e-10,
    'acetate': 1.2e-9,
    'phosphate': 1.0e-9,
    'glycine': 1.05e-9,
    'ps80': 5.0e-11,  # Approximate for large surfactant
}

def calculate_radius(mw: float, compound_class: str, pH: Optional[float] = None, pI: Optional[float] = None) -> float:
    """
    Estimate hydrodynamic radius r_h in meters.
    Base formula: r_h ≈ 0.066 * MW^(1/3) nm (empirical for globular proteins)
    Adjusted for class and pH (near pI → aggregation).
    """
    if mw <= 0:
        raise ValueError("Molecular weight must be positive")

    base_r_nm = 0.066 * (mw ** (1/3))

    # Class-specific adjustments
    if compound_class == 'protein':
        base_r_nm *= 1.05  # Slightly larger for globular proteins
        if pH is not None and pI is not None and abs(pH - pI) < 1.0:
            base_r_nm *= 1.15  # ~15% increase near isoelectric point (aggregation)
    elif compound_class in ['polymer', 'surfactant']:
        base_r_nm *= 1.25  # Extended chains or micelles
    elif compound_class == 'sugar':
        base_r_nm *= 0.9   # Compact
    elif compound_class == 'salt':
        base_r_nm *= 0.7   # Small ions

    return base_r_nm * 1e-9  # nm → m

def calculate_volume_fraction(concentrations: List[float], classes: List[str]) -> float:
    """
    Calculate total volume fraction φ = Σ (conc_i * specific_volume_i)
    concentrations in mg/mL, converted to g/mL
    """
    phi = 0.0
    for conc_mg_ml, cls in zip(concentrations, classes):
        if conc_mg_ml < 0:
            raise ValueError("Concentration must be non-negative")
        conc_g_ml = conc_mg_ml / 1000.0
        # Get specific volume from DB or defaults
        sv = DEFAULT_SPECIFIC_VOLUMES.get(cls, DEFAULT_SPECIFIC_VOLUMES['other'])
        phi += conc_g_ml * sv
    return min(phi, 0.99)  # Cap to avoid >1

def calculate_diffusion_for_compounds(
    compounds: List[Dict],
    T: float = 298.15,
    eta_eff: float = 1.36e-3,
    k_crowding: float = 2.0
) -> Dict[str, float]:
    """
    Calculate diffusion coefficient D_i for each compound.
    
    D_i = (k_B * T) / (6 * π * η_eff * r_h) * f(φ)
    f(φ) = (1 - φ)^k  # simple crowding correction
    
    Parameters:
    -----------
    compounds : List[Dict]
        Each dict: {'name': str, 'mw': float, 'conc_mg_ml': float, 'class': str, 'pH': float (opt), 'pI': float (opt)}
    T : float
        Temperature (K)
    eta_eff : float
        Effective viscosity (Pa·s)
    k_crowding : float
        Crowding exponent (2-3 typical)
    
    Returns:
    --------
    Dict[str, float] : {compound_name: D in m²/s}
    """
    concentrations = [c['conc_mg_ml'] for c in compounds]
    classes = [c['class'] for c in compounds]
    phi = calculate_volume_fraction(concentrations, classes)
    f_phi = (1 - phi) ** k_crowding if phi < 1 else 1e-6  # Avoid division by zero

    D_values = {}
    for comp in compounds:
        name = comp['name']
        name_lower = name.lower()
        
        # Known small molecule/ion values
        if name_lower in KNOWN_D:
            D = KNOWN_D[name_lower]
            D *= T / T_REF  # Temperature correction (Stokes-Einstein)
            D_values[name] = D
            continue
        
        # Calculate hydrodynamic radius
        r_h = calculate_radius(
            comp['mw'],
            comp['class'],
            comp.get('pH'),
            comp.get('pI')
        )
        
        # Stokes-Einstein with crowding correction
        D = (k_B * T) / (6 * math.pi * eta_eff * r_h) * f_phi
        D_values[name] = D
    
    return D_values

def calculate_diffusion_coefficient_interactive():
    """
    Interactive CLI entry point for manual calculation/testing.
    Matches overview's emphasis on user-friendly inputs.
    """
    print("=== Diffusion Coefficient Calculator (Interactive) ===")
    print("Enter parameters for multi-component system\n")

    try:
        num = int(input("Number of compounds (1-5): "))
        if not 1 <= num <= 5:
            raise ValueError("Must be 1-5")

        compounds = []
        for i in range(num):
            print(f"\nCompound {i+1}:")
            name = input("Name: ").strip()
            mw = float(input("MW (Da): "))
            conc = float(input("Conc (mg/mL): "))
            cls_input = input("Class (protein/polymer/sugar/salt/surfactant/other): ").strip().lower()
            pH = float(input("pH (optional, Enter to skip): ") or 0) or None
            pI = float(input("pI (optional, Enter to skip): ") or 0) or None
            
            # Auto-classify if not provided
            if not cls_input:
                props = classify_compound(name, mw, conc, is_drug=(i==0))
                cls_input = props['class']
            
            compounds.append({
                'name': name,
                'mw': mw,
                'conc_mg_ml': conc,
                'class': cls_input,
                'pH': pH,
                'pI': pI
            })

        T = float(input("\nTemperature (K, e.g. 298): ") or 298.15)
        eta_eff_cp = float(input("Effective viscosity (cP): ") or 1.36)
        eta_eff = eta_eff_cp * 1e-3  # Pa·s
        k = float(input("Crowding factor k (2-3): ") or 2.5)

        D_dict = calculate_diffusion_for_compounds(compounds, T, eta_eff, k)

        print("\nResults:")
        print(f"Total volume fraction φ: {calculate_volume_fraction([c['conc_mg_ml'] for c in compounds], [c['class'] for c in compounds]):.4f}")
        for name, D in D_dict.items():
            print(f"{name}: D = {D:.2e} m²/s = {D*1e4:.2e} cm²/s")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    calculate_diffusion_coefficient_interactive()