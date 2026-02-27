"""
calculate_mixed_solution_density.py

Calculate density of a mixed aqueous solution using volume additivity.
Handles multi-component feeds typical in mAb spray drying (drug, Moni, buffer, stabilizer, additives).

Uses class-specific densities from properties.py classification.

Author: Doug Hecker (enhanced with Grok assistance)
Date: February 2026
"""

import numpy as np
from typing import List, Dict, Optional

# Relative import from same core/ package
try:
    from .properties import classify_compound
except ImportError:
    print("Warning: classify_compound not available - using fallback classification")
    def classify_compound(name, mw, conc_mg_ml=0.0, is_drug=False):
        return {'class': 'other', 'density': 1.0}


# Class-specific densities (g/cm³) - can be overridden by compound_props.json
DEFAULT_DENSITIES = {
    'protein': 1.35,        # mAbs, IgG, drug substances
    'polymer': 1.30,        # Moni, amphiphilic polymers
    'sugar': 1.59,          # trehalose, sucrose, stabilizers
    'amino_acid': 1.30,     # glycine, histidine
    'salt': 1.20,           # buffer salts
    'surfactant': 1.05,     # PS80, Tween
    'other': 1.00           # water-like or unknown
}


def calculate_mixed_solution_density(
    compounds: List[Dict],
    water_density: float = 1.0,
    use_compound_props: bool = True
) -> float:
    """
    Calculate density of aqueous solution using volume additivity rule.

    ρ_solution = total_mass / total_volume

    Parameters:
    -----------
    compounds : List[Dict]
        Each dict should have:
        - 'conc_mg_ml': float (concentration in mg/mL)
        - 'name': str (optional)
        - 'mw': float (optional, used for classification)
        - 'class': str (optional, overrides classification)
    water_density : float
        Density of pure water (g/cm³), default 1.0
    use_compound_props : bool
        If True, tries to use density from compound_props.json via classify_compound

    Returns:
    --------
    rho_solution : float
        Solution density in g/cm³ (≈ g/mL)
    """
    if not compounds:
        return water_density

    total_mass_g_per_ml = 0.0
    total_volume_ml_per_ml = 0.0

    for comp in compounds:
        conc_mg_ml = comp.get('conc_mg_ml', 0.0)
        if conc_mg_ml <= 0:
            continue

        conc_g_ml = conc_mg_ml / 1000.0  # mg/mL → g/mL

        # Get compound class
        comp_class = comp.get('class')
        if not comp_class:
            # Auto-classify if name/mw provided
            name = comp.get('name', 'unknown')
            mw = comp.get('mw', 0.0)
            props = classify_compound(name, mw, conc_mg_ml, is_drug=True)
            comp_class = props.get('class', 'other')

        # Get density for this class
        rho_comp = DEFAULT_DENSITIES.get(comp_class.lower(), 1.0)

        # Optional override from compound_props.json (if density key exists)
        if use_compound_props and 'density' in props:
            rho_comp = props['density']

        # Accumulate mass and volume
        total_mass_g_per_ml += conc_g_ml
        total_volume_ml_per_ml += conc_g_ml / rho_comp

    # Water contribution (remaining volume)
    volume_water = 1.0 - total_volume_ml_per_ml
    if volume_water < 0:
        print("Warning: Solute volume exceeds 1 mL - capping density")
        volume_water = 0.0

    mass_water = volume_water * water_density
    total_mass = mass_water + total_mass_g_per_ml
    total_volume = volume_water + total_volume_ml_per_ml

    if total_volume <= 0:
        return water_density

    rho_solution = total_mass / total_volume

    return rho_solution


# ────────────────────────────────────────────────
# Example / Test when run directly
# ────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("MIXED SOLUTION DENSITY CALCULATOR - TEST")
    print("="*60)

    # Example formulation (typical mAb + Moni + excipients)
    test_compounds = [
        {'name': 'BHV-1400', 'conc_mg_ml': 100.0, 'mw': 150000},  # drug/protein
        {'name': 'Moni', 'conc_mg_ml': 20.0, 'mw': 6800},         # polymer
        {'name': 'Trehalose', 'conc_mg_ml': 50.0, 'mw': 342},     # stabilizer/sugar
        {'name': 'Glycine', 'conc_mg_ml': 10.0, 'mw': 75},        # buffer
        {'name': 'PS80', 'conc_mg_ml': 1.0, 'mw': 1310}           # surfactant
    ]

    rho = calculate_mixed_solution_density(test_compounds)
    print(f"Test formulation total solute conc: {sum(c['conc_mg_ml'] for c in test_compounds):.1f} mg/mL")
    print(f"Calculated solution density: {rho:.4f} g/cm³")
    print(f"Compared to water: {rho:.4f} g/cm³ (increase of {(rho-1.0)*100:.1f}%)")