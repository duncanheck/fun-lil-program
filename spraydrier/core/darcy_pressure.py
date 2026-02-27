"""
Darcy Pressure Module for Spray Drying Morphology Prediction
=============================================================

Implements Darcy's law to calculate internal pressure buildup in drying particles.
Key for morphology: spheres → dimpled → hollow/donuts → cracked.

Uses:
- Shell thickness from radius shrinkage
- Kozeny-Carman permeability (porosity + pore size)
- Vapor flow through shell (evaporation rate → volumetric flow)
- Temperature-dependent vapor viscosity (Sutherland)
- Dimensionless pressure Π = ΔP / P_capillary
- Morphology decision tree (pressure + Pe + Ma + shell formation)

Author: Doug Hecker (enhanced with Grok assistance)
Date: February 2026
Version: 1.2
"""



import numpy as np
import math
from typing import Dict, Tuple, Optional

# Absolute import for reliability during debugging
from spraydrier.core.properties import calculate_psat_tetens
print("[DARCY] Successfully imported calculate_psat_tetens")  # DEBUG - remove later

# Physical constants
PI = math.pi
R_VAPOR = 461.5
P_AMBIENT = 101325.0

# ... rest of your functions unchanged (calculate_shell_thickness, estimate_shell_permeability, etc.)
# Make sure calculate_internal_pressure_darcy uses the imported calculate_psat_tetens

# Typical pore size ranges (m) - used when porosity estimated
PORE_SIZE_RANGES = {
    'moni': 10e-9,        # amphiphilic polymer - very dense
    'protein': 20e-9,     # large proteins - dense packing
    'sugar': 75e-9,       # stabilizers/sugars - open structure
    'buffer': 55e-9,      # small molecules
    'additive': 45e-9,    # variable
    'default': 40e-9
}

def calculate_shell_thickness(
    R_current: float,
    R_initial: float,
    solids_fraction: float,
    moisture: float
) -> float:
    """
    Calculate dried shell thickness assuming outside-in drying.
    Uses volume conservation of solids + minimum thickness guard.
    """
    if R_current >= R_initial:
        return 0.1e-6  # No shrinkage yet

    V_initial = (4/3) * PI * R_initial**3
    V_solids = V_initial * solids_fraction
    V_current = (4/3) * PI * R_current**3

    # If significant shrinkage, assume hollow core + shell
    if V_current < V_initial * 0.85:
        L_shell = R_initial - R_current
    else:
        L_shell = R_initial * (1 - (V_current / V_initial)**(1/3))

    return max(L_shell, 0.1e-6)  # 0.1 μm minimum


def estimate_shell_permeability(
    composition: Dict[str, float],
    moisture: float,
    T_droplet: float = 80.0,
    porosity: Optional[float] = None
) -> Tuple[float, float]:
    """
    Estimate shell permeability κ using Kozeny-Carman.
    κ = (d_pore² × ε³) / (180 × (1-ε)²)
    
    Porosity estimated from moisture, composition, and temperature.
    Pore size weighted by composition fractions.
    """
    # Estimate porosity if not provided
    if porosity is None:
        base_porosity = 0.35  # Typical for spray-dried biologics

        # Drier = more porous
        moisture_effect = 0.25 * (1 - moisture)

        # Composition effect (dense = lower porosity)
        protein_frac = composition.get('protein', 0) + composition.get('drug', 0) + composition.get('mab', 0)
        moni_frac = composition.get('moni', 0)
        sugar_frac = composition.get('sugar', 0) + composition.get('stabilizer', 0)
        buffer_frac = composition.get('buffer', 0)
        additive_frac = composition.get('additive', 0)

        comp_effect = -0.15 * protein_frac - 0.25 * moni_frac + 0.1 * sugar_frac + 0.05 * (buffer_frac + additive_frac)
        porosity = base_porosity + moisture_effect + comp_effect
        porosity = np.clip(porosity, 0.10, 0.70)

    # Weighted pore size based on composition fractions
    total_frac = sum(composition.values())
    if total_frac > 0:
        weighted_pore = sum(
            frac * PORE_SIZE_RANGES.get(comp.lower(), PORE_SIZE_RANGES['default'])
            for comp, frac in composition.items()
        ) / total_frac
    else:
        weighted_pore = PORE_SIZE_RANGES['default']

    d_pore = weighted_pore

    # Temperature effect: higher T → slight pore expansion
    if T_droplet > 100:
        d_pore *= 1.15

    # Kozeny-Carman
    if porosity <= 0 or porosity >= 1:
        kappa = 1e-15  # Very low permeability
    else:
        kappa = (d_pore**2 * porosity**3) / (180 * (1 - porosity)**2)

    return kappa, porosity


def calculate_vapor_viscosity(T_celsius: float) -> float:
    """
    Water vapor viscosity using Sutherland's formula.
    """
    T_k = T_celsius + 273.15
    T0 = 373.15
    mu0 = 12.27e-6
    S = 961.0
    return mu0 * ((T0 + S) / (T_k + S)) * (T_k / T0)**1.5


def calculate_internal_pressure_darcy(
    R_current: float,
    L_shell: float,
    kappa: float,
    evaporation_rate: float,  # v_s (m/s)
    T_droplet: float,
    moisture: float,
    P_ambient: float = P_AMBIENT
) -> Tuple[float, float, float]:
    """
    Calculate internal pressure ΔP using Darcy's law for vapor flow through shell.
    ΔP = (μ × L × v_superficial) / κ
    """
    if L_shell <= 0 or kappa <= 0:
        return P_ambient, 0.0, 0.0

    # Vapor generation rate
    rho_water = 1000.0
    m_dot_surface = evaporation_rate * rho_water

    A_surface = 4 * math.pi * R_current**2
    m_dot_total = m_dot_surface * A_surface

    # Vapor density at saturation
    P_sat = calculate_psat_tetens(T_droplet)
    T_k = T_droplet + 273.15
    rho_vapor = P_sat / (R_VAPOR * T_k) if R_VAPOR * T_k != 0 else 1.0

    Q_vapor = m_dot_total / rho_vapor

    v_superficial = Q_vapor / A_surface

    mu_vapor = calculate_vapor_viscosity(T_droplet)

    # Darcy's law
    Delta_P = (mu_vapor * L_shell * v_superficial) / kappa if kappa > 0 else 1e7

    # Safety cap
    Delta_P = min(Delta_P, 5e6)  # ~50 atm max

    P_internal = P_ambient + Delta_P

    return P_internal, Delta_P, v_superficial


def calculate_dimensionless_pressure(
    Delta_P: float,
    surface_tension: float,
    R_current: float
) -> float:
    """Π = ΔP / P_capillary (Laplace pressure = 2σ/R)"""
    if R_current <= 0:
        return 0.0
    P_cap = 2 * surface_tension / R_current
    return Delta_P / P_cap if P_cap > 0 else 0.0


def calculate_darcy_number(kappa: float, L_shell: float) -> float:
    """Da = κ / L_shell²"""
    if L_shell <= 0:
        return 1e10
    return kappa / L_shell**2


def predict_morphology_from_pressure(
    P_internal: float,
    P_ambient: float,
    Pi: float,
    Pe: float,
    Ma: float,
    shell_formed: bool,
    pe_values: Optional[Dict[str, float]] = None
) -> Tuple[str, str]:
    """
    Morphology prediction tree using pressure, Pe, Ma, and shell formation.
    """
    Delta_P = P_internal - P_ambient

    # Enhanced Pe values
    if pe_values:
        pe_eff = pe_values.get('effective_pe', Pe)
        pe_max = pe_values.get('max_pe', Pe)
        pe_int = pe_values.get('integrated_pe', Pe)
    else:
        pe_eff = pe_max = pe_int = Pe

    if not shell_formed:
        return "spheres", "No rigid shell formed"

    if pe_eff < 1.0:
        return "spheres", f"Diffusion-dominated drying (Pe_eff = {pe_eff:.2f} < 1)"

    if Pi < 0.1:
        return "spheres", f"Low pressure buildup (Π = {Pi:.2f} < 0.1)"

    if 0.1 <= Pi < 1.0:
        return "dimpled spheres", f"Moderate pressure causes dimpling (0.1 ≤ Π < 1.0)"

    if 1.0 <= Pi < 5.0:
        if pe_max > 10 or Ma > 1.0:
            return "donuts/cracked shells", f"High pressure + strong transport/Marangoni stresses (Π = {Pi:.2f}, Pe_max = {pe_max:.2f}, Ma = {Ma:.2f})"
        return "donuts", f"High pressure drives expansion (1.0 ≤ Π < 5.0)"

    # Pi >= 5.0
    if pe_int > 5:
        return "cracked shells", f"Excessive pressure + cumulative stress (Π = {Pi:.2f}, Pe_int = {pe_int:.2f})"
    return "cracked shells", f"Excessive pressure causes failure (Π ≥ 5.0)"

    return "unknown", "Insufficient data for prediction"


def calculate_complete_darcy_analysis(
    R_current: float,
    R_initial: float,
    solids_fraction: float,
    moisture: float,
    evaporation_rate: float,
    T_droplet: float,
    surface_tension: float,
    composition: Dict[str, float],
    Pe: float = 1.0,
    Ma: float = 0.5,
    shell_formed: bool = True,
    pe_values: Optional[Dict[str, float]] = None
) -> Dict:
    """
    Full Darcy pressure analysis — main entry point for simulation.py.
    Returns all key results in a dict for easy integration.
    """
    results = {}

    # Shell thickness
    L_shell = calculate_shell_thickness(R_current, R_initial, solids_fraction, moisture)
    results['shell_thickness_m'] = L_shell
    results['shell_thickness_um'] = L_shell * 1e6

    # Permeability & porosity
    kappa, porosity = estimate_shell_permeability(composition, moisture, T_droplet)
    results['permeability_m2'] = kappa
    results['porosity'] = porosity

    # Internal pressure
    P_internal, Delta_P, v_superficial = calculate_internal_pressure_darcy(
        R_current, L_shell, kappa, evaporation_rate, T_droplet, moisture
    )
    results['P_internal_Pa'] = P_internal
    results['Delta_P_Pa'] = Delta_P
    results['Delta_P_atm'] = Delta_P / 101325.0
    results['v_superficial_m_s'] = v_superficial

    # Dimensionless numbers
    Pi = calculate_dimensionless_pressure(Delta_P, surface_tension, R_current)
    results['Pi_pressure_ratio'] = Pi

    Da = calculate_darcy_number(kappa, L_shell)
    results['Darcy_number'] = Da

    # Morphology prediction
    morphology, mechanism = predict_morphology_from_pressure(
        P_internal, P_AMBIENT, Pi, Pe, Ma, shell_formed, pe_values
    )
    results['morphology_predicted'] = morphology
    results['morphology_mechanism'] = mechanism

    # Extra useful values
    results['P_capillary_Pa'] = 2 * surface_tension / R_current if R_current > 0 else 0.0
    results['pressure_buildup_relative'] = Delta_P / P_AMBIENT if P_AMBIENT > 0 else 0.0

    return results


# ────────────────────────────────────────────────
# Example / Test Usage
# ────────────────────────────────────────────────
if __name__ == "__main__":
    test_params = {
        'R_current': 2.5e-6,
        'R_initial': 5.0e-6,
        'solids_fraction': 0.10,
        'moisture': 0.05,
        'evaporation_rate': 1e-6,
        'T_droplet': 80.0,
        'surface_tension': 0.05,
        'composition': {'drug': 70, 'moni': 20, 'sugar': 10},
        'Pe': 12.0,
        'Ma': 0.5,
        'shell_formed': True,
        'pe_values': {'effective_pe': 6.0, 'max_pe': 15.0, 'integrated_pe': 8.0}
    }

    results = calculate_complete_darcy_analysis(**test_params)

    print("\n" + "="*60)
    print("DARCY PRESSURE ANALYSIS RESULTS")
    print("="*60)
    for key, value in results.items():
        print(f"{key:25}: {value}")
    print("="*60)