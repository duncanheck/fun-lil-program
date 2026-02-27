# models/adsorption_model.py
import numpy as np
def adsorption_model(*args, **kwargs):
    return 0.0
def predict_moisture_content(*args, **kwargs):
    return 0.05
def is_surfactant(compound_name: str) -> bool:
    """Return True if the compound is known to be surface-active."""
    surfactants = {
        'moni', 'ps80', 'ps20', 'polysorbate 80', 'polysorbate 20',
        'poloxamer', 'pluronic', 'tween', 'brij', 'span'
    }
    if not compound_name:
        return False
    name_lower = str(compound_name).strip().lower()
    return any(s in name_lower for s in surfactants)


def langmuir_surface_excess_mg_m2(
    bulk_conc_mg_ml: float,
    mw_da: float = 1226.0,           # PS80 ≈ 1310, PS20 ≈ 1226, average used
    gamma_max_mg_m2: float = 2.2,    # typical monolayer for polysorbates
    K_L_per_mol: float = 5e6        # tuned from literature (strong binding)
) -> float:
    """
    Langmuir isotherm for surfactant surface excess (mg/m²).
    Returns mass-based surface excess.
    """
    if bulk_conc_mg_ml <= 0 or mw_da <= 0:
        return 0.0

    c_mol_L = (bulk_conc_mg_ml / mw_da) * 1000.0
    gamma = gamma_max_mg_m2 * (K_L_per_mol * c_mol_L) / (1.0 + K_L_per_mol * c_mol_L)
    return float(gamma)


def estimate_surface_coverage_pct(
    gamma_mg_m2: float,
    gamma_max_mg_m2: float = 2.2
) -> float:
    """Convert surface excess to approximate % of monolayer."""
    coverage = gamma_mg_m2 / gamma_max_mg_m2 * 100.0
    return min(coverage, 99.9)   # cap at nearly full monolayer


def surfactant_priority_override(
    moni_conc_mg_ml: float,
    moni_name: str = None,
    moni_mw: float = None
) -> dict:
    """
    Returns a dict with surface percentages when a surfactant is present.
    This completely overrides the diffusive Peclet model for that component.
    """
    if not is_surfactant(moni_name) or moni_conc_mg_ml <= 0:
        return {}

    gamma = langmuir_surface_excess_mg_m2(
        bulk_conc_mg_ml=moni_conc_mg_ml,
        mw_da=moni_mw or 1226.0
    )
    coverage_pct = estimate_surface_coverage_pct(gamma)

    # Even at very low bulk concentration, polysorbates dominate the interface
    # Typical real-world observation: 0.01–0.1% bulk → 40–90% surface
    if moni_conc_mg_ml < 0.5:
        coverage_pct = max(coverage_pct, 35.0)   # minimum realistic coverage
    elif moni_conc_mg_ml < 2.0:
        coverage_pct = max(coverage_pct, 60.0)

    return {
        "moni_surface_pct": round(coverage_pct, 2),
        "force_surfactant_override": True
    }