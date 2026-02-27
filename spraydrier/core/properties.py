# spraydrier/core/properties.py

import math
import json
from pathlib import Path
from scipy.optimize import brentq
import numpy as np
from typing import Dict, Any, List, Optional

# ────────────────────────────────────────────────
# Absolute imports (fixed for root-level execution)
# ────────────────────────────────────────────────
from spraydrier.core.constants import R_AIR, R_N2, R_v, C_p_water, C_p_vapor, R_UNIVERSAL, M_AIR, M_N2

# Load compound database
def load_compound_database() -> Dict:
    pkg_root = Path(__file__).resolve().parents[1]  # up to spraydrier/
    json_path = pkg_root / "data" / "compound_props.json"
    if not json_path.exists():
        print(f"Warning: compound_props.json not found at {json_path} - using defaults")
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading compound_props.json: {e}")
        return {}

COMPOUND_DB = load_compound_database()

def classify_compound(name: str, mw: float, conc_mg_ml: float = 0.0, is_drug: bool = False) -> Dict[str, Any]:
    """
    Classify compound and return properties.
    Priority: exact name match in compound_props.json > MW-based fallback.
    """
    name_lower = str(name).strip().lower()
    
    # 1. Exact match from database
    if name_lower in COMPOUND_DB:
        entry = COMPOUND_DB[name_lower]
        return {
            'class': entry.get('class', infer_class_from_mw(mw)),
            'tg': entry.get('tg', default_tg_from_mw(mw, is_drug)),
            'k': entry.get('k', 0.25 if mw > 50000 else 0.5),
            'specific_volume': entry.get('specific_volume', default_specific_volume(mw)),
            'mw': entry.get('mw', mw),
            'density': entry.get('density', 1.35 if mw > 50000 else 1.0)
        }
    
    # 2. MW-based fallback
    return {
        'class': infer_class_from_mw(mw),
        'tg': default_tg_from_mw(mw, is_drug),
        'k': 0.25 if mw > 50000 else 0.5,
        'specific_volume': default_specific_volume(mw),
        'mw': mw,
        'density': 1.35 if mw > 50000 else 1.0
    }

def infer_class_from_mw(mw: float) -> str:
    if mw > 50000:
        return 'protein'
    elif mw > 1000:
        return 'polymer'
    elif mw > 200:
        return 'sugar'
    else:
        return 'salt'

def default_specific_volume(mw: float) -> float:
    if mw > 50000:
        return 0.73
    elif mw > 1000:
        return 0.8
    elif mw > 200:
        return 0.6
    else:
        return 0.5

def default_tg_from_mw(mw: float, is_drug: bool = False) -> float | None:
    if is_drug or mw > 50000:
        return 160.0
    elif mw > 1000:
        return 150.0
    elif mw > 200:
        return 110.0
    else:
        return None

def calculate_psat_tetens(T_C: float) -> float:
    """
    Saturation vapor pressure of water using Tetens formula (valid 0–50°C).
    Fallback to Antoine for higher T.
    Returns Pa.
    """
    if T_C >= 50:
        # Antoine equation (mmHg → Pa)
        A, B, C = 8.07131, 1730.63, 233.426
        Psat_mmHg = 10 ** (A - B / (T_C + C))
        return Psat_mmHg * 133.322
    return 610.78 * math.exp(17.27 * T_C / (T_C + 237.3))

def calculate_wet_bulb_temperature(T_dry_C: float, RH: float, C_p_gas: float = 1005.0, h_vap: float = 2260e3) -> float:
    """
    Calculate wet-bulb temperature using psychrometric equation.
    Solves iteratively with brentq.
    """
    P_ambient = 101325.0
    P_v = (RH / 100) * calculate_psat_tetens(T_dry_C)
    
    def residual(T_wb_C):
        P_sat_wb = calculate_psat_tetens(max(T_wb_C, 0))
        return (P_sat_wb - P_v) - (C_p_gas * (T_dry_C - T_wb_C)) / (h_vap * 0.622)
    
    try:
        T_wb_C = brentq(residual, -10, T_dry_C, xtol=1e-5)
    except ValueError:
        T_wb_C = (T_dry_C + 0) / 2  # Fallback midpoint
    return T_wb_C

def fetch_gas_properties_from_table(gas: str, temperature_C: float) -> Dict:
    """
    Fetch gas properties (k_g, mu_g, Pr_g, D_v) from internal tables or approximation.
    """
    gas = gas.strip().lower()
    if gas == "air":
        # Approximate values (can be replaced with your table)
        k_g = 0.024 + 0.00007 * temperature_C
        mu_g = 1.716e-5 + 4.5e-8 * temperature_C
        Pr_g = 0.707
        D_v = 2.1e-5 + 1.5e-7 * temperature_C
        return {"R": R_AIR, "gamma": 1.4, "k_g": k_g, "mu_g": mu_g, "Pr_g": Pr_g, "D_v": D_v}
    elif gas in ("nitrogen", "n2"):
        return {"R": R_N2, "gamma": 1.33, "k_g": 0.024, "mu_g": 1.7e-5, "Pr_g": 0.71, "D_v": 2.0e-5}
    raise KeyError(f"Unsupported gas: {gas}")

def calculate_mixed_solution_density(compounds: List[Dict]) -> float:
    """
    Calculate density of mixed aqueous solution using volume additivity.
    compounds: list of {'conc_mg_ml': float, 'class': str}
    Returns g/cm³
    """
    if not compounds:
        return 1.0

    rho_water = 1.0
    total_solute_mass_g_ml = 0.0
    total_solute_volume_ml_ml = 0.0

    for comp in compounds:
        conc = comp.get('conc_mg_ml', 0.0) / 1000.0  # g/mL
        if conc <= 0:
            continue
        cls = comp.get('class', 'other')
        rho = 1.35 if cls == 'protein' else 1.59 if cls == 'sugar' else 1.0
        total_solute_mass_g_ml += conc
        total_solute_volume_ml_ml += conc / rho

    if total_solute_volume_ml_ml == 0:
        return rho_water

    volume_water = 1.0 - total_solute_volume_ml_ml
    mass_water = volume_water * rho_water
    total_mass = mass_water + total_solute_mass_g_ml
    total_volume = volume_water + total_solute_volume_ml_ml

    return total_mass / total_volume if total_volume > 0 else rho_water