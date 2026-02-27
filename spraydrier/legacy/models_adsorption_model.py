# models/adsorption_model.py
import numpy as np

def predict_moisture_content(RH_out, T_outlet_C, solids_frac, feed_g_min, T1_C, calibration_factor=None):
    """
    Predict powder moisture content based on outlet RH and other process parameters.
    
    Uses a simplified GAB (Guggenheim-Anderson-de Boer) isotherm model for moisture sorption.
    
    Args:
        RH_out: Relative humidity at outlet (fraction, 0-1)
        T_outlet_C: Outlet temperature (Â°C)
        solids_frac: Solids fraction in feed
        feed_g_min: Feed rate (g/min)
        T1_C: Inlet temperature (Â°C)
        calibration_factor: Optional multiplicative calibration factor
    
    Returns:
        Predicted moisture content (fraction, kg water / kg dry solids)
    """
    if calibration_factor is None:
        calibration_factor = 1.0
    
    # Ensure RH is in valid range
    RH = max(0.0, min(RH_out, 0.95))
    
    # Simplified GAB isotherm parameters (typical for spray-dried powders)
    # These would ideally be material-specific
    C_gab = 10.0  # Related to enthalpy of sorption of first layer
    K_gab = 0.85  # Related to enthalpy of multilayer sorption
    
    # Monolayer moisture content (kg water / kg dry solid)
    # Temperature-dependent: decreases with temperature
    m_mono = 0.08 * np.exp(-0.005 * (T_outlet_C - 25))  # Typical range: 0.05-0.10
    
    # GAB equation
    if RH < 0.001:
        moisture = 0.0
    else:
        numerator = m_mono * C_gab * K_gab * RH
        denominator = (1 - K_gab * RH) * (1 - K_gab * RH + C_gab * K_gab * RH)
        
        if denominator > 0:
            moisture = numerator / denominator
        else:
            moisture = 0.0
    
    # Apply calibration factor
    moisture = moisture * calibration_factor
    
    # Clamp to reasonable range
    moisture = max(0.0, min(moisture, 0.30))  # Max 30% moisture
    
    return moisture


def adsorption_model(t, theta, V_initial, C_initial, k_ads, k_des):
    """
    ODE for surfactant adsorption kinetics.
    
    Models the time-evolution of surface coverage (theta) due to adsorption/desorption.
    
    Args:
        t: Time (s)
        theta: Current surface coverage (0-1)
        V_initial: Initial droplet volume (mÂ³)
        C_initial: Initial bulk concentration (kg/mÂ³)
        k_ads: Adsorption rate constant (1/s)
        k_des: Desorption rate constant (1/s)
    
    Returns:
        dtheta/dt: Rate of change of surface coverage
    """
    # Available sites for adsorption
    theta_available = max(0.0, 1.0 - theta)
    
    # Langmuir-type kinetics
    # Adsorption rate proportional to: (1) bulk concentration, (2) available sites
    # Desorption rate proportional to: occupied sites
    
    # Concentration factor (decreases as material is depleted to surface)
    C_factor = max(0.1, C_initial)  # Avoid zero
    
    rate_ads = k_ads * C_factor * theta_available
    rate_des = k_des * theta
    
    dtheta_dt = rate_ads - rate_des
    
    return dtheta_dt

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
    mw_da: float = 1226.0,           # PS80 â‰ˆ 1310, PS20 â‰ˆ 1226, average used
    gamma_max_mg_m2: float = 2.2,    # typical monolayer for polysorbates
    K_L_per_mol: float = 5e6        # tuned from literature (strong binding)
) -> float:
    """
    Langmuir isotherm for surfactant surface excess (mg/mÂ²).
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
    # Typical real-world observation: 0.01â€“0.1% bulk â†’ 40â€“90% surface
    if moni_conc_mg_ml < 0.5:
        coverage_pct = max(coverage_pct, 35.0)   # minimum realistic coverage
    elif moni_conc_mg_ml < 2.0:
        coverage_pct = max(coverage_pct, 60.0)

    return {
        "moni_surface_pct": round(coverage_pct, 2),
        "force_surfactant_override": True
    }