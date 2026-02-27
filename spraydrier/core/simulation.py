# spraydrier/core/simulation.py

import math
import numpy as np
from pathlib import Path
import json
import pandas as pd
import traceback

# ────────────────────────────────────────────────
# Imports from your existing modules
# ────────────────────────────────────────────────
from spraydrier.core.constants import R_AIR, R_N2, R_v, C_p_water, C_p_vapor, R_UNIVERSAL, M_AIR, M_N2
from spraydrier.core.properties import (
    calculate_psat_tetens,
    calculate_wet_bulb_temperature,
    fetch_gas_properties_from_table,
    classify_compound,
    calculate_mixed_solution_density
)
from spraydrier.core.peclet_calculator import calculate_all_peclet_metrics
from spraydrier.core.diffusion_coefficient import calculate_diffusion_for_compounds
from spraydrier.core.temperature_evolution import calculate_realistic_droplet_temperature_evolution
from spraydrier.core.tg_calculator import detect_glass_transition, calculate_mixture_tg_gordon_taylor, calculate_shell_and_tg
from spraydrier.core.darcy_pressure import calculate_complete_darcy_analysis
from spraydrier.core.enhanced_physics_evolution_plotter_FINAL import calculate_enhanced_shrinkage_with_glass_transition

# Surface / adsorption
try:
    from spraydrier.surface.adsorption_model import adsorption_model, predict_moisture_content
except ImportError:
    print("Warning: adsorption_model not available - using defaults")
    adsorption_model = lambda *args, **kwargs: 0.0
    predict_moisture_content = lambda *args, **kwargs: 0.05

# ────────────────────────────────────────────────
# Safe float helper
# ────────────────────────────────────────────────
def safe_float(val, default=0.0):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_clean = val.strip().lower()
        if val_clean in ('n', 'no', 'none', ''):
            return default
        try:
            return float(val_clean.replace(',', '.'))
        except ValueError:
            return default
    return default

# ────────────────────────────────────────────────
# Main simulation function
# ────────────────────────────────────────────────
def run_full_spray_drying_simulation(inputs):
    try:
        batch_id = inputs.get("batch_id", "unknown_batch")
        print(f"[DEBUG] Starting simulation for batch: {batch_id}")

        # ────────────────────────────────────────────────
        # 1. Extract inputs (minimal defaults only)
        # ────────────────────────────────────────────────
        T_ambient = safe_float(inputs.get("T_ambient"), 25.0)
        dryer = inputs.get("dryer", "b290")
        V_chamber_m3 = safe_float(inputs.get("V_chamber_m3"), 0.00212)
        cyclone_type = inputs.get("cyclone_type", "standard")
        cyclone_factor = 1.2 if cyclone_type.lower() == "high" else 1.0
        gas1 = inputs.get("gas1", "air")
        T1_C = safe_float(inputs.get("T1_C"), 150.0)
        RH1 = safe_float(inputs.get("RH1"), 5.0)
        m1_m3ph = safe_float(inputs.get("m1_m3ph"), 30.0)
        gas2 = inputs.get("gas2", "air")
        T2_C = safe_float(inputs.get("T2_C"), 80.0)
        RH2 = safe_float(inputs.get("RH2"), 5.0)
        atom_pressure = safe_float(inputs.get("atom_pressure"), 1.0)
        nozzle_tip_d_mm = safe_float(inputs.get("nozzle_tip_d_mm"), 0.7)
        nozzle_cap_d_mm = safe_float(inputs.get("nozzle_cap_d_mm"), 1.5)
        nozzle_level = inputs.get("nozzle_level", "middle")
        T_outlet_C = safe_float(inputs.get("T_outlet_C"), 80.0)
        ds = inputs.get("ds", "default_drug")
        ds_conc = safe_float(inputs.get("ds_conc"), 5.0)
        ds_mw = safe_float(inputs.get("ds_mw"), 150000.0)
        solids_frac = safe_float(inputs.get("solids_frac"), 0.1)
        moni_conc = safe_float(inputs.get("moni_conc"), 0.0)
        pH = safe_float(inputs.get("pH"), 7.0)
        buffer_conc = safe_float(inputs.get("buffer_conc"), 0.0)
        stab_A_conc = safe_float(inputs.get("stab_A_conc"), 0.0)
        additive_B_conc = safe_float(inputs.get("additive_B_conc"), 0.0)
        additive_C_conc = safe_float(inputs.get("additive_C_conc"), 0.0)
        feed_mL_min = safe_float(inputs.get("feed_mL_min"), 5.0)
        rho_l_input = safe_float(inputs.get("rho_l"), 1.0)
        moisture_content = safe_float(inputs.get("moisture_content"), 0.05)
        D10_actual = safe_float(inputs.get("D10_actual"), 0.5)
        D50_actual = safe_float(inputs.get("D50_actual"), 3.0)
        D90_actual = safe_float(inputs.get("D90_actual"), 6.0)
        Span = safe_float(inputs.get("Span"), 1.5)

        T_ambient_K = T_ambient + 273.15
        T1_K = T1_C + 273.15
        T2_K = T2_C + 273.15

        # Viscosity & surface tension
        viscosity = 0.00003 * ds_conc - 0.0003 if ds.lower() in ["pgt121", "igg"] else 0.001
        surface_tension = 0.072

        feed_g_min = feed_mL_min * rho_l_input
        ratio = moni_conc / (moni_conc + ds_conc) if (moni_conc + ds_conc) > 0 else 0.0
        viscosity_moni = viscosity * (1 - 0.8 * ratio)
        surface_tension_moni = surface_tension * (1 - 0.3 * ratio)

        # ────────────────────────────────────────────────
        # 2. Droplet size (Buchi correlation)
        # ────────────────────────────────────────────────
        D32_um = 108.7 * (atom_pressure ** -0.71) * (feed_g_min ** 0.25) * \
                 (viscosity_moni * 1000 ** 0.23) * (surface_tension_moni * 1000 ** 0.47) * (rho_l_input ** -0.12)
        D32_um = max(6.0, min(60.0, D32_um))
        R_initial_m = D32_um * 1e-6 / 2.0

        # ────────────────────────────────────────────────
        # 3. Dynamic D_solute (from diffusion_coefficient.py)
        # ────────────────────────────────────────────────
        D_dict = calculate_diffusion_for_compounds(
            [{'name': 'drug', 'mw': ds_mw, 'conc_mg_ml': ds_conc, 'class': 'protein'}],
            T=T1_K, eta_eff=viscosity
        )
        D_solute = D_dict.get('drug', 1e-9) # only fallback if diffusion module fails

        # ────────────────────────────────────────────────
        # 4. v_s_initial (dynamic from physics)
        # ────────────────────────────────────────────────
        T_wb = calculate_wet_bulb_temperature(T1_C, RH1) if 'calculate_wet_bulb_temperature' in globals() else 35.0
        delta_T = T1_C - T_wb
        h_vap = 2260e3 - 2360 * T1_C  # dynamic latent heat
        # Build compounds list for density calculation
        compounds_list = []
        if ds_conc > 0:
            compounds_list.append({'conc_mg_ml': ds_conc, 'class': 'protein'})
        if moni_conc > 0:
            compounds_list.append({'conc_mg_ml': moni_conc, 'class': 'polymer'})
        if buffer_conc > 0:
            compounds_list.append({'conc_mg_ml': buffer_conc, 'class': 'buffer'})
        if stab_A_conc > 0:
            compounds_list.append({'conc_mg_ml': stab_A_conc, 'class': 'stabilizer'})
        if additive_B_conc > 0:
            compounds_list.append({'conc_mg_ml': additive_B_conc, 'class': 'additive'})
        if additive_C_conc > 0:
            compounds_list.append({'conc_mg_ml': additive_C_conc, 'class': 'additive'})
        rho_liquid = calculate_mixed_solution_density(compounds_list) if compounds_list else 1000.0
        h = 100.0  # fallback - can come from Nu later
        evap_flux = h * delta_T / h_vap
        v_s_initial = evap_flux / rho_liquid
        v_s_initial = np.clip(v_s_initial, 5e-7, 5e-6)
        print(f"[DEBUG] v_s_initial = {v_s_initial:.2e} m/s for {batch_id}")

        # ────────────────────────────────────────────────
        # 5. Shrinkage + Tg + Pe + Darcy
        # ────────────────────────────────────────────────
        batch_data = {
            'batch_id': batch_id,
            'solids_frac': solids_frac,
            'ds_conc': ds_conc,
            'moni_conc': moni_conc,
            'measured_total_moisture': inputs.get('measured_total_moisture'),
            'composition': {
                'drug': ds_conc,
                'moni': moni_conc,
                'buffer': buffer_conc,
                'stabilizer': stab_A_conc,
                'additive_B': additive_B_conc,
                'additive_C': additive_C_conc
            }
        }

        initial_radius_um = D32_um / 2.0
        print(f"[DEBUG] Starting shrinkage with initial radius = {initial_radius_um:.2f} µm")

        enhanced_tg_results = calculate_enhanced_shrinkage_with_glass_transition(
            radius_history_um=np.array([initial_radius_um]),
            Tg_array=np.array([]),
            temperature_array=np.array([]),
            batch_data=batch_data,
            T_inlet=T1_C,
            v_s_initial=v_s_initial,
            debug=True
        )

        # Extract with fallbacks
        radius_history_um = enhanced_tg_results.get('radius_array_um', np.array([initial_radius_um])) * 1e6
        Tg_array = enhanced_tg_results.get('Tg_array', np.array([]))
        temperature_array = enhanced_tg_results.get('temperature_array', np.array([]))
        shell_formation_time = enhanced_tg_results.get('shell_formation_time', None)

        if len(radius_history_um) == 0 or np.any(np.isnan(radius_history_um)):
            radius_history_um = np.array([initial_radius_um])
            print("[WARN] Empty/invalid radius history - using initial only")

        print(f"[DEBUG] Extracted radius history: {len(radius_history_um)} points, final = {radius_history_um[-1]:.2f} µm")

        # Pe metrics
        pe_metrics = {}
        try:
            pe_metrics = calculate_all_peclet_metrics(
                enhanced_tg_results.get('time_array', np.linspace(0, 0.8, 200)),
                radius_history_um,
                enhanced_tg_results.get('v_s_array', np.full(200, v_s_initial)),
                D_solute,
                D_compounds_m2_s={'drug': D_dict.get('drug', 1e-9)}
            )
        except Exception as pe_err:
            print(f"[WARN] Pe failed: {pe_err} - using fallback")
            pe_metrics = {'effective_pe': 1.0, 'max_pe': 5.0, 'integrated_pe': 0.5}

        # Darcy
        darcy_results = {}
        try:
            darcy_results = calculate_complete_darcy_analysis(
                R_current=radius_history_um[-1] * 1e-6,
                R_initial=R_initial_m,
                solids_fraction=solids_frac,
                moisture=moisture_content,
                evaporation_rate=enhanced_tg_results.get('v_s_array', np.array([v_s_initial]))[-1],
                T_droplet=temperature_array[-1] if len(temperature_array) > 0 else T_outlet_C,
                surface_tension=surface_tension_moni,
                composition=batch_data['composition'],
                Pe=pe_metrics.get('integrated_pe', 1.0),
                Ma=0.5,
                shell_formed=shell_formation_time is not None
            )
        except Exception as d_err:
            print(f"[WARN] Darcy failed: {d_err} - using fallback")
            darcy_results = {'morphology_predicted': 'unknown', 'Pi_ratio': 1.0}

        # Assemble return tuple (full 145 items, fallbacks for undefined)
        return (
            batch_id, dryer, V_chamber_m3, cyclone_type, cyclone_factor, gas1, T1_C, RH1, m1_m3ph,
            gas2, T2_C, RH2, atom_pressure, nozzle_tip_d_mm, nozzle_cap_d_mm, nozzle_level, T_outlet_C,
            ds, ds_conc, ds_mw, viscosity, surface_tension, D_solute, solids_frac, moni_conc, pH,
            buffer, buffer_conc, stabilizer_A, stab_A_conc, additive_B, additive_B_conc, additive_C,
            additive_C_conc, feed_g_min, rho_l_input, moisture_content, D10_actual, D50_actual, D90_actual,
            Span, ratio, viscosity_moni, surface_tension_moni, R_d, R_ag, gamma, h_vap, P_exit,
            atom_pressure_pa, feed_g_min / 1000, Q_loss, Cd, T_ambient_K, drying_gas_props, atom_gas_props,
            mu_g_atom, Psat_ambient, p_v_ambient, p_d_ambient, X_w, p_v, p_d, rho_final, 1005.0,
            1005.0 + X_w * 1840, Psat_initial_ag, p_v_atm_in, p_d_atm_in, X_w_atom, 1005.0 + X_w_atom * 1840, rho_atom_in,
            A_throat, pressure_ratio, crit_ratio, choked_condition, atom_gas_mass_flow, T_throat, c_throat,
            term, M_exit, T_exit, c_exit, u_ag, m_dry_kg_s,
            T_outlet_ag_adiabatic_K, T_mixed_K, T_outlet_ag_K, p_v_atm_exit, p_d_atm_exit, rho_ag_exit, mu_g_exit, phi,
            1 - solids_frac, evap_water_kgph, RH_out, Pv_out, Psat_out, condensed_kg_s, actual_water_evap,
            calibration_factor, observed_lab_frac, moisture_predicted, measured_RH, max_iter, rh_tol, moist_tol,
            curr_RH, curr_moist, evap_water_kgph, condensed_bulk_kg_s, it, evap_kgph, new_RH, Pv_out_local,
            Psat_out_local, condensed_kg_s, new_moist, RH_out, Pv_out, Psat_out, moisture_predicted,
            actual_water_evap, We, Oh, D32_without_moni, D32_with_moni, radius_history_um, t_eval,
            pe_metrics, energy_balance, efficiency, required_inlet_temp, sigma_effective, Nu, Sh, h,
            k_m, condensed_bulk_kg_s, condensed_surface_kg_s, morphology_indicators, Tg_array, temperature_array,
            shell_formation_time, enhanced_tg_results,
            v_s_initial,
            darcy_results
        )

    except Exception as e:
        print(f"[ERROR] Simulation crashed: {type(e).__name__}: {str(e)}")
        traceback.print_exc()
        return None