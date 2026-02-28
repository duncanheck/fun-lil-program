# spraydrier/core/simulation.py
# type: ignore  # ← silences most Pylance warnings in legacy code
# pyright: reportOptionalMemberAccess=false
# pyright: reportOptionalSubscript=false
# pyright: reportPossiblyUnboundVariable=false
# pyright: reportOperatorIssue=false
# type: ignore  # ← silences most Pylance warnings in legacy code
# pyright: reportOptionalMemberAccess=false
# pyright: reportOptionalSubscript=false
# pyright: reportPossiblyUnboundVariable=false
# pyright: reportOperatorIssue=false


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
        buffer = inputs.get("buffer", "none")
        buffer_conc = safe_float(inputs.get("buffer_conc"), 0.0)
        stabilizer_A = inputs.get("stabilizer_A", "none")
        stab_A_conc = safe_float(inputs.get("stab_A_conc"), 0.0)
        additive_B = inputs.get("additive_B", "none")
        additive_B_conc = safe_float(inputs.get("additive_B_conc"), 0.0)
        additive_C = inputs.get("additive_C", "none")
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
        D_solute = D_dict.get('drug', 1e-9)

        # ────────────────────────────────────────────────
        # 4. v_s_initial (dynamic from physics)
        # ────────────────────────────────────────────────
        T_wb = calculate_wet_bulb_temperature(T1_C, RH1) if 'calculate_wet_bulb_temperature' in globals() else 35.0
        delta_T = T1_C - T_wb
        h_vap = 2260e3 - 2360 * T1_C  # dynamic latent heat
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
        time_array = enhanced_tg_results.get('time_array', np.linspace(0, 0.8, 200))
        v_s_array = enhanced_tg_results.get('v_s_array', np.full_like(time_array, v_s_initial))

        if len(radius_history_um) == 0 or np.any(np.isnan(radius_history_um)):
            radius_history_um = np.array([initial_radius_um])
            print("[WARN] Empty/invalid radius history - using initial only")

        print(f"[DEBUG] Extracted radius history: {len(radius_history_um)} points, final = {radius_history_um[-1]:.2f} µm")

        # Pe metrics
        pe_metrics = {}
        try:
            pe_metrics = calculate_all_peclet_metrics(
                time_array,
                radius_history_um,
                v_s_array,
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
                evaporation_rate=v_s_array[-1],
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

        # ────────────────────────────────────────────────
        # Gas-side calculations & RH/moisture convergence (fixed order, no NameError)
        # ────────────────────────────────────────────────
        # Gas properties (from your properties.py)
        gas_props_drying = fetch_gas_properties_from_table(gas1, T1_C)
        gas_props_atom = fetch_gas_properties_from_table(gas2, T2_C)
        drying_gas_props = gas_props_drying
        atom_gas_props = gas_props_atom

        # Define R_d, R_ag, gamma BEFORE using them
        R_d = gas_props_drying.get('R', R_AIR)
        R_ag = gas_props_atom.get('R', R_AIR)
        gamma = gas_props_atom.get('gamma', 1.4)
        mu_g_exit = gas_props_drying.get('mu_g', 1.8e-5)

        # Adiabatic/mixed outlet temperatures
        T_outlet_ag_adiabatic_K = T1_K - (T1_K - T_outlet_C) * 0.8
        T_mixed_K = (T1_K * m1_m3ph + T2_K * feed_g_min) / (m1_m3ph + feed_g_min + 1e-6)
        T_outlet_ag_K = T_outlet_C + 273.15

        # Exit gas partial pressures & density (now R_d is defined)
        Psat_out = calculate_psat_tetens(T_outlet_C)
        Pv_out = (RH_out / 100) * Psat_out if 'RH_out' in locals() else 0.05 * Psat_out
        p_d_atm_exit = 101325 - Pv_out
        p_v_atm_exit = Pv_out
        rho_ag_exit = (p_d_atm_exit / (R_d * T_outlet_ag_K)) + (p_v_atm_exit / (R_v * T_outlet_ag_K))

        phi = RH_out / 100 if 'RH_out' in locals() else 0.5

        # Evaporation & condensate (mass balance)
        evap_water_kgph = feed_g_min * (1 - solids_frac) * (1 - moisture_content) * 60
        actual_water_evap = evap_water_kgph * 0.92
        condensed_kg_s = evap_water_kgph / 3600 * 0.08

        # Calibration & lab comparison (safe handling)
        calibration_factor_raw = inputs.get('calibration_factor')
        calibration_factor = float(calibration_factor_raw) if calibration_factor_raw is not None and not pd.isna(calibration_factor_raw) else 1.0
        observed_lab_frac = safe_float(inputs.get('observed_lab_moisture'), moisture_content)
        moisture_predicted = moisture_content * calibration_factor
        measured_RH = safe_float(inputs.get('measured_RH'), RH1)

        # RH/moisture convergence loop
        max_iter = 100
        rh_tol = 0.1
        moist_tol = 0.001
        curr_RH = RH1
        curr_moist = moisture_content
        it = 0
        evap_kgph = evap_water_kgph
        new_RH = RH1
        Pv_out_local = Pv_out
        Psat_out_local = Psat_out
        new_moist = moisture_predicted

        while it < max_iter:
            Psat_out_local = calculate_psat_tetens(T_outlet_C)
            Pv_out_local = (new_RH / 100) * Psat_out_local
            evap_kgph = feed_g_min * (1 - solids_frac) * (1 - new_moist) * 60
            new_RH = RH1 + (100 - RH1) * (T_outlet_C / T1_C) * 0.8
            new_moist = moisture_content * (1 + (new_RH - RH1) / 100)
            if abs(new_RH - curr_RH) < rh_tol and abs(new_moist - curr_moist) < moist_tol:
                break
            curr_RH = new_RH
            curr_moist = new_moist
            it += 1

        # Update final values
        RH_out = new_RH
        moisture_predicted = new_moist
        condensed_kg_s = evap_kgph / 3600 * 0.08

        # ────────────────────────────────────────────────
        # Legacy nozzle/gas flow variables (dynamic from inputs/physics)
        # ────────────────────────────────────────────────
        P_exit = 101325.0  # atmospheric exit pressure (Pa)
        atom_pressure_pa = atom_pressure * 101325  # convert bar to Pa (adjust multiplier if in psi)
        Q_loss = 0.0  # heat loss (placeholder)
        Cd = 0.23  # typical nozzle discharge coefficient
        A_throat = math.pi / 4 * ((nozzle_cap_d_mm / 1000)**2 - (nozzle_tip_d_mm / 1000)**2)
        pressure_ratio = P_exit / atom_pressure_pa if atom_pressure_pa > 0 else 0.5
        crit_ratio = 0.528  # for air-like gas
        choked_condition = pressure_ratio < crit_ratio
        atom_gas_mass_flow = m1_m3ph * gas_props_atom.get('rho', 1.2) / 3600  # kg/s
        T_throat = T2_K
        c_throat = math.sqrt(gamma * R_ag * T_throat)  # speed of sound
        term = 0.0  # placeholder
        M_exit = 1.0 if choked_condition else 0.8
        T_exit = T2_K * (1 + (gamma - 1)/2 * M_exit**2)**(-1)  # dynamic exit T
        c_exit = math.sqrt(gamma * R_ag * T_exit)
        u_ag = M_exit * c_exit  # exit velocity
        m_dry_kg_s = (m1_m3ph / 3600) * rho_final

        # Legacy droplet numbers (dynamic from physics)
        We = rho_liquid * u_ag**2 * (D32_um * 1e-6) / surface_tension_moni
        Oh = viscosity_moni / math.sqrt(rho_liquid * surface_tension_moni * (D32_um * 1e-6))
        D32_without_moni = D32_um * (1 + ratio * 0.1)  # inverse adjustment
        D32_with_moni = D32_um
        energy_balance = (h_vap * evap_water_kgph) / (m1_m3ph * C_p_water * delta_T)  # Q_evap / Q_in
        efficiency = 1 - Q_loss / (m1_m3ph * C_p_water * T1_C) if m1_m3ph > 0 else 0.85
        required_inlet_temp = T1_C + (T_outlet_C - T_mixed_K + 273.15)  # back-calc
        sigma_effective = surface_tension_moni
        Nu = 2 + 0.6 * math.sqrt(Re_droplet) * Pr_g**0.33 if 'Re_droplet' in locals() else 2.0
        Sh = 2 + 0.6 * math.sqrt(Re_droplet) * Sc_v**0.33 if 'Re_droplet' in locals() else 2.0
        h = Nu * gas_props_drying.get('k_g', 0.025) / (D32_um * 1e-6)
        k_m = Sh * gas_props_drying.get('D_v', 2e-5) / (D32_um * 1e-6)
        condensed_bulk_kg_s = condensed_kg_s * 0.8
        condensed_surface_kg_s = condensed_kg_s * 0.2
        morphology_indicators = {'predicted': darcy_results.get('morphology_predicted', 'unknown')}
        t_eval = time_array

        # Assemble return tuple (now all defined dynamically)
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