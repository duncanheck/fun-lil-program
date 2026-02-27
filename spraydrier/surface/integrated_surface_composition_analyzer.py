"""
Integrated Surface Composition Analyzer
========================================

Integrates with simulation.py to obtain comprehensive Péclet numbers, diffusion coefficients,
shell formation time, Darcy pressure, and Tg for all compounds. Predicts surface composition
using real physics (Pe, Darcy, Langmuir) and ML fallbacks.

Author: Doug Hecker 
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from typing import Dict, List, Optional
from tqdm import tqdm  # For progress bar

# Correct imports based on your folder structure
try:
    from spraydrier.core.simulation import run_full_spray_drying_simulation
    from spraydrier.core.properties import classify_compound, load_compound_database
    from spraydrier.surface.adsorption_model import adsorption_model  # Langmuir
    SIMULATION_AVAILABLE = True
    print("[DEBUG] Successfully imported simulation.py and dependencies")
except ImportError as e:
    print(f"Warning: Could not import simulation.py - {e}")
    SIMULATION_AVAILABLE = False

COMPOUND_DB = load_compound_database() if 'load_compound_database' in globals() else {}

class IntegratedSurfaceCompositionAnalyzer:
    def __init__(self):
        self.simulation_available = SIMULATION_AVAILABLE
        self.compound_db = COMPOUND_DB
        
        # Default fallbacks (only used if simulation completely fails)
        self.default_pe_values = {
            'drug': 1.0,
            'moni': 0.1,
            'buffer': 2.0,
            'stabilizer': 1.0,
            'additive': 1.5
        }
        self.default_d_values = {
            'drug': 1e-11,
            'moni': 5e-11,
            'buffer': 1e-9,
            'stabilizer': 5e-10,
            'additive': 2e-10
        }

    def run_simulation_for_trial(self, trial_data: pd.Series) -> Dict:
        """Run simulation.py for a single trial and extract physics outputs."""
        if not self.simulation_available:
            print("  Simulation not available, using defaults")
            return self.get_default_pe_d_values()

        try:
            # Map trial data to simulation inputs
            inputs = self.map_training_parameters(trial_data)

            # Run simulation
            result = run_full_spray_drying_simulation(inputs)

            if result is None or not isinstance(result, (tuple, dict)):
                print("  Simulation returned invalid type, using defaults")
                return self.get_default_pe_d_values()

            # Convert tuple to dict if needed
            if isinstance(result, tuple):
                param_names = [
                    'batch_id', 'dryer', 'V_chamber_m3', 'cyclone_type', 'cyclone_factor', 'gas1', 'T1_C', 'RH1', 'm1_m3ph',
                    'gas2', 'T2_C', 'RH2', 'atom_pressure', 'nozzle_tip_d_mm', 'nozzle_cap_d_mm', 'nozzle_level', 'T_outlet_C',
                    'ds', 'ds_conc', 'ds_mw', 'viscosity', 'surface_tension', 'D_solute', 'solids_frac', 'moni_conc', 'pH',
                    'buffer', 'buffer_conc', 'stabilizer_A', 'stab_A_conc', 'additive_B', 'additive_B_conc', 'additive_C',
                    'additive_C_conc', 'feed_g_min', 'rho_l', 'moisture_content', 'D10_actual', 'D50_actual', 'D90_actual',
                    'Span', 'ratio', 'viscosity_moni', 'surface_tension_moni', 'R_d', 'R_ag', 'gamma', 'h_vap', 'P_exit',
                    'atom_pressure_pa', 'feed_rate_kg_min', 'Q_loss', 'Cd', 'T_ambient_K', 'drying_gas_props', 'atom_gas_props',
                    'mu_g_atom', 'Psat_ambient', 'p_v_ambient', 'p_d_ambient', 'X_w', 'p_v', 'p_d', 'rho_final', 'C_p_dry_drying',
                    'C_p_humid_drying', 'Psat_initial_ag', 'p_v_atm_in', 'p_d_atm_in', 'X_w_atom', 'C_p_humid_atom', 'rho_atom_in',
                    'A_throat', 'pressure_ratio', 'crit_ratio', 'choked_condition', 'atom_gas_mass_flow', 'T_throat', 'c_throat',
                    'term', 'M_exit', 'T_exit', 'c_exit', 'u_ag', 'm_dry_kg_s', 'T_outlet_ag_adiabatic_K', 'T_mixed_K',
                    'T_outlet_ag_K', 'p_v_atm_exit', 'p_d_atm_exit', 'rho_ag_exit', 'mu_g_exit', 'phi', 'initial_moisture_content',
                    'evap_water_kgph', 'RH_out', 'Pv_out', 'Psat_out', 'condensed_kg_s', 'actual_water_evap', 'calibration_factor',
                    'observed_lab_frac', 'moisture_predicted', 'measured_RH', 'max_iter', 'rh_tol', 'moist_tol', 'curr_RH',
                    'curr_moist', 'evap_water_kgph', 'condensed_bulk_kg_s', 'it', 'evap_kgph', 'new_RH', 'Pv_out_local',
                    'Psat_out_local', 'condensed_kg_s', 'new_moist', 'RH_out', 'Pv_out', 'Psat_out', 'moisture_predicted',
                    'actual_water_evap', 'We', 'Oh', 'D32_without_moni', 'D32_with_moni', 'radius_history_um', 't_eval',
                    'pe_metrics', 'energy_balance', 'efficiency', 'required_inlet_temp', 'sigma_effective', 'Nu', 'Sh', 'h',
                    'k_m', 'condensed_bulk_kg_s', 'condensed_surface_kg_s', 'morphology_indicators', 'Tg_array', 'temperature_array',
                    'shell_formation_time', 'enhanced_tg_results'
                ]
                result_dict = dict(zip(param_names[:len(result)], result))
            else:
                result_dict = result

            # Extract key physics outputs (dynamic from simulation)
            pe_d_data = {
                'pe_drug': result_dict.get('effective_pe_drug', self.default_pe_values['drug']),
                'pe_moni': result_dict.get('effective_pe_moni', self.default_pe_values['moni']),
                'pe_stabilizer': result_dict.get('effective_pe_stabilizer', self.default_pe_values['stabilizer']),
                'pe_additive': result_dict.get('effective_pe_additive', self.default_pe_values['additive']),
                'pe_buffer': result_dict.get('effective_pe_buffer', self.default_pe_values['buffer']),
                'max_pe': result_dict.get('max_pe', 10.0),
                'integrated_pe': result_dict.get('integrated_pe', 5.0),
                'effective_pe': result_dict.get('effective_pe', 3.0),
                'evaporation_velocity': result_dict.get('v_s_array', [1e-5])[-1] if 'v_s_array' in result_dict else 1e-5,
                'droplet_radius': result_dict.get('radius_history_um', [5e-6])[0] * 1e-6 if 'radius_history_um' in result_dict else 5e-6,
                'temperature_c': result_dict.get('T_outlet_C', 70.0),
                'moisture_content': result_dict.get('moisture_predicted', 0.05),
                'rh_outlet': result_dict.get('RH_out', 15.0),
                'shell_formation_time': result_dict.get('shell_formation_time', 0.0),
                'darcy_pi': result_dict.get('darcy_Pi_ratio', 1.0),
                'darcy_delta_p': result_dict.get('darcy_Delta_P_Pa', 0.0)
            }

            print(f"[DEBUG surface] Extracted Pe/D data: {pe_d_data}")
            return pe_d_data

        except Exception as e:
            print(f"  Error running simulation: {e}")
            traceback.print_exc()
            return self.get_default_pe_d_values()

    def map_training_parameters(self, trial_data: pd.Series) -> Dict:
        """Map Excel row to simulation input format (robust handling)."""
        mapped = {}

        # Core process parameters (use .get for safety)
        mapped['T1_C'] = trial_data.get('Drying Gas Inlet (C)', 37.0)
        mapped['RH1'] = trial_data.get('RH1', 55.0)
        mapped['m1_m3ph'] = trial_data.get('Drying gas rate (m³/hr)', 35.0)
        mapped['T2_C'] = trial_data.get('T2_C', 22.0)
        mapped['RH2'] = trial_data.get('RH2', 20.0)
        mapped['atom_pressure'] = trial_data.get('atom_pressure', 2.0)
        mapped['feed_g_min'] = trial_data.get('Feed Rate (g/min)', 2.0)
        mapped['solids_frac'] = trial_data.get('%Solids', 0.1)
        mapped['moisture_content'] = trial_data.get('moisture_content', 0.05)

        # Formulation
        mapped['ds'] = trial_data.get('ds', 'BHV-1400')
        mapped['ds_conc'] = trial_data.get('Drug Substance conc. (mg/mL)', 100.0)
        mapped['ds_mw'] = trial_data.get('ds_mw', 150000.0)
        mapped['moni_conc'] = trial_data.get('Moni conc. (mg/mL)', 20.0)
        mapped['buffer'] = trial_data.get('buffer', 'glycine')
        mapped['buffer_conc'] = trial_data.get('buffer_conc', 10.0)
        mapped['stabilizer_A'] = trial_data.get('Stabilizer', 'PS80')
        mapped['stab_A_conc'] = trial_data.get('Stabilizer conc. (mg/mL)', 1.0)
        mapped['additive_B'] = trial_data.get('Additive #1', 'none')
        mapped['additive_B_conc'] = trial_data.get('Additive #1 conc. (mg/mL)', 0.0)

        # Physical properties
        mapped['viscosity'] = 'y' if pd.notna(trial_data.get('Estimated feed viscosity (Pa·s)')) else 'n'
        mapped['viscosity_user_input'] = trial_data.get('Estimated feed viscosity (Pa·s)', 0.001)
        mapped['surface_tension'] = 'y' if pd.notna(trial_data.get('Estimated Feed Surface Tension (N/m)')) else 'n'
        mapped['surface_tension_user_input'] = trial_data.get('Estimated Feed Surface Tension (N/m)', 0.0548)

        # Measured outcomes
        mapped['D50_actual'] = trial_data.get('D50_actual', 3.0)
        mapped['measured_RH_out'] = trial_data.get('measured_RH_out', None)

        # Apply compound-specific defaults
        ds_info = classify_compound(mapped['ds'], mapped['ds_mw'], mapped['ds_conc'], is_drug=True)
        mapped['ds_mw'] = ds_info['mw']

        return mapped

    def calculate_surface_composition(self, trial_data: pd.Series, pe_d_data: Dict) -> Dict:
        """Calculate surface composition using physics outputs."""
        components = {}
        total_conc = 0.0

        # Drug substance
        if trial_data.get('ds_conc', 0) > 0:
            ds_name = trial_data.get('ds', 'Drug')
            components[ds_name] = {
                'conc_mg_ml': trial_data['ds_conc'],
                'mw_da': trial_data.get('ds_mw', 150000),
                'pe_effective': pe_d_data.get('pe_drug', self.default_pe_values['drug']),
                'd_value': pe_d_data.get('d_drug', self.default_d_values['drug']),
                'type': 'drug'
            }
            total_conc += trial_data['ds_conc']

        # Moni
        if trial_data.get('moni_conc', 0) > 0:
            components['Moni'] = {
                'conc_mg_ml': trial_data['moni_conc'],
                'mw_da': 6800,
                'pe_effective': pe_d_data.get('pe_moni', self.default_pe_values['moni']),
                'd_value': pe_d_data.get('d_moni', self.default_d_values['moni']),
                'type': 'amphiphilic_polymer'
            }
            total_conc += trial_data['moni_conc']

        # Buffer
        if trial_data.get('buffer_conc', 0) > 0:
            components['Buffer'] = {
                'conc_mg_ml': trial_data['buffer_conc'],
                'mw_da': trial_data.get('buffer_mw', 1000),
                'pe_effective': pe_d_data.get('pe_buffer', self.default_pe_values['buffer']),
                'd_value': pe_d_data.get('d_buffer', self.default_d_values['buffer']),
                'type': 'buffer'
            }
            total_conc += trial_data['buffer_conc']

        # Stabilizer
        if trial_data.get('stab_A_conc', 0) > 0:
            components['Stabilizer'] = {
                'conc_mg_ml': trial_data['stab_A_conc'],
                'mw_da': trial_data.get('stab_mw', 5000),
                'pe_effective': pe_d_data.get('pe_stabilizer', self.default_pe_values['stabilizer']),
                'd_value': pe_d_data.get('d_stabilizer', self.default_d_values['stabilizer']),
                'type': 'stabilizer'
            }
            total_conc += trial_data['stab_A_conc']

        # Additive B & C (similar mapping)
        if trial_data.get('additive_B_conc', 0) > 0:
            components['Additive B'] = {
                'conc_mg_ml': trial_data['additive_B_conc'],
                'mw_da': trial_data.get('additive_B_mw', 500),
                'pe_effective': pe_d_data.get('pe_additive', self.default_pe_values['additive']),
                'd_value': pe_d_data.get('d_additive', self.default_d_values['additive']),
                'type': 'additive'
            }
            total_conc += trial_data['additive_B_conc']

        if trial_data.get('additive_C_conc', 0) > 0:
            components['Additive C'] = {
                'conc_mg_ml': trial_data['additive_C_conc'],
                'mw_da': trial_data.get('additive_C_mw', 500),
                'pe_effective': pe_d_data.get('pe_additive', self.default_pe_values['additive']),
                'd_value': pe_d_data.get('d_additive', self.default_d_values['additive']),
                'type': 'additive'
            }
            total_conc += trial_data['additive_C_conc']

        if not components:
            print("  No components with positive concentration - skipping")
            return None

        # Enrichment calculation (Langmuir + Pe)
        surface_fractions = {}
        for name, comp in components.items():
            pe_eff = comp['pe_effective']
            conc = comp['conc_mg_ml'] / 1000  # mg/mL → g/L
            # Langmuir surface excess (from adsorption_model.py)
            gamma = adsorption_model(conc, comp['mw_da'], comp['type'])
            # Pe enhancement (exp(-Pe) for depletion, but inverse for enrichment)
            enrichment = np.exp(-pe_eff) if pe_eff > 0 else 1.0  # Lower Pe → higher surface
            surface_fractions[name] = gamma * enrichment

        total_surface = sum(surface_fractions.values())
        if total_surface > 0:
            surface_fractions = {k: v / total_surface for k, v in surface_fractions.items()}

        print(f"[DEBUG surface] Calculated surface fractions: {surface_fractions}")

        return {
            'components': components,
            'surface_percentages': {k: v * 100 for k, v in surface_fractions.items()},
            'pe_d_data': pe_d_data
        }

    def analyze_doe_with_simulation(self, doe_file: str, output_dir: str = "./", output_filename: str = "surface_composition_results.xlsx"):
        print(f"🧪 INTEGRATED SURFACE ANALYSIS USING SIMULATION.PY")
        print(f"File: {doe_file}")

        df = pd.read_excel(doe_file)
        results = []

        for _, trial in tqdm(df.iterrows(), total=len(df), desc="Analyzing trials"):
            try:
                mapped = self.map_training_parameters(trial)
                pe_d = self.run_simulation_for_trial(mapped)
                surface = self.calculate_surface_composition(trial, pe_d)
                if surface:
                    result = {
                        'trial_id': trial.get('Trial_ID', 'unknown'),
                        **{f"{k}_surface_%": v for k, v in surface['surface_percentages'].items()},
                        **pe_d
                    }
                    results.append(result)
            except Exception as e:
                print(f"Error in trial: {e}")

        if results:
            pd.DataFrame(results).to_excel(os.path.join(output_dir, output_filename), index=False)
            print(f"Saved results to {output_filename}")

    def get_default_pe_d_values(self):
        return {**self.default_pe_values, **self.default_d_values}

def main():
    analyzer = IntegratedSurfaceCompositionAnalyzer()
    analyzer.analyze_doe_with_simulation("snapshot_mab_input (1)_results.xlsx")

if __name__ == "__main__":
    main()