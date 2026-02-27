# main.py - Modified with Calibration/Production Mode Selection

from simulation import run_full_spray_drying_simulation

# Parameter names for simulation results (70 elements based on return tuple - UPDATED to include est_T_outlet_C, shell_formation_time, and Tg values)
SIMULATION_PARAM_NAMES = [
    'batch_id', 'dryer', 'cyclone_type', 'gas1', 'gas2', 'T1_C', 'T_inlet_req_C', 'T_outlet_C',
    'atom_pressure', 'atom_gas_mass_flow', 'GLR', 'u_ag', 'solids_frac', 'viscosity_moni', 'surface_tension_moni', 'rho_final',
    't_dry', 'v_evap', 'Pe', 'Ma', 'Re_droplet', 'Re_g', 'Nu', 'Sh', 'D10_actual', 'D50_actual', 'D90_actual', 'D50_calc', 'Span', 'RH_out', 'h', 'k_m', 'D_solute',
    'moisture_content', 'ds', 'feed_mL_min', 'feed_g_min', 'rho_v_droplet', 'Efficiency', 'pH', 'buffer', 'stabilizer_A', 'stab_A_conc', 'additive_B', 'additive_B_conc', 'additive_C', 'additive_C_conc',
    'm1_m3ph', 'buffer_conc', 'D32_um_moni', 'D50_calc_moni', 'nozzle_tip_d_mm', 'ds_conc', 'moni_conc', 'moisture_predicted', 'moisture_input', 'condensed_total_kg_ph', 'calibration_factor', 'est_RH_pct', 'delta_RH_pct', 'spm', 'bmp', 'measured_total_computed',
    # === Péclet metrics ===
    'effective_pe', 'max_pe', 'integrated_pe',
    'effective_pe_drug', 'effective_pe_moni', 'effective_pe_buffer', 'effective_pe_stabilizer', 'effective_pe_additive_B', 'effective_pe_additive_C',
    'max_pe_drug', 'max_pe_moni', 'max_pe_buffer', 'max_pe_stabilizer', 'max_pe_additive_B', 'max_pe_additive_C',
    'integrated_pe_drug', 'integrated_pe_moni', 'integrated_pe_buffer', 'integrated_pe_stabilizer', 'integrated_pe_additive_B', 'integrated_pe_additive_C',
    # === D values per component ===
    'D_drug', 'D_moni', 'D_buffer', 'D_stabilizer', 'D_additive_B', 'D_additive_C',
    # === Darcy results ===
    'darcy_P_internal_Pa', 'darcy_Delta_P_Pa', 'darcy_Pi_ratio', 'darcy_morphology_predicted', 'darcy_morphology_mechanism', 'darcy_shell_thickness_um', 'darcy_permeability_m2', 'darcy_Darcy_number',
    # === Tg results ===
    'Final_Tg_C', 'Tg_drug', 'Tg_moni', 'Tg_buffer', 'Tg_stabilizer', 'Tg_additive_B', 'Tg_additive_C',
    # === Shell formation ===
    'shell_formation_time', 'shell_formation_fraction'
]


# Preferred order of display labels (match your Snapshot_mab_output.xlsx as closely as possible)
preferred_order = [
    'Parameter / Batch ID',
    'batch_id',
    'ds',
    'dryer',
    'V_chamber_m3',
    'cyclone_type',
    'gas1',
    'Drying Gas Inlet (C)',
    'RH1',
    'gas2',
    'T2_C',
    'RH2',
    'atom_pressure',
    'nozzle_tip_d_mm',
    'nozzle_cap_d_mm',
    'nozzle_level',
    'T_outlet_C',
    'measured_RH_out',
    'D_solute',
    '%Solids',
    'Feed Rate (g/min)',
    'feed_mL_min',
    'rho_l',
    'Drug Substance conc. (mg/mL)',
    'Moni conc. (mg/mL)',
    'Feed solution pH',
    'buffer',
    'buffer_conc',
    'Stabilizer',
    'Stabilizer conc. (mg/mL)',
    'Additive #1',
    'Additive #1 conc. (mg/mL)',
    'D10_actual',
    'D50_actual',
    'D90_actual',
    'Span',
    'measured_surface_moisture',
    'measured_bound_moisture',
    'Measured total moisture (%)',
    'D32 diameter (With Moni)',
    'D50 calculated (with Moni)',
    'Predicted powder moisture content (%)',
    'Required Inlet Temperature',
    'est_T_outlet_C',
    'est_RH_pct',
    'atom_gas_mass_flow',
    'GLR',
    'u_ag',
    'Estimated feed viscosity (Pa·s)',
    'Estimated Feed Surface Tension (N/m)',
    'rho_final',
    't_dry',
    'Surface recession velocity',
    'Marangoni Number',
    'Reynolds number',
    'Re_g',
    'Nusselt number Nu',
    'Sherwood number Sh',
    'D50 calculated (No Moni)',
    'Heat Transfer coefficient',
    'Mass Transfer coefficient',
    'rho_v_droplet',
    'Efficiency',
    'calibration_factor',
    'D (Drug)',
    'D (Moni)',
    'D (Buffer)',
    'D (Stabilizer)',
    'D (Additive B)',
    'D (Additive C)',
    'Effective Pe (Drug)',
    'Effective Pe (Moni)',
    'Effective Pe (Buffer)',
    'Effective Pe (Stabilizer)',
    'Effective Pe (Additive B)',
    'Effective Pe (Additive C)',
    'Max Pe (Drug)',
    'Max Pe (Moni)',
    'Max Pe (Buffer)',
    'Max Pe (Stabilizer)',
    'Max Pe (Additive B)',
    'Max Pe (Additive C)',
    'Integrated Pe (Drug)',
    'Integrated Pe (Moni)',
    'Integrated Pe (Buffer)',
    'Integrated Pe (Stabilizer)',
    'Integrated Pe (Additive B)',
    'Integrated Pe (Additive C)',
    'Final Tg (Powder)',
    'Tg (Drug)',
    'Tg (Moni)',
    'Tg (Buffer)',
    'Tg (Stabilizer)',
    'Tg (Additive B)',
    'Tg (Additive C)',
    'Darcy Internal Pressure (Pa)',
    'Darcy Pressure Drop (Pa)',
    'Darcy Pi Ratio',
    'Darcy Shell Thickness (μm)',
    'Darcy Permeability (m²)',
    'Darcy Number',
    'Shell Formation Time (ms)',
    'Shell Formation Fraction',
    'Morphology (known)',
    'Predicted Morphology (ML)',
    'Morphology Confidence',
    'Darcy Predicted Morphology',
    'Darcy Morphology Mechanism'
]
preferred_order_internal = [
    'header_row',
    'batch_id',
    'ds',
    'dryer',
    'V_chamber_m3',
    'cyclone_type',
    'gas1',
    'T1_C',
    'RH1',
    'gas2',
    'T2_C',
    'RH2',
    'atom_pressure',
    'nozzle_tip_d_mm',
    'nozzle_cap_d_mm',
    'nozzle_level',
    'T_outlet_C',
    'measured_RH_out',
    'D_solute',
    'solids_frac',
    'feed_g_min',
    'feed_mL_min',
    'rho_l',
    'ds_conc',
    'moni_conc',
    'pH',
    'buffer',
    'buffer_conc',
    'stabilizer_A',
    'stab_A_conc',
    'additive_B',
    'additive_B_conc',
    'D10_actual',
    'D50_actual',
    'D90_actual',
    'Span',
    'spm',
    'bmp',
    'measured_total_computed',
    'D32_um_moni',
    'D50_calc_moni',
    'moisture_predicted',
    'T_inlet_req_C',
    'est_T_outlet_C',
    'est_RH_pct',
    'atom_gas_mass_flow',
    'GLR',
    'u_ag',
    'viscosity_moni',
    'surface_tension_moni',
    'rho_final',
    't_dry',
    'v_evap',
    'Ma',
    'Re_droplet',
    'Re_g',
    'Nu',
    'Sh',
    'D50_calc',
    'h',
    'k_m',
    'rho_v_droplet',
    'Efficiency',
    'calibration_factor',
    'D_drug',
    'D_moni',
    'D_buffer',
    'D_stabilizer',
    'D_additive_B',
    'D_additive_C',
    'effective_pe_drug',
    'effective_pe_moni',
    'effective_pe_buffer',
    'effective_pe_stabilizer',
    'effective_pe_additive_B',
    'effective_pe_additive_C',
    'max_pe_drug',
    'max_pe_moni',
    'max_pe_buffer',
    'max_pe_stabilizer',
    'max_pe_additive_B',
    'max_pe_additive_C',
    'integrated_pe_drug',
    'integrated_pe_moni',
    'integrated_pe_buffer',
    'integrated_pe_stabilizer',
    'integrated_pe_additive_B',
    'integrated_pe_additive_C',
    'Final_Tg_C',
    'Tg_drug',
    'Tg_moni',
    'Tg_buffer',
    'Tg_stabilizer',
    'Tg_additive_B',
    'Tg_additive_C',
    'darcy_P_internal_Pa',
    'darcy_Delta_P_Pa',
    'darcy_Pi_ratio',
    'darcy_shell_thickness_um',
    'darcy_permeability_m2',
    'darcy_Darcy_number',
    'shell_formation_time',
    'shell_formation_fraction',
    'morphology_known',
    'morphology_predicted',
    'morphology_confidence',
    'darcy_morphology_predicted',
    'darcy_morphology_mechanism'
]


import sys
import numpy as np
import pandas as pd
import scipy
import argparse
import os

# ========================================================================
# MODE SELECTION FUNCTION
# ========================================================================

def get_run_mode():
    """
    Determine if this is a calibration run or production run with ML models.
    
    Returns:
        str: 'calibration' or 'production'
    """
    print("\n" + "="*80)
    print("SPRAY DRYING SIMULATION - MODE SELECTION")
    print("="*80)
    print("\n1. CALIBRATION MODE (Training Data Generation)")
    print("   - Full physics calculations (Pe, Ma, Darcy, etc.)")
    print("   - No morphology PREDICTION (no trained model to use)")
    print("   - Output used by learn_calibration.py to train ML models")
    print("   - Ideal for initial data generation or model retraining")
    print("   - Files needed: None (physics only, no ML)")
    print("\n2. PRODUCTION MODE (With ML Predictions)")
    print("   - Full physics calculations")
    print("   - PLUS morphology predictions using trained model")
    print("   - Requires: calibration.json, morphology_model.pkl")
    print("   - Optional: compound_props.json for MW lookups")
    print("="*80)
    
    while True:
        mode = input("\nSelect mode (1=Calibration, 2=Production, or press Enter for Production): ").strip()
        if mode == '1':
            return 'calibration'
        elif mode == '2' or mode == '':
            return 'production'
        else:
            print("Invalid choice. Please enter 1 for Calibration or 2 for Production.")

# ========================================================================
# ARGUMENT PARSING
# ========================================================================

parser = argparse.ArgumentParser(
    description='Spray Drying Simulation - Physics-based particle formation model',
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument('--debug', action='store_true', 
                    help='Show startup diagnostics')
parser.add_argument('--excel', type=str, 
                    help='Excel file containing input parameters (first column: parameter names, subsequent columns: values for each batch)')
parser.add_argument('--plot', action='store_true',
                    help='Legacy option: plotting is now automatic after simulation completion')
parser.add_argument('--batch-id', type=str,
                    help='Legacy option: specific batch ID to plot (only used with --plot flag)')
parser.add_argument('--mode', type=str, choices=['calibration', 'production'], 
                    default=None,
                    help='Run mode: calibration (training data generation) or production (with ML predictions). If not specified, will prompt user.')
args, _unknown = parser.parse_known_args()

if args.debug:
    print(f"Debug: Python version: {sys.version}")
    print(f"Debug: NumPy version: {np.__version__}")
    print(f"Debug: Pandas version: {pd.__version__}")
    print(f"Debug: SciPy version: {scipy.__version__}")
    import pathlib, time, sys as _sys
    try:
        _this_path = pathlib.Path(__file__).resolve()
        print(f"Debug: Running main.py from: {_this_path}")
        print(f"Debug: main.py last modified: {time.ctime(_this_path.stat().st_mtime)}")
        _sys.stdout.flush()
    except Exception:
        pass

# ========================================================================
# DETERMINE RUN MODE
# ========================================================================

if args.mode:
    RUN_MODE = args.mode
    print(f"\n*** Running in {RUN_MODE.upper()} MODE (specified via --mode argument) ***\n")
else:
    RUN_MODE = get_run_mode()
    print(f"\n*** Running in {RUN_MODE.upper()} MODE ***\n")

# Print mode-specific information
if RUN_MODE == 'calibration':
    print("CALIBRATION MODE: Full physics calculations (Pe, Ma, Darcy, etc.)")
    print("Morphology prediction DISABLED (no trained model to use yet)")
    print("Output suitable for learn_calibration.py training\n")
else:
    print("PRODUCTION MODE: Full physics + ML morphology predictions")
    print("Required files: calibration.json, morphology_model.pkl, compound_props.json")
    print("If these files are missing, predictions will use defaults or fail gracefully.\n")

import traceback
try:
    from input_utils import collect_inputs
except Exception as e:
    print("Failed to import collect_inputs from input_utils:")
    import importlib, inspect, pathlib
    try:
        spec = importlib.util.find_spec('input_utils')
        if spec and spec.origin:
            print(f"input_utils module path: {spec.origin}")
            try:
                print('--- head of input_utils.py ---')
                print(pathlib.Path(spec.origin).read_text().splitlines()[:40])
            except Exception:
                pass
    except Exception:
        pass
    traceback.print_exc()
    raise

# ========================================================================
# MORPHOLOGY PREDICTION FUNCTION (PRODUCTION MODE ONLY)
# ========================================================================

def predict_morphology(inputs, results):
    """
    Predict morphology using trained Random Forest model.
    ONLY CALLED IN PRODUCTION MODE.
    
    Args:
        inputs: Input parameters dictionary
        results: Simulation results dictionary
        
    Returns:
        dict: Dictionary with 'prediction' (str) and 'confidence' (float) keys
    """
    import pickle
    import pandas as pd
    import numpy as np
    
    try:
        # Load the trained morphology model
        with open('morphology_model.pkl', 'rb') as f:
            model_dict = pickle.load(f)
        
        model = model_dict['model']
        label_encoder = model_dict['label_encoder']
        features = model_dict['features']
        
        # Create feature vector from inputs and results
        feature_data = {}
        
        # Map available inputs/results to expected features
        feature_mapping = {
            'Drying Gas Inlet (C)': inputs.get('Drying Gas Inlet (C)', inputs.get('T1_C', 0)),
            'Drying gas rate (m³/hr)': inputs.get('Drying gas rate (m³/hr)', inputs.get('m1_m3ph', 0)),
            'Drug Substance conc. (mg/mL)': inputs.get('Drug Substance conc. (mg/mL)', inputs.get('ds_conc', 0)),
            'Moni conc. (mg/mL)': inputs.get('Moni conc. (mg/mL)', inputs.get('moni_conc', 0)),
            'Feed solution pH': inputs.get('Feed solution pH', inputs.get('pH', 0)),
            'Stabilizer conc. (mg/mL)': inputs.get('Stabilizer conc. (mg/mL)', inputs.get('stab_A_conc', 0)),
            'Additive #1 conc. (mg/mL)': inputs.get('Additive #1 conc. (mg/mL)', inputs.get('additive_B_conc', 0)),
            '%Solids': inputs.get('%Solids', inputs.get('solids_frac', 0) * 100 if inputs.get('solids_frac') is not None else 0),
            'D50_actual': inputs.get('D50_actual', 0),
            'D10_actual': inputs.get('D10_actual', 0),
            'D90_actual': inputs.get('D90_actual', 0),
            'Feed Rate (g/min)': inputs.get('Feed Rate (g/min)', inputs.get('feed_g_min', 0)),
            'Surface recession velocity': results.get('Surface recession velocity', results.get('v_evap', 0)),
            'Max Peclet Number': results.get('Max Peclet Number', results.get('Pe', 0)),
            'Effective Peclet Number': results.get('Effective Peclet Number', results.get('Pe', 0)),
            'Peclet Number': results.get('Peclet Number', results.get('Pe', 0)),
            'Integrated Peclet Number': results.get('Integrated Peclet Number', results.get('Pe', 0)),
            'Reynolds number': results.get('Reynolds number', results.get('Re_droplet', 0)),
            'Marangoni Number': results.get('Marangoni Number', results.get('Ma', 0)),
            'Estimated feed viscosity (Pa·s)': inputs.get('Estimated feed viscosity (Pa·s)', 
                                                        inputs.get('viscosity_user_input', 
                                                                   inputs.get('viscosity_moni', 0))),
            'Estimated Feed Surface Tension (N/m)': inputs.get('Estimated Feed Surface Tension (N/m)', 
                                                              inputs.get('surface_tension_user_input', 
                                                                         inputs.get('surface_tension_moni', 0))),
            'Measured total moisture (%)': inputs.get('Measured total moisture (%)', 
                                                     inputs.get('measured_total_moisture', 0)),
            'moisture_predicted': results.get('moisture_predicted', 
                                                                     results.get('moisture_predicted', 0) * 100 
                                                                     if results.get('moisture_predicted') is not None else 0),
            # Per-component effective Peclet numbers – default to 0 when missing
            'effective_pe_moni': results.get('effective_pe_moni', 0),
            'effective_pe_drug': results.get('effective_pe_drug', 0),
            'effective_pe_stabilizer': results.get('effective_pe_stabilizer', 0),
            'effective_pe_additive_b': results.get('effective_pe_additive_b', 0),
            'effective_pe_buffer': results.get('effective_pe_buffer', 0),
            # Darcy features that the model was trained on
            'Darcy Internal Pressure (Pa)': results.get('darcy_P_internal_Pa', 0),
            'Darcy Number': results.get('darcy_Darcy_number', 0),
            'Darcy Permeability (m²)': results.get('darcy_permeability_m2', 0),
            'Darcy Pi Ratio': results.get('darcy_Pi_ratio', 0),
            'Darcy Pressure Drop (Pa)': results.get('darcy_Delta_P_Pa', 0),
            'Darcy Shell Thickness (μm)': results.get('darcy_shell_thickness_um', 0),
            'Shell_Formation_Time_ms': (results.get('shell_formation_time', results.get('t_dry', 0.1) * 0.3) * 1000),
            'Chamber_Time_ms': results.get('t_dry', 0.1) * 1000,
            'Final_Tg_C': results.get('Final_Tg_C', 55.0),
            'T_droplet_at_exit_C': results.get('T_droplet_at_exit_C', results.get('T_outlet_C', 80) + 5),
            'T_inlet_gas_C': results.get('T1_C', inputs.get('T1_C', 140)),
            'T_outlet_C': results.get('T_outlet_C', 80),
            'Temperature_Drop_C': (results.get('T1_C', 140) - results.get('T_outlet_C', 80)),
            'Delta_T_minus_Tg_C': (results.get('T_droplet_at_exit_C', 85) - results.get('Final_Tg_C', 55)),
            'Cooling_Rate_C_per_ms': ((results.get('T1_C', 140) - results.get('T_outlet_C', 80)) / (results.get('t_dry', 0.1) * 1000 + 1e-6)),
            'Shell_Formation_Fraction': ((results.get('shell_formation_time', results.get('t_dry', 0.1) * 0.3) * 1000) / (results.get('t_dry', 0.1) * 1000 + 1e-6)),
            'buffer_conc': inputs.get('buffer_conc', 0),
            'nozzle_cap_d_mm': inputs.get('nozzle_cap_d_mm', 0),
            'nozzle_tip_d_mm': inputs.get('nozzle_tip_d_mm', 0)
        }
        
        # CRITICAL SAFETY NET: Replace any accidental None with 0
        feature_mapping = {k: (v if v is not None else 0) for k, v in feature_mapping.items()}
        
        # Build feature vector in correct order
        feature_vector = []
        for feature in features:
            value = feature_mapping.get(feature, 0)  # Default to 0 if feature not available
            feature_vector.append(float(value))
        
        # Convert to DataFrame for prediction
        feature_df = pd.DataFrame([feature_vector], columns=features)
        
        # DEBUG: Print feature values for first batch
        if not hasattr(predict_morphology, 'debug_printed'):
            print("\n=== MORPHOLOGY PREDICTION DEBUG ===")
            print("Raw results values:")
            print(f"  results.get('T1_C'): {results.get('T1_C')}")
            print(f"  results.get('T_outlet_C'): {results.get('T_outlet_C')}")
            print(f"  results.get('Temperature_Drop_C'): {results.get('Temperature_Drop_C')}")
            print(f"  results.get('darcy_P_internal'): {results.get('darcy_P_internal')}")
            print(f"  results.get('Final_Tg_C'): {results.get('Final_Tg_C')}")
            print("Feature values being passed to model:")
            for i, (feature, value) in enumerate(zip(features, feature_vector)):
                print(f"{feature}: {value}")
            print("===================================\n")
            predict_morphology.debug_printed = True
        
        # Make prediction
        prediction_encoded = model.predict(feature_df)[0]
        prediction_decoded = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities
        probabilities = model.predict_proba(feature_df)[0]
        
        # Map the predicted encoded class to the model's internal class index
        try:
            model_class_index = list(model.classes_).index(prediction_encoded)
            confidence = float(probabilities[model_class_index])
        except (ValueError, IndexError):
            # Fallback if mapping fails
            confidence = 0.5
        
        return {
            'prediction': prediction_decoded,
            'confidence': confidence
        }
        
    except Exception as e:
        print(f"Morphology prediction error: {e}")
        return {
            'prediction': "unknown",
            'confidence': 0.0
        }

try:
    from simulation import run_full_spray_drying_simulation
    from output import save_output
except Exception:
    print("Failed to import simulation or output module")
    traceback.print_exc()
    raise

import json
import os

def load_inputs_from_excel(filename):
    """Load inputs from Excel file for all columns starting from column 2 (Value.1)."""
    try:
        df = pd.read_excel(filename, header=None)
        
        if df.iloc[0, 0] in ['Parameter / Batch ID', 'Parameter']:
            # === NEW: Remove accidental duplicate headers at the bottom ===
            header_rows = df[df.iloc[:, 0] == 'Parameter / Batch ID'].index.tolist()
            if len(header_rows) > 1:
                print(f"Warning: Found {len(header_rows)} occurrences of 'Parameter / Batch ID'. Removing {len(header_rows)-1} duplicate(s) at the bottom.")
                df = df.drop(header_rows[1:])   # keep only the first one
            # ===========================================================

            # Transposed: rows are parameters, columns are batches
            params = df.iloc[1:, 0].values
            batches_data = df.iloc[1:, 1:].values
            num_batches = batches_data.shape[1]
            inputs_list = []
            for col_idx in range(num_batches):
                batch = dict(zip(params, batches_data[:, col_idx]))
                
                # Calculate D_solute if not provided
                print(f"[D_solute] Checking: 'D_solute' in batch={('D_solute' in batch)}, value={batch.get('D_solute')}, is None={batch.get('D_solute') is None}, pd.isna={pd.isna(batch.get('D_solute'))}")
                if 'D_solute' not in batch or batch['D_solute'] is None or pd.isna(batch['D_solute']):
                    try:
                        print("[D_solute] Calculating from composition...")
                        ds = batch.get('ds', 'igg')
                        if isinstance(ds, str):
                            ds = ds.lower()
                        else:
                            ds = 'igg'
                        ds_mw = float(batch.get('ds_mw', 150))  # kDa
                        
                        # Handle viscosity flag (same logic as simulation.py)
                        visc_raw = batch.get('viscosity', 'n')
                        print(f"[D_solute] viscosity raw value: '{visc_raw}'")
                        
                        if str(visc_raw).lower() == 'y':
                            viscosity = float(batch.get('viscosity_user_input', 0.001))
                            print(f"[D_solute] Using user viscosity: {viscosity} Pa·s")
                        elif str(visc_raw).lower() == 'n':
                            ds_conc = float(batch.get('ds_conc', 50))
                            viscosity = 0.00003 * ds_conc - 0.0003 if ds in ["pgt121", "igg"] else 0.001
                            print(f"[D_solute] Calculated viscosity: {viscosity} Pa·s (ds_conc={ds_conc})")
                        else:
                            viscosity = float(visc_raw)
                            print(f"[D_solute] Using numeric viscosity: {viscosity} Pa·s")
                        
                        T = float(batch.get('T1_C', 70)) + 273.15  # K
                        print(f"[D_solute] Temperature: {T-273.15:.1f}°C")
                
                        # === CORRECT HYDRODYNAMIC RADIUS FOR PROTEINS ===
                        if ds in ['igg', 'mab', 'antibody', 'protein', 'bhv-1400', 'bhv1400']:
                            r_h_nm = 5.5  # nm
                        elif ds_mw < 1:
                            r_h_nm = 0.33 * (ds_mw * 1000)**0.5
                        else:
                            r_h_nm = 0.051 * (ds_mw ** 0.33) * 10
                        
                        print(f"[D_solute] ds='{ds}', ds_mw={ds_mw} kDa, r_h={r_h_nm:.2f} nm")
                        
                        r_h = r_h_nm * 1e-9  # convert to meters
                        D = scipy.constants.k * T / (6 * np.pi * viscosity * r_h)
                
                        batch['D_solute'] = D
                        print(f"[D_solute] ✓ Calculated: {D:.2e} m²/s (r_h={r_h_nm:.2f} nm, T={T-273.15:.0f}°C, η={viscosity:.4f} Pa·s)")
                
                    except Exception as e:
                        print(f"[D_solute] ✗ ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"[D_solute] Using fallback: 4e-11 m²/s")
                        batch['D_solute'] = 4e-11
                else:
                    print(f"[D_solute] Using value from input: {batch['D_solute']:.2e} m²/s")
                
                # Calculate D_moni if not provided (using hardcoded MW but batch temperature/viscosity)
                if 'D_moni' not in batch or batch.get('D_moni') is None or pd.isna(batch.get('D_moni')):
                    try:
                        moni_mw = 6800  # Fixed surfactant MW (g/mol) - consistent with rest of codebase
                        moni_r_h_method = 'small_molecule'  # Surfactants are typically small molecules
                        
                        # Calculate hydrodynamic radius (same logic as D_drug)
                        if moni_r_h_method == 'small_molecule':
                            r_h_m = 1.6  # Hardcoded Moni hydrodynamic radius
                        else:
                            r_h_m = float(batch.get('moni_r_h_custom', 1.6))

                        r_h = r_h_m * 1e-9  # convert to meters
                        D_moni = scipy.constants.k * T / (6 * np.pi * viscosity * r_h)
                        batch['D_moni'] = D_moni
                        print(f"[D_moni] ✓ Calculated: {D_moni:.2e} m²/s (hardcoded MW={moni_mw} g/mol, r_h={r_h_m:.2f} nm)")
                    except Exception as e:
                        print(f"[D_moni] ✗ ERROR: {e}")
                        batch['D_moni'] = 1e-11  # Fallback
                
                # Calculate D_stabilizer if not provided (using stab_A_mw from input) and stabilizer is present
                stab_conc = float(batch.get('stab_A_conc', 0))
                if stab_conc > 0 and ('D_stabilizer' not in batch or batch.get('D_stabilizer') is None or pd.isna(batch.get('D_stabilizer'))):
                    try:
                        stab_mw = float(batch.get('stab_A_mw', 300))  # Use existing stab_A_mw parameter
                        stab_r_h_method = batch.get('stabilizer_r_h_method', 'small_molecule')
                        
                        # Calculate hydrodynamic radius based on method (same logic as D_drug)
                        if stab_r_h_method == 'protein':
                            r_h_nm = 5.5  # nm (typical for proteins)
                        elif stab_r_h_method == 'small_molecule':
                            r_h_nm = 0.33 * (stab_mw ** 0.5)
                        else:  # custom
                            r_h_nm = float(batch.get('stabilizer_r_h_custom', 1.0))
                        
                        r_h = r_h_nm * 1e-9  # convert to meters
                        D_stabilizer = scipy.constants.k * T / (6 * np.pi * viscosity * r_h)
                        batch['D_stabilizer'] = D_stabilizer
                        print(f"[D_stabilizer] ✓ Calculated: {D_stabilizer:.2e} m²/s (MW={stab_mw} g/mol, r_h={r_h_nm:.2f} nm)")
                    except Exception as e:
                        print(f"[D_stabilizer] ✗ ERROR: {e}")
                        batch['D_stabilizer'] = 5e-10  # Fallback
                
                # Calculate D_additive_B if not provided (using additive_B_mw from input) and additive_B is present
                addb_conc = float(batch.get('additive_B_conc', 0))
                if addb_conc > 0 and ('D_additive_B' not in batch or batch.get('D_additive_B') is None or pd.isna(batch.get('D_additive_B'))):
                    try:
                        addb_mw = float(batch.get('additive_B_mw', 200))  # Use existing additive_B_mw parameter
                        addb_r_h_method = batch.get('additive_B_r_h_method', 'small_molecule')
                        
                        # Calculate hydrodynamic radius based on method (same logic as D_drug)
                        if addb_r_h_method == 'protein':
                            r_h_nm = 5.5  # nm (typical for proteins)
                        elif addb_r_h_method == 'small_molecule':
                            r_h_nm = 0.33 * (addb_mw ** 0.5)
                        else:  # custom
                            r_h_nm = float(batch.get('additive_B_r_h_custom', 1.0))
                        
                        r_h = r_h_nm * 1e-9  # convert to meters
                        D_additive_B = scipy.constants.k * T / (6 * np.pi * viscosity * r_h)
                        batch['D_additive_B'] = D_additive_B
                        print(f"[D_additive_B] ✓ Calculated: {D_additive_B:.2e} m²/s (MW={addb_mw} g/mol, r_h={r_h_nm:.2f} nm)")
                    except Exception as e:
                        print(f"[D_additive_B] ✗ ERROR: {e}")
                        batch['D_additive_B'] = 1e-10  # Fallback
                
                # Calculate D_additive_C if not provided (using additive_C_mw from input) and additive_C is present
                addc_conc = float(batch.get('additive_C_conc', 0))
                if addc_conc > 0 and ('D_additive_C' not in batch or batch.get('D_additive_C') is None or pd.isna(batch.get('D_additive_C'))):
                    try:
                        addc_mw = float(batch.get('additive_C_mw', 200))  # Use existing additive_C_mw parameter
                        addc_r_h_method = batch.get('additive_C_r_h_method', 'small_molecule')
                        
                        # Calculate hydrodynamic radius based on method (same logic as D_drug)
                        if addc_r_h_method == 'protein':
                            r_h_nm = 5.5  # nm (typical for proteins)
                        elif addc_r_h_method == 'small_molecule':
                            r_h_nm = 0.33 * (addc_mw ** 0.5)
                        else:  # custom
                            r_h_nm = float(batch.get('additive_C_r_h_custom', 1.0))
                        
                        r_h = r_h_nm * 1e-9  # convert to meters
                        D_additive_C = scipy.constants.k * T / (6 * np.pi * viscosity * r_h)
                        batch['D_additive_C'] = D_additive_C
                        print(f"[D_additive_C] ✓ Calculated: {D_additive_C:.2e} m²/s (MW={addc_mw} g/mol, r_h={r_h_nm:.2f} nm)")
                    except Exception as e:
                        print(f"[D_additive_C] ✗ ERROR: {e}")
                        batch['D_additive_C'] = 1e-10  # Fallback
                
                inputs_list.append(batch)
            param_order = params.tolist()
            return (inputs_list, param_order)
        else:
            # Normal format: columns are parameters
            params = df.iloc[0, :].values
            batches_data = df.iloc[1:, :].values
            num_batches = batches_data.shape[0]
            inputs_list = []
            for row_idx in range(num_batches):
                batch = dict(zip(params, batches_data[row_idx, :]))
                
                # Calculate D_solute if not provided
                print(f"[D_solute] Checking: 'D_solute' in batch={('D_solute' in batch)}, value={batch.get('D_solute')}, is None={batch.get('D_solute') is None}, pd.isna={pd.isna(batch.get('D_solute'))}")
                if 'D_solute' not in batch or batch['D_solute'] is None or pd.isna(batch['D_solute']):
                    try:
                        print("[D_solute] Calculating from composition...")
                        ds = batch.get('ds', 'igg')
                        if isinstance(ds, str):
                            ds = ds.lower()
                        else:
                            ds = 'igg'
                        ds_mw = float(batch.get('ds_mw', 150))  # kDa
                        
                        # Handle viscosity flag (same logic as simulation.py)
                        visc_raw = batch.get('viscosity', 'n')
                        print(f"[D_solute] viscosity raw value: '{visc_raw}'")
                        
                        if str(visc_raw).lower() == 'y':
                            viscosity = float(batch.get('viscosity_user_input', 0.001))
                            print(f"[D_solute] Using user viscosity: {viscosity} Pa·s")
                        elif str(visc_raw).lower() == 'n':
                            ds_conc = float(batch.get('ds_conc', 50))
                            viscosity = 0.00003 * ds_conc - 0.0003 if ds in ["pgt121", "igg"] else 0.001
                            print(f"[D_solute] Calculated viscosity: {viscosity} Pa·s (ds_conc={ds_conc})")
                        else:
                            viscosity = float(visc_raw)
                            print(f"[D_solute] Using numeric viscosity: {viscosity} Pa·s")
                        
                        T = float(batch.get('T1_C', 70)) + 273.15  # K
                        print(f"[D_solute] Temperature: {T-273.15:.1f}°C")
                
                        # === CORRECT HYDRODYNAMIC RADIUS FOR PROTEINS ===
                        if ds in ['igg', 'mab', 'antibody', 'protein', 'bhv-1400', 'bhv1400']:
                            r_h_nm = 5.5  # nm
                        elif ds_mw < 1:
                            r_h_nm = 0.33 * (ds_mw * 1000)**0.5
                        else:
                            r_h_nm = 0.051 * (ds_mw ** 0.33) * 10
                        
                        print(f"[D_solute] ds='{ds}', ds_mw={ds_mw} kDa, r_h={r_h_nm:.2f} nm")
                        
                        r_h = r_h_nm * 1e-9  # convert to meters
                        D = scipy.constants.k * T / (6 * np.pi * viscosity * r_h)
                
                        batch['D_solute'] = D
                        print(f"[D_solute] ✓ Calculated: {D:.2e} m²/s (r_h={r_h_nm:.2f} nm, T={T-273.15:.0f}°C, η={viscosity:.4f} Pa·s)")
                
                    except Exception as e:
                        print(f"[D_solute] ✗ ERROR: {e}")
                        import traceback
                        traceback.print_exc()
                        print(f"[D_solute] Using fallback: 4e-11 m²/s")
                        batch['D_solute'] = 4e-11
                else:
                    print(f"[D_solute] Using value from input: {batch['D_solute']:.2e} m²/s")
                
                # Calculate D_moni if not provided (using hardcoded MW but batch temperature/viscosity)
                if 'D_moni' not in batch or batch.get('D_moni') is None or pd.isna(batch.get('D_moni')):
                    try:
                        moni_mw = 6800  # Fixed surfactant MW (g/mol) - consistent with rest of codebase
                        moni_r_h_method = 'small_molecule'  # Surfactants are typically small molecules
                        
                        # Calculate hydrodynamic radius (same logic as D_drug)
                        if moni_r_h_method == 'small_molecule':
                            r_h_nm = 0.33 * (moni_mw ** 0.5)
                        else:
                            r_h_nm = float(batch.get('moni_r_h_custom', 1.0))
                        
                        r_h = r_h_nm * 1e-9  # convert to meters
                        D_moni = scipy.constants.k * T / (6 * np.pi * viscosity * r_h)
                        batch['D_moni'] = D_moni
                        print(f"[D_moni] ✓ Calculated: {D_moni:.2e} m²/s (hardcoded MW={moni_mw} g/mol, r_h={r_h_nm:.2f} nm)")
                    except Exception as e:
                        print(f"[D_moni] ✗ ERROR: {e}")
                        batch['D_moni'] = 1e-11  # Fallback
                
                # Calculate D_stabilizer if not provided (using stab_A_mw from input) and stabilizer is present
                stab_conc = float(batch.get('stab_A_conc', 0))
                if stab_conc > 0 and ('D_stabilizer' not in batch or batch.get('D_stabilizer') is None or pd.isna(batch.get('D_stabilizer'))):
                    try:
                        stab_mw = float(batch.get('stab_A_mw', 300))  # Use existing stab_A_mw parameter
                        stab_r_h_method = batch.get('stabilizer_r_h_method', 'small_molecule')
                        
                        # Calculate hydrodynamic radius based on method (same logic as D_drug)
                        if stab_r_h_method == 'protein':
                            r_h_nm = 5.5  # nm (typical for proteins)
                        elif stab_r_h_method == 'small_molecule':
                            r_h_nm = 0.33 * (stab_mw ** 0.5)
                        else:  # custom
                            r_h_nm = float(batch.get('stabilizer_r_h_custom', 1.0))
                        
                        r_h = r_h_nm * 1e-9  # convert to meters
                        D_stabilizer = scipy.constants.k * T / (6 * np.pi * viscosity * r_h)
                        batch['D_stabilizer'] = D_stabilizer
                        print(f"[D_stabilizer] ✓ Calculated: {D_stabilizer:.2e} m²/s (MW={stab_mw} g/mol, r_h={r_h_nm:.2f} nm)")
                    except Exception as e:
                        print(f"[D_stabilizer] ✗ ERROR: {e}")
                        batch['D_stabilizer'] = 5e-10  # Fallback
                
                # Calculate D_additive_B if not provided (using additive_B_mw from input) and additive_B is present
                addb_conc = float(batch.get('additive_B_conc', 0))
                if addb_conc > 0 and ('D_additive_B' not in batch or batch.get('D_additive_B') is None or pd.isna(batch.get('D_additive_B'))):
                    try:
                        addb_mw = float(batch.get('additive_B_mw', 200))  # Use existing additive_B_mw parameter
                        addb_r_h_method = batch.get('additive_B_r_h_method', 'small_molecule')
                        
                        # Calculate hydrodynamic radius based on method (same logic as D_drug)
                        if addb_r_h_method == 'protein':
                            r_h_nm = 5.5  # nm (typical for proteins)
                        elif addb_r_h_method == 'small_molecule':
                            r_h_nm = 0.33 * (addb_mw ** 0.5)
                        else:  # custom
                            r_h_nm = float(batch.get('additive_B_r_h_custom', 1.0))
                        
                        r_h = r_h_nm * 1e-9  # convert to meters
                        D_additive_B = scipy.constants.k * T / (6 * np.pi * viscosity * r_h)
                        batch['D_additive_B'] = D_additive_B
                        print(f"[D_additive_B] ✓ Calculated: {D_additive_B:.2e} m²/s (MW={addb_mw} g/mol, r_h={r_h_nm:.2f} nm)")
                    except Exception as e:
                        print(f"[D_additive_B] ✗ ERROR: {e}")
                        batch['D_additive_B'] = 1e-10  # Fallback
                
                # Calculate D_additive_C if not provided (using additive_C_mw from input) and additive_C is present
                addc_conc = float(batch.get('additive_C_conc', 0))
                if addc_conc > 0 and ('D_additive_C' not in batch or batch.get('D_additive_C') is None or pd.isna(batch.get('D_additive_C'))):
                    try:
                        addc_mw = float(batch.get('additive_C_mw', 200))  # Use existing additive_C_mw parameter
                        addc_r_h_method = batch.get('additive_C_r_h_method', 'small_molecule')
                        
                        # Calculate hydrodynamic radius based on method (same logic as D_drug)
                        if addc_r_h_method == 'protein':
                            r_h_nm = 5.5  # nm (typical for proteins)
                        elif addc_r_h_method == 'small_molecule':
                            r_h_nm = 0.33 * (addc_mw ** 0.5)
                        else:  # custom
                            r_h_nm = float(batch.get('additive_C_r_h_custom', 1.0))
                        
                        r_h = r_h_nm * 1e-9  # convert to meters
                        D_additive_C = scipy.constants.k * T / (6 * np.pi * viscosity * r_h)
                        batch['D_additive_C'] = D_additive_C
                        print(f"[D_additive_C] ✓ Calculated: {D_additive_C:.2e} m²/s (MW={addc_mw} g/mol, r_h={r_h_nm:.2f} nm)")
                    except Exception as e:
                        print(f"[D_additive_C] ✗ ERROR: {e}")
                        batch['D_additive_C'] = 1e-10  # Fallback
                
                inputs_list.append(batch)
            param_order = params.tolist()
            return (inputs_list, param_order)
    except Exception as e:
        print(f"Error loading inputs from Excel: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Load inputs from file or collect manually
    if args.excel:
        # Prepend the Python directory path if not absolute
        excel_path = args.excel
        if not os.path.isabs(excel_path):
            pass  # excel_path is used as-is (relative or absolute)
        result = load_inputs_from_excel(excel_path)
        if result is not None:
            # Updated function returns (inputs_list, param_order)
            inputs_list, param_order = result
            print(f"Loaded {len(inputs_list)} batches from Excel with param_order ({len(param_order)} rows)")
        else:
            inputs_list = None
            param_order = None
        
        if inputs_list is None or len(inputs_list) == 0:
            print("Failed to load inputs from file. Switching to manual input.")
            inputs = collect_inputs()
            inputs_list = [inputs]
            param_order = None  # No input order for manual
    else:
        # Ask user for input method
        choice = input("Do you want to load inputs from an Excel file? (Y/N): ").strip().lower()
        if choice == 'y':
            filename = input("Enter the Excel file path: ").strip()
            # Prepend the Python directory path if not absolute
            if not os.path.isabs(filename):
                pass  # filename is used as-is (relative or absolute)
            result = load_inputs_from_excel(filename)
            if result is not None:
                # Updated function returns (inputs_list, param_order)
                inputs_list, param_order = result
                print(f"Loaded {len(inputs_list)} batches from Excel with param_order ({len(param_order)} rows)")
            else:
                inputs_list = None
                param_order = None
            
            if inputs_list is None or len(inputs_list) == 0:
                print("Failed to load inputs from file. Switching to manual input.")
                inputs = collect_inputs()
                inputs_list = [inputs]
                param_order = None  # No input order for manual
        else:
            inputs = collect_inputs()
            inputs_list = [inputs]
            param_order = None
    
    # ========================================================================
    # ASK FOR OUTPUT FILENAME WITH MODE-SPECIFIC DEFAULTS
    # ========================================================================
    
    if RUN_MODE == 'calibration':
        default_filename = "Snapshot_mab_training.xlsx"
        prompt = f"Enter the name of the Excel file to save results (default: {default_filename}): "
    else:
        default_filename = "Snapshot_mab_output.xlsx"
        prompt = f"Enter the name of the Excel file to save results (default: {default_filename}): "
    
    output_filename = input(prompt).strip()
    if not output_filename:
        output_filename = default_filename
    if not output_filename.endswith('.xlsx'):
        output_filename += '.xlsx'
    # Prepend the Python directory path if not absolute
    if not os.path.isabs(output_filename):
        pass  # output_filename is used as-is (relative or absolute)
    print(f"Results will be saved to: {output_filename}")
    
    # ========================================================================
    # MAIN SIMULATION LOOP
    # ========================================================================
    
    # Loop over each set of inputs
    for idx, inputs in enumerate(inputs_list):
        batch_name = f"batch_{idx+1}" if len(inputs_list) > 1 else "single_run"
        print(f"\n{'='*80}")
        print(f"Running simulation for {batch_name} ({RUN_MODE.upper()} MODE)")
        print(f"{'='*80}\n")
        
        # Set moni_mw to fixed value since it's firm at 6800 Da
        inputs['moni_mw'] = 6800
        
        # Handle multiple column names for RH (same logic as enhanced_physics_evolution_plotter_FINAL.py)
        # Try multiple column names for RH1
        if 'RH1' not in inputs or inputs['RH1'] is None or pd.isna(inputs['RH1']):
            for key in ['RH1', 'Drying Gas RH (ambient)']:
                if key in inputs and inputs[key] is not None and not pd.isna(inputs[key]):
                    inputs['RH1'] = inputs[key]
                    print(f"Using '{key}' as RH1 (drying gas RH): {inputs['RH1']}%")
                    break
        
        # Try multiple column names for RH2 (atomization gas RH)
        if 'RH2' not in inputs or inputs['RH2'] is None or pd.isna(inputs['RH2']):
            for key in ['RH2', 'Atomization Gas RH', 'Atom. Gas RH']:
                if key in inputs and inputs[key] is not None and not pd.isna(inputs[key]):
                    inputs['RH2'] = inputs[key]
                    print(f"Using '{key}' as RH2 (atomization gas RH): {inputs['RH2']}%")
                    break
        
        # Calculate PSD Span if not provided
        if 'Span' not in inputs or inputs['Span'] is None or pd.isna(inputs['Span']):
            try:
                d10 = inputs.get('D10_actual')
                d50 = inputs.get('D50_actual')
                d90 = inputs.get('D90_actual')
                if d10 is not None and d50 is not None and d90 is not None and d50 != 0:
                    inputs['Span'] = (d90 - d10) / d50
                    print(f"Calculated PSD Span: {inputs['Span']:.3f}")
                else:
                    inputs['Span'] = None
            except Exception as e:
                print(f"Error calculating Span: {e}")
                inputs['Span'] = None
        
        # Calculate D_solute if not provided (automatic calculation)
        if 'D_solute' not in inputs or inputs['D_solute'] is None or pd.isna(inputs['D_solute']):
            try:
                ds = str(inputs.get('ds', 'igg')).lower()
                ds_mw = float(inputs.get('ds_mw', 150))  # kDa
                viscosity = float(inputs.get('viscosity_user_input', inputs.get('viscosity', 0.001)))  # Pa·s
                T = float(inputs.get('T1_C', 70)) + 273.15  # K (use inlet temp)
        
                # === CORRECT HYDRODYNAMIC RADIUS FOR PROTEINS ===
                if ds in ['igg', 'mab', 'antibody', 'protein', 'bhv-1400']:
                    # Literature value for monoclonal antibodies / IgG
                    r_h_nm = 5.5  # nm
                elif ds_mw < 1:  # small molecule (e.g. moni, histidine)
                    r_h_nm = 0.33 * (ds_mw * 1000)**0.5  # classic small-molecule scaling
                else:
                    # General protein scaling (falls back to ~5 nm for 100–200 kDa)
                    r_h_nm = 0.051 * (ds_mw ** 0.33) * 10  # empirical fit
        
                r_h = r_h_nm * 1e-9  # convert to meters
                D = scipy.constants.k * T / (6 * np.pi * viscosity * r_h)
        
                inputs['D_solute'] = D
                print(f"Calculated D_solute = {D:.2e} m²/s (r_h = {r_h_nm:.2f} nm, T = {T-273.15:.0f}°C, η = {viscosity:.4f} Pa·s)")
        
            except Exception as e:
                print(f"Error calculating D_solute: {e}. Using fallback 4e-11 m²/s")
                inputs['D_solute'] = 4e-11
        
        inputs_file = f'last_inputs_{batch_name}.json'
        with open(inputs_file, 'w') as f:
            json.dump(inputs, f, indent=4)
        print(f"Inputs saved to {inputs_file}.")
        
        if inputs is None:
            print(f"Input collection returned None for {batch_name}. Skipping.")
            continue

        # Check if T_outlet_C is provided (required since outlet temperature prediction was removed)
        if 'T_outlet_C' not in inputs or inputs['T_outlet_C'] is None or pd.isna(inputs['T_outlet_C']):
            # Use a reasonable default: inlet temperature minus 35°C (typical for spray dryers)
            t_inlet = None
            for temp_key in ['T1_C', 'Drying Gas Inlet (C)', 'Inlet Temperature required (C)']:
                if temp_key in inputs and inputs[temp_key] is not None and not pd.isna(inputs[temp_key]):
                    t_inlet = float(inputs[temp_key])
                    break
            if t_inlet is None:
                t_inlet = 70  # Default 70°C if not provided
            inputs['T_outlet_C'] = max(20, min(100, t_inlet - 35))  # Clamp between 20-100°C
            print(f"Warning: T_outlet_C not provided, using default {inputs['T_outlet_C']:.1f}°C (inlet temp - 35°C)")

        # ========================================================================
        # RUN BASIC SIMULATION (BOTH MODES)
        # ========================================================================
        
        try:
            result = run_full_spray_drying_simulation(inputs)
        except Exception:
            print("Exception raised during simulation:")
            traceback.print_exc()
            continue  # Skip to next batch
        if result is None:
            print("Simulation failed. Check inputs and debug log.")
            continue

        # DEBUG - Print result type
        print(f"\n{'='*70}")
        print(f"DEBUG at batch processing section:")
        print(f"  type(result) = {type(result)}")
        print(f"  isinstance(result, dict) = {isinstance(result, dict)}")
        if isinstance(result, tuple):
            print(f"  result is tuple with {len(result)} elements")
        print(f"{'='*70}\n")
        
        # Convert result to dictionary
        if isinstance(result, dict):
            result_dict = result
        elif isinstance(result, tuple) and len(result) == 2:
            # Old format: (param_names, results_tuple)
            param_names, results_tuple = result
            result_dict = dict(zip(param_names, results_tuple))
        elif isinstance(result, tuple):
            # Raw tuple format from simulation.py
            result_dict = dict(zip(SIMULATION_PARAM_NAMES, result))
            
            # Add aliases for morphology prediction feature mapping (used in production mode)
            result_dict['Surface recession velocity'] = result_dict.get('v_evap', 1e-4)
            result_dict['Peclet Number'] = result_dict.get('Pe', 5)
            result_dict['Marangoni Number'] = result_dict.get('Ma', 1)
            result_dict['Reynolds number'] = result_dict.get('Re_droplet', 100)
            result_dict['Nusselt number Nu'] = result_dict.get('Nu', 10)
            result_dict['Sherwood number Sh'] = result_dict.get('Sh', 10)
        
        # === FINAL est_RH_pct: Realistic psychrometric estimate (always available) ===
        # This restores previous reliable behavior: psychrometric RH_out when no measured
        est_RH_pct = result_dict.get('RH_out', 10.0)  # Current RH_out is psychrometric fallback or measured
        print(f"Info: est_RH_pct set to {est_RH_pct:.2f}% (psychrometric/measured)")

        # Optional: Production mode ML override for improvement with more data
        # if RUN_MODE == 'production' and 'ml_rh_model' in globals():  # If ML model loaded from calibration.json
        #     try:
        #         features = extract_rh_features(inputs, result_dict)  # Your existing feature func
        #         est_RH_pct_ml = ml_rh_model.predict([features])[0]
        #         est_RH_pct = est_RH_pct_ml
        #         print(f"Info: Production mode — ML improved est_RH_pct: {est_RH_pct:.2f}%")
        #     except Exception as e:
        #         print(f"Warning: ML RH prediction failed ({e}) — using psychrometric {est_RH_pct:.2f}%")

        result_dict['est_RH_pct'] = est_RH_pct
        # Use est_RH_pct for moisture prediction below
        
        # ========================================================================
        # MODE-SPECIFIC PROCESSING
        # ========================================================================
        
        if RUN_MODE == 'production':
            # PRODUCTION MODE: Run all ML predictions
            print(f"\n{'-'*70}")
            print("PRODUCTION MODE: Running ML predictions...")
            print(f"{'-'*70}\n")
            
            # --- Morphology Prediction ---
            try:
                morphology_result = predict_morphology(inputs, result_dict)
                morphology = morphology_result['prediction']
                confidence = morphology_result['confidence']
                result_dict['Predicted Morphology'] = morphology
                result_dict['Morphology Confidence'] = confidence
                print(f"✓ Morphology prediction: {morphology} (confidence: {confidence:.3f})")
            except Exception as e:
                print(f"✗ Morphology prediction failed: {e}")
                result_dict['Predicted Morphology'] = "unknown"
                result_dict['Morphology Confidence'] = 0.0

            # --- Advanced Droplet Model (Péclet numbers) ---
            try:
                from advanced_droplet_model import run_drying_simulation
                row = pd.Series(inputs)
                advanced_results = run_drying_simulation(row)
                
                # Preserve input morphology as "Morphology (known)"
                if 'Morphology' in inputs and inputs['Morphology']:
                    result_dict['Morphology (known)'] = inputs['Morphology']
                
                # Add Péclet numbers and other advanced results to result_dict
                peclet_keys = ['effective_pe', 'max_pe', 'integrated_pe', 'effective_pe_drug', 'max_pe_drug', 
                              'integrated_pe_drug', 'effective_pe_moni', 'max_pe_moni', 'integrated_pe_moni',
                              'effective_pe_stabilizer', 'max_pe_stabilizer', 'integrated_pe_stabilizer',
                              'effective_pe_buffer', 'max_pe_buffer', 'integrated_pe_buffer',
                              'effective_pe_additive_B', 'max_pe_additive_B', 'integrated_pe_additive_B',
                              'effective_pe_additive_C', 'max_pe_additive_C', 'integrated_pe_additive_C']
                
                # Get component concentrations to check presence
                buffer_conc = inputs.get('buffer_conc', 0)
                stab_conc = inputs.get('stab_A_conc', 0)
                additive_b_conc = inputs.get('additive_B_conc', 0)
                additive_c_conc = inputs.get('additive_C_conc', 0)
                
                # Handle NaN values
                import math
                buffer_present = buffer_conc and not math.isnan(float(buffer_conc)) and float(buffer_conc) > 0
                stab_present = stab_conc and not math.isnan(float(stab_conc)) and float(stab_conc) > 0
                additive_b_present = additive_b_conc and not math.isnan(float(additive_b_conc)) and float(additive_b_conc) > 0
                additive_c_present = additive_c_conc and not math.isnan(float(additive_c_conc)) and float(additive_c_conc) > 0
                
                for key in peclet_keys:
                    if key in advanced_results:
                        result_dict[key] = advanced_results[key]
                    else:
                        # Only set defaults for components that are actually present in the formulation
                        should_set_default = True
                        
                        if 'buffer' in key and not buffer_present:
                            should_set_default = False
                        elif 'stabilizer' in key and not stab_present:
                            should_set_default = False
                        elif 'additive_B' in key and not additive_b_present:
                            should_set_default = False
                        elif 'additive_C' in key and not additive_c_present:
                            should_set_default = False
                        
                        if should_set_default:
                            # Set defaults if not calculated
                            if 'effective_pe' in key:
                                result_dict[key] = 1.0
                            elif 'max_pe' in key:
                                result_dict[key] = 10.0
                            else:
                                result_dict[key] = 1.0
                
                print(f"✓ Advanced droplet model: Péclet numbers calculated")
            except Exception as e:
                print(f"✗ Advanced droplet model calculation failed: {e}")
                # Set default Péclet values based on which components are present in the formulation
                result_dict.update({
                    'effective_pe': 1.0, 'max_pe': 10.0, 'integrated_pe': 5.0,
                    'effective_pe_drug': 1.0, 'max_pe_drug': 10.0,
                    'effective_pe_moni': 0.1, 'max_pe_moni': 1.0,
                })

                # Only set Pe values for components that are actually present in the formulation
                buffer_conc = inputs.get('buffer_conc', 0)
                if buffer_conc and buffer_conc > 0:
                    result_dict.update({
                        'effective_pe_buffer': 0.1, 'max_pe_buffer': 1.0, 'integrated_pe_buffer': 0.5
                    })

                stab_conc = inputs.get('stab_A_conc', 0)
                if stab_conc and stab_conc > 0:
                    result_dict.update({
                        'effective_pe_stabilizer': 1.0, 'max_pe_stabilizer': 1.0, 'integrated_pe_stabilizer': 0.5
                    })

                additive_b_conc = inputs.get('additive_B_conc', 0)
                if additive_b_conc and additive_b_conc > 0:
                    result_dict.update({
                        'effective_pe_additive_B': 0.1, 'max_pe_additive_B': 1.0, 'integrated_pe_additive_B': 0.5
                    })

                additive_c_conc = inputs.get('additive_C_conc', 0)
                if additive_c_conc and additive_c_conc > 0:
                    result_dict.update({
                        'effective_pe_additive_C': 0.1, 'max_pe_additive_C': 1.0, 'integrated_pe_additive_C': 0.5
                    })

            # Add Shell Formation Time to results
            result_dict['Shell_Formation_Time_ms'] = (result_dict.get('shell_formation_time', result_dict.get('t_dry', 0.1) * 0.3) * 1000)
            result_dict['Shell_Formation_Fraction'] = (result_dict['Shell_Formation_Time_ms'] / (result_dict.get('t_dry', 0.1) * 1000 + 1e-6))

            # --- Darcy Pressure Analysis ---
            try:
                from darcy_pressure import calculate_complete_darcy_analysis
                
                # Extract parameters needed for Darcy calculation
                R_current = (result_dict.get('D50_calc', 3.44e-6) / 2) / 1e6  # Convert μm to m
                R_initial = R_current * 2  # Assume initial radius is 2x final (rough estimate)
                solids_fraction = result_dict.get('solids_frac', 0.05)
                moisture = result_dict.get('moisture_predicted', 0.05)
                evaporation_rate = result_dict.get('v_evap', 1e-6)  # m/s
                T_droplet = result_dict.get('T_outlet_C', 80) + 273.15  # Convert to K
                surface_tension = inputs.get('surface_tension_user_input', 0.072) if inputs.get('surface_tension') == 'y' else 0.072
                
                # Composition based on actual formulation data
                # Calculate fractions from available concentration data
                ds_conc = inputs.get('ds_conc', 50)  # Drug substance concentration
                moni_conc = inputs.get('moni_conc', 5)  # Moni concentration  
                stab_conc = inputs.get('stab_A_conc', 10)  # Stabilizer concentration
                buffer_conc = inputs.get('buffer_conc', 10)  # Buffer concentration
                additive_conc = (inputs.get('additive_B_conc', 0) + inputs.get('additive_C_conc', 0))  # Combined additives
                
                # Calculate total solids for fraction calculation
                total_solids_conc = ds_conc + moni_conc + stab_conc + buffer_conc + additive_conc
                if total_solids_conc > 0:
                    composition = {
                        'drug': ds_conc / total_solids_conc,        # Drug substance fraction
                        'moni': moni_conc / total_solids_conc,      # Moni fraction  
                        'stabilizer': stab_conc / total_solids_conc, # Stabilizer fraction
                        'buffer': buffer_conc / total_solids_conc,   # Buffer fraction
                        'additive': additive_conc / total_solids_conc, # Additive fraction
                        'salt': 0.0  # Placeholder for salt if needed
                    }
                else:
                    # Fallback to generic composition if no concentration data
                    composition = {'drug': 0.7, 'stabilizer': 0.2, 'buffer': 0.1}
                
                Pe = result_dict.get('Pe', 10.0)
                Ma = result_dict.get('Ma', 1.0)
                shell_formed = True  # Assume shell forms
                
                # Péclet values for morphology prediction
                pe_values = {
                    'max_pe': result_dict.get('max_pe', 10.0),
                    'integrated_pe': result_dict.get('integrated_pe', 5.0),
                    'effective_pe': result_dict.get('effective_pe', 1.0)
                }
                
                darcy_results = calculate_complete_darcy_analysis(
                    R_current, R_initial, solids_fraction, moisture, evaporation_rate,
                    T_droplet, surface_tension, composition, Pe, Ma, shell_formed, pe_values
                )
                
                # Add Darcy results to result_dict
                result_dict['darcy_P_internal_Pa'] = darcy_results.get('P_internal_Pa')
                result_dict['darcy_Delta_P_Pa'] = darcy_results.get('Delta_P_Pa')
                result_dict['darcy_Pi_ratio'] = darcy_results.get('Pi_pressure_ratio')
                result_dict['darcy_morphology_predicted'] = darcy_results.get('morphology_predicted')
                result_dict['darcy_morphology_mechanism'] = darcy_results.get('morphology_mechanism')
                result_dict['darcy_shell_thickness_um'] = darcy_results.get('shell_thickness_um')
                result_dict['darcy_permeability_m2'] = darcy_results.get('permeability_m2')
                result_dict['darcy_Darcy_number'] = darcy_results.get('Darcy_number')
                
                print(f"✓ Darcy pressure analysis: Internal pressure = {darcy_results.get('P_internal_Pa', 0):.0f} Pa")
            except Exception as e:
                print(f"✗ Darcy pressure analysis failed: {e}")
                # Set default Darcy values
                result_dict.update({
                    'darcy_P_internal': None, 'darcy_Delta_P': None, 'darcy_Pi_ratio': None,
                    'darcy_morphology': None, 'darcy_mechanism': None, 'darcy_shell_thickness_um': None,
                    'darcy_permeability': None, 'darcy_Darcy_number': None
                })
        
        else:  # CALIBRATION MODE
            print(f"\n{'-'*70}")
            print("CALIBRATION MODE: Running physics calculations (no ML predictions)")
            print(f"{'-'*70}\n")
            
            # IMPORTANT: In calibration mode, we still need ALL physics calculations
            # because these are FEATURES for training the ML model!
            # We only skip the morphology PREDICTION (since we don't have a trained model yet)
            
            # Set placeholder for morphology prediction (no model to use yet)
            result_dict['Predicted Morphology'] = "N/A (Calibration)"
            result_dict['Morphology Confidence'] = 0.0
            
            # KEEP all Péclet values from simulation.py - these are training features!
            # simulation.py already calculated: Pe, Ma, effective_pe, max_pe, 
            # effective_pe_drug, effective_pe_moni, effective_pe_stabilizer, etc.
            # DO NOT overwrite them!
            
            # Calculate advanced droplet model for additional physics features
            try:
                from advanced_droplet_model import run_drying_simulation
                row = pd.Series(inputs)
                advanced_results = run_drying_simulation(row)
                
                # Preserve input morphology as "Morphology (known)"
                if 'Morphology' in inputs and inputs['Morphology']:
                    result_dict['Morphology (known)'] = inputs['Morphology']
                
                # Add any additional Péclet numbers from advanced model
                peclet_keys = ['integrated_pe', 'effective_pe_buffer', 'max_pe_buffer']
                for key in peclet_keys:
                    if key in advanced_results and advanced_results[key] is not None:
                        result_dict[key] = advanced_results[key]
                
                print(f"✓ Advanced physics calculations complete")
            except Exception as e:
                print(f"⚠ Advanced droplet model calculation failed (continuing): {e}")
                # Set defaults only for truly missing values
                if 'integrated_pe' not in result_dict:
                    result_dict['integrated_pe'] = result_dict.get('Pe', 5.0)

            # Add Shell Formation Time to results
            result_dict['Shell_Formation_Time_ms'] = (result_dict.get('shell_formation_time', result_dict.get('t_dry', 0.1) * 0.3) * 1000)
            result_dict['Shell_Formation_Fraction'] = (result_dict['Shell_Formation_Time_ms'] / (result_dict.get('t_dry', 0.1) * 1000 + 1e-6))

            # Calculate Darcy pressure analysis (needed for training features!)
            try:
                from darcy_pressure import calculate_complete_darcy_analysis
                
                # Extract parameters needed for Darcy calculation
                R_current = (result_dict.get('D50_calc', 3.44e-6) / 2) / 1e6  # Convert μm to m
                R_initial = R_current * 2  # Assume initial radius is 2x final
                solids_fraction = result_dict.get('solids_frac', 0.05)
                moisture = result_dict.get('moisture_predicted', 0.05)
                evaporation_rate = result_dict.get('v_evap', 1e-6)  # m/s
                T_droplet = result_dict.get('T_outlet_C', 80) + 273.15  # Convert to K
                surface_tension = inputs.get('surface_tension_user_input', 0.072) if inputs.get('surface_tension') == 'y' else 0.072
                
                # Composition based on actual formulation data
                # Calculate fractions from available concentration data
                ds_conc = inputs.get('ds_conc', 50)  # Drug substance concentration
                moni_conc = inputs.get('moni_conc', 5)  # Moni concentration  
                stab_conc = inputs.get('stab_A_conc', 10)  # Stabilizer concentration
                buffer_conc = inputs.get('buffer_conc', 10)  # Buffer concentration
                additive_conc = (inputs.get('additive_B_conc', 0) + inputs.get('additive_C_conc', 0))  # Combined additives
                
                # Calculate total solids for fraction calculation
                total_solids_conc = ds_conc + moni_conc + stab_conc + buffer_conc + additive_conc
                if total_solids_conc > 0:
                    composition = {
                        'drug': ds_conc / total_solids_conc,        # Drug substance fraction
                        'moni': moni_conc / total_solids_conc,      # Moni fraction  
                        'stabilizer': stab_conc / total_solids_conc, # Stabilizer fraction
                        'buffer': buffer_conc / total_solids_conc,   # Buffer fraction
                        'additive': additive_conc / total_solids_conc, # Additive fraction
                        'salt': 0.0  # Placeholder for salt if needed
                    }
                else:
                    # Fallback to generic composition if no concentration data
                    composition = {'drug': 0.7, 'stabilizer': 0.2, 'buffer': 0.1}
                
                Pe = result_dict.get('Pe', 10.0)
                Ma = result_dict.get('Ma', 1.0)
                shell_formed = True  # Assume shell forms
                
                # Péclet values for morphology prediction
                pe_values = {
                    'max_pe': result_dict.get('max_pe', 10.0),
                    'integrated_pe': result_dict.get('integrated_pe', 5.0),
                    'effective_pe': result_dict.get('effective_pe', 1.0)
                }
                
                darcy_results = calculate_complete_darcy_analysis(
                    R_current, R_initial, solids_fraction, moisture, evaporation_rate,
                    T_droplet, surface_tension, composition, Pe, Ma, shell_formed, pe_values
                )
                
                # Add Darcy results to result_dict
                result_dict['darcy_P_internal_Pa'] = darcy_results.get('P_internal_Pa')
                result_dict['darcy_Delta_P_Pa'] = darcy_results.get('Delta_P_Pa')
                result_dict['darcy_Pi_ratio'] = darcy_results.get('Pi_pressure_ratio')
                result_dict['darcy_morphology_predicted'] = darcy_results.get('morphology_predicted')
                result_dict['darcy_morphology_mechanism'] = darcy_results.get('morphology_mechanism')
                result_dict['darcy_shell_thickness_um'] = darcy_results.get('shell_thickness_um')
                result_dict['darcy_permeability_m2'] = darcy_results.get('permeability_m2')
                result_dict['darcy_Darcy_number'] = darcy_results.get('Darcy_number')
                
                print(f"✓ Darcy pressure analysis complete: Internal pressure = {darcy_results.get('P_internal_Pa', 0):.0f} Pa")
            except Exception as e:
                print(f"⚠ Darcy pressure analysis failed (continuing): {e}")
                # Set N/A only if calculation failed
                result_dict.update({
                    'darcy_P_internal': None, 'darcy_Delta_P': None, 'darcy_Pi_ratio': None,
                    'darcy_morphology': None, 'darcy_mechanism': None, 'darcy_shell_thickness_um': None,
                    'darcy_permeability': None, 'darcy_Darcy_number': None
                })
            
            print("✓ Physics simulation complete (training data ready)")
        
        # ========================================================================
        # SAVE RESULTS
        # ========================================================================
        
        # Remove predicted_T_outlet_C from inputs before saving to avoid duplicate rows
        inputs_for_save = inputs.copy()
        
        save_output(result_dict, inputs_for_save, filename=output_filename, input_param_order=preferred_order)

        print(f"\n{'='*80}")
        print(f"Simulation completed for {batch_name}")
        print(f"Results saved to: {output_filename}")
        print(f"{'='*80}\n")
    
    # ========================================================================
    # RERUN OPTION (ONLY FOR INTERACTIVE SINGLE-BATCH MODE)
    # ========================================================================
    
    # Only allow rerun if inputs were collected interactively (not from file)
    if not args.excel and len(inputs_list) == 1:
        while True:
            rerun = input("Run again? (Y/N): ").strip().lower()
            if rerun != "y":
                break
            print("Available variables to change:", ", ".join(inputs.keys()))
            change_vars = input("Enter comma-separated list of variables to change (e.g., T1_C,feed_g_min): ").strip().split(',')
            for var in change_vars:
                var = var.strip()
                if var in inputs:
                    new_value = input(f"Enter new value for {var}: ").strip()
                    try:
                        if var == "solids_frac":
                            solids_frac_input = new_value.strip()
                            if '%' in solids_frac_input:
                                inputs[var] = float(solids_frac_input.rstrip('%')) / 100.0
                            else:
                                val = float(solids_frac_input)
                                if val >= 1:
                                    inputs[var] = val / 100.0
                                else:
                                    inputs[var] = val
                        else:
                            inputs[var] = float(new_value)
                    except ValueError:
                        inputs[var] = new_value
                    print(f"Updated {var} to {inputs[var]}")
                else:
                    print(f"Variable {var} not found in inputs.")
            
            # Calculate PSD Span if not provided
            if 'Span' not in inputs or inputs['Span'] is None or pd.isna(inputs['Span']):
                try:
                    d10 = inputs.get('D10_actual')
                    d50 = inputs.get('D50_actual')
                    d90 = inputs.get('D90_actual')
                    if d10 is not None and d50 is not None and d90 is not None and d50 != 0:
                        inputs['Span'] = (d90 - d10) / d50
                        print(f"Calculated PSD Span: {inputs['Span']:.3f}")
                    else:
                        inputs['Span'] = None
                except Exception as e:
                    print(f"Error calculating Span: {e}")
                    inputs['Span'] = None
            
            result = run_full_spray_drying_simulation(inputs)
            if result is None:
                print("Simulation failed. Check inputs and debug log.")
                continue

            if isinstance(result, dict):
                result_dict = result
            elif isinstance(result, tuple) and len(result) == 2:
                # Old format: (param_names, results_tuple)
                param_names, results_tuple = result
                result_dict = dict(zip(param_names, results_tuple))
            elif isinstance(result, tuple):
                # Raw tuple format from simulation.py
                result_dict = dict(zip(SIMULATION_PARAM_NAMES, result))
                
                # Add aliases for morphology prediction feature mapping
                result_dict['Surface recession velocity'] = result_dict.get('v_evap', 1e-4)
                result_dict['Peclet Number'] = result_dict.get('Pe', 5)
                result_dict['Marangoni Number'] = result_dict.get('Ma', 1)
                result_dict['Reynolds number'] = result_dict.get('Re_droplet', 100)
                result_dict['Nusselt number Nu'] = result_dict.get('Nu', 10)
                result_dict['Sherwood number Sh'] = result_dict.get('Sh', 10)
            
            # Apply mode-specific processing for rerun (same as above)
            if RUN_MODE == 'production':
                try:
                    morphology_result = predict_morphology(inputs, result_dict)
                    morphology = morphology_result['prediction']
                    confidence = morphology_result['confidence']
                    result_dict['Predicted Morphology'] = morphology
                    result_dict['Morphology Confidence'] = confidence
                    print(f"Predicted morphology: {morphology} (confidence: {confidence:.3f})")
                except Exception as e:
                    print(f"Morphology prediction failed: {e}")
                    result_dict['Predicted Morphology'] = "unknown"
                    result_dict['Morphology Confidence'] = 0.0
            else:
                result_dict['Predicted Morphology'] = "N/A (Calibration)"
                result_dict['Morphology Confidence'] = 0.0
            
            # Remove predicted_T_outlet_C from inputs before saving to avoid duplicate rows
            inputs_for_save = inputs.copy()
            
            save_output((list(result_dict.keys()), tuple(result_dict.values())), inputs_for_save, filename=output_filename, input_param_order=param_order)
            print(f"Rerun completed and saved to {output_filename}")

    print("\n" + "="*80)
    print(f"All simulations complete! Results saved to: {output_filename}")
    if RUN_MODE == 'calibration':
        print("\nNext step: Run learn_calibration.py with this output file to train ML models.")
    print("="*80 + "\n")

    # ========================================================================
    # AUTOMATIC PHYSICS EVOLUTION PLOTTING
    # ========================================================================

    print("\n" + "="*80)
    print("AUTOMATIC PHYSICS EVOLUTION PLOTTING")
    print("="*80)
    print("Generating enhanced physics-based particle evolution plots...")

    try:
        # Import the plotting function from the enhanced plotter
        from enhanced_physics_evolution_plotter_FINAL import run_physics_evolution_plotting

        # Use the output Excel file that was just created for plotting
        plot_excel_file = output_filename

        # Run plotting for all batches (batch_id=None plots all)
        success = run_physics_evolution_plotting(
            excel_file=plot_excel_file,
            batch_id=None,  # Plot all batches
            output_dir="./outputs"
        )

        if success:
            print("\n✅ PLOTTING COMPLETED SUCCESSFULLY!")
            print("Check the 'outputs' directory for plots and Excel results.")
        else:
            print("\n❌ PLOTTING FAILED!")
            print("Check the debug output above for error details.")

    except ImportError as e:
        print(f"\n❌ PLOTTING IMPORT ERROR: Could not import plotting function: {e}")
        print("Make sure enhanced_physics_evolution_plotter_FINAL.py is in the same directory.")
    except Exception as e:
        print(f"\n❌ PLOTTING ERROR: {e}")
        import traceback
        traceback.print_exc()

    print("="*80 + "\n")
