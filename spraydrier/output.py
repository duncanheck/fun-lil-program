import pandas as pd
import numpy as np
import os
import traceback  # Added to fix NameError in except block
from openpyxl import load_workbook

# ────────────────────────────────────────────────
# Top-level definitions (exported for import in engine.py)
# ────────────────────────────────────────────────

# Your preferred_order list (full from your message - used for row order enforcement)
preferred_order = [
    'batch_id', 'ds', 'dryer', 'V_chamber_m3', 'cyclone_type', 'gas1',
    'Drying Gas Inlet (C)', 'RH1', 'Drying gas rate (m³/hr)', 'gas2', 'T2_C', 'RH2',
    'atom_pressure', 'nozzle_tip_d_mm', 'nozzle_cap_d_mm', 'nozzle_level', 'T_outlet_C',
    'measured_RH_out', 'D_solute', '%Solids', 'Feed Rate (g/min)', 'feed_mL_min', 'rho_l',
    'Estimated feed viscosity (Pa·s)', 'Estimated Feed Surface Tension (N/m)', 'rho_final',
    'Drug Substance conc. (mg/mL)', 'Moni conc. (mg/mL)', 'Feed solution pH',
    'buffer', 'buffer_conc', 'Stabilizer', 'Stabilizer conc. (mg/mL)',
    'Additive #1', 'Additive #1 conc. (mg/mL)',
    'D10_actual', 'D50_actual', 'D90_actual', 'Morphology', 'Span',
    'Measured total moisture (%)', 'measured_surface_moisture', 'measured_bound_moisture',
    'atom_gas_mass_flow', 'GLR', 'u_ag', 'Outlet Relative Humidity',
    't_dry', 'Surface recession velocity', 'Peclet Number', 'Marangoni Number',
    'Reynolds number', 'Nusselt number Nu', 'Sherwood number Sh',
    'Heat Transfer coefficient', 'Mass Transfer coefficient', 'rho_v_droplet',
    'condensed_total_kg_ph', 'Required Inlet Temperature',
    'est_T_outlet_C', 'est_RH_pct', 'delta_RH_pct',
    'D32 diameter (With Moni)', 'D50 calculated (with Moni)', 'D50 calculated (No Moni)',
    'Predicted powder moisture content (%)', 'Efficiency', 'Predicted Morphology', 'Morphology Confidence', 'Shell Formation Time (s)',
    'Darcy Internal Pressure (Pa)', 'Darcy Pressure Drop (Pa)', 'Darcy Pi Ratio', 'Darcy Predicted Morphology',
    'Darcy Morphology Mechanism', 'Darcy Shell Thickness (μm)', 'Darcy Permeability (m²)', 'Darcy Number',
    'Effective Péclet Number', 'Maximum Péclet Number', 'Integrated Péclet Number',
    'Effective Pe (Drug)', 'Max Pe (Drug)', 'Effective Pe (Moni)', 'Max Pe (Moni)', 'Effective Pe (Stabilizer)', 'Max Pe (Stabilizer)'
]

# Your full label_map (expanded with missing mappings from your message)
label_map = {
    'batch_id': 'batch_id',
    'ds': 'ds',
    'dryer': 'dryer',
    'V_chamber_m3': 'V_chamber_m3',
    'cyclone_type': 'cyclone_type',
    'gas1': 'gas1',
    'T1_C': 'Drying Gas Inlet (C)',
    'RH1': 'RH1',
    'm1_m3ph': 'Drying gas rate (m³/hr)',
    'gas2': 'gas2',
    'T2_C': 'T2_C',
    'RH2': 'RH2',
    'atom_pressure': 'atom_pressure',
    'nozzle_tip_d_mm': 'nozzle_tip_d_mm',
    'nozzle_cap_d_mm': 'nozzle_cap_d_mm',
    'nozzle_level': 'nozzle_level',
    'T_outlet_C': 'T_outlet_C',
    'measured_RH_out': 'measured_RH_out',
    'D_solute': 'D_solute',
    'solids_frac': '%Solids',
    'feed_g_min': 'Feed Rate (g/min)',
    'feed_mL_min': 'feed_mL_min',
    'rho_l': 'rho_l',
    'viscosity_moni': 'Estimated feed viscosity (Pa·s)',
    'surface_tension_moni': 'Estimated Feed Surface Tension (N/m)',
    'rho_final': 'rho_final',
    'ds_conc': 'Drug Substance conc. (mg/mL)',
    'moni_conc': 'Moni conc. (mg/mL)',
    'pH': 'Feed solution pH',
    'buffer': 'buffer',
    'buffer_conc': 'buffer_conc',
    'stabilizer_A': 'Stabilizer',
    'stab_A_conc': 'Stabilizer conc. (mg/mL)',
    'additive_B': 'Additive #1',
    'additive_B_conc': 'Additive #1 conc. (mg/mL)',
    'D10_actual': 'D10_actual',
    'D50_actual': 'D50_actual',
    'D90_actual': 'D90_actual',
    'Morphology Confidence': 'Morphology Confidence',
    'Span': 'Span',
    'measured_total_computed': 'Measured total moisture (%)',
    'spm': 'measured_surface_moisture',
    'bmp': 'measured_bound_moisture',
    'atom_gas_mass_flow': 'atom_gas_mass_flow',
    'GLR': 'GLR',
    'u_ag': 'u_ag',
    'RH_out': 'Outlet Relative Humidity',
    't_dry': 't_dry',
    'v_evap': 'Surface recession velocity',
    'Ma': 'Marangoni Number',
    'Re_droplet': 'Reynolds number',
    'Nu': 'Nusselt number Nu',
    'Sh': 'Sherwood number Sh',
    'h': 'Heat Transfer coefficient',
    'k_m': 'Mass Transfer coefficient',
    'rho_v_droplet': 'rho_v_droplet',
    'condensed_total_kg_ph': 'condensed_total_kg_ph',
    'T_inlet_req_C': 'Required Inlet Temperature',
    'est_T_outlet_C': 'est_T_outlet_C',
    'est_RH_pct': 'est_RH_pct',
    'delta_RH_pct': 'delta_RH_pct',
    'D32_um_moni': 'D32 diameter (With Moni)',
    'D50_calc_moni': 'D50 calculated (with Moni)',
    'D50_calc': 'D50 calculated (No Moni)',
    'moisture_predicted': 'Predicted powder moisture content (%)',
    'Efficiency': 'Efficiency',
    'Predicted Morphology': 'Predicted Morphology',
    'shell_formation_time': 'Shell Formation Time (s)',
    'darcy_P_internal_Pa': 'Darcy Internal Pressure (Pa)',
    'darcy_Delta_P_Pa': 'Darcy Pressure Drop (Pa)',
    'darcy_Pi_ratio': 'Darcy Pi Ratio',
    'darcy_morphology_predicted': 'Darcy Predicted Morphology',
    'darcy_morphology_mechanism': 'Darcy Morphology Mechanism',
    'darcy_shell_thickness_um': 'Darcy Shell Thickness (μm)',
    'darcy_permeability_m2': 'Darcy Permeability (m²)',
    'darcy_Darcy_number': 'Darcy Number',
    # === Centralized Péclet metrics ===
    'effective_pe': 'Effective Péclet Number',
    'max_pe': 'Maximum Péclet Number',
    'integrated_pe': 'Integrated Péclet Number',
    'effective_pe_drug': 'Effective Pe (Drug)',
    'max_pe_drug': 'Max Pe (Drug)',
    'effective_pe_moni': 'Effective Pe (Moni)',
    'max_pe_moni': 'Max Pe (Moni)',
    'effective_pe_stabilizer': 'Effective Pe (Stabilizer)',
    'max_pe_stabilizer': 'Max Pe (Stabilizer)',
    'effective_pe_buffer': 'Effective Pe (Buffer)',
    'max_pe_buffer': 'Max Pe (Buffer)',
    'effective_pe_additive_B': 'Effective Pe (Additive B)',
    'max_pe_additive_B': 'Max Pe (Additive B)',
    'effective_pe_additive_C': 'Effective Pe (Additive C)',
    'max_pe_additive_C': 'Max Pe (Additive C)',
    'integrated_pe_drug': 'Integrated Pe (Drug)',
    'integrated_pe_moni': 'Integrated Pe (Moni)',
    'integrated_pe_stabilizer': 'Integrated Pe (Stabilizer)',
    'integrated_pe_additive_B': 'Integrated Pe (Additive B)',
    'integrated_pe_additive_C': 'Integrated Pe (Additive C)',
    'D_drug': 'D (Drug)',
    'D_moni': 'D (Moni)',
    'D_stabilizer': 'D (Stabilizer)',
    'D_additive_B': 'D (Additive B)',
    'D_additive_C': 'D (Additive C)',
    'D_buffer': 'D (Buffer)',
    'Final_Tg_C': 'Final Tg (Powder)',
    'Tg_drug': 'Tg (Drug)',
    'Tg_moni': 'Tg (Moni)',
    'Tg_buffer': 'Tg (Buffer)',
    'Tg_stabilizer': 'Tg (Stabilizer)',
    'Tg_additive_B': 'Tg (Additive B)',
    'Tg_additive_C': 'Tg (Additive C)',
    'Re_g': 'Re_g',
    'calibration_factor': 'calibration_factor',
    'shell_formation_fraction': 'Shell Formation Fraction',
    'morphology_predicted': 'Predicted Morphology (ML)',
    'morphology_known': 'Morphology (known)',
    # Add more mappings as needed
}

def save_output(result, inputs, filename="Upperton_data.xlsx", input_param_order=None):
    """
    Save results with clean, readable row labels by mapping internal keys to display names.
    Enforces preferred_order when provided for perfect alignment.
    Matches Upperton_data (1).xlsx style: parameters as rows, batches as columns.
    """
    # === DEBUG: Log inputs ===
    print(f"\n=== DEBUG save_output ===")
    print(f"input_param_order type: {type(input_param_order)}")
    print(f"input_param_order is None: {input_param_order is None}")
    if input_param_order is not None:
        print(f"input_param_order length: {len(input_param_order)}")
        print(f"First 5 params: {input_param_order[:5]}")
    print(f"=========================\n")

    # Convert tuple result to dict if needed
    if isinstance(result, tuple) and len(result) == 2:
        param_names, values = result
        result_dict = dict(zip(param_names, values))
    elif isinstance(result, tuple):
        # Raw tuple from simulation.py – use SIMULATION_PARAM_NAMES from enhanced plotter
        try:
            from enhanced_physics_evolution_plotter_FINAL import SIMULATION_PARAM_NAMES
            result_dict = dict(zip(SIMULATION_PARAM_NAMES, result))
        except ImportError:
            result_dict = {f"param_{i}": v for i, v in enumerate(result)}
    else:
        result_dict = result

    # Combine all data (calculated overrides inputs where keys match)
    data_source = {**inputs, **result_dict}

    # === Build rows in strict order ===
    rows = []
    seen_labels = set()

    reverse_label_map = {v: k for k, v in label_map.items()}

    # Use preferred_order if provided (full control over row sequence)
    if input_param_order is not None:
        for display_label in input_param_order:
            if display_label in seen_labels:
                continue

            # Get internal key from label_map reverse
            internal_key = reverse_label_map.get(display_label.strip(), display_label)

            # Prefer calculated result over input
            value = result_dict.get(internal_key, inputs.get(internal_key, "Not Available"))

            # Special handling: moisture_predicted fraction → percentage
            if internal_key == 'moisture_predicted' and isinstance(value, (int, float)) and value < 1:
                value = value * 100

            rows.append((display_label, value))
            seen_labels.add(display_label)

    # Fallback: if no preferred_order, add inputs first, then calculated
    else:
        for key, value in inputs.items():
            display = label_map.get(key, key)
            if display not in seen_labels:
                rows.append((display, value))
                seen_labels.add(display)

        for key, value in result_dict.items():
            if key not in inputs:  # Avoid duplicating inputs
                display = label_map.get(key, key)
                if display not in seen_labels:
                    rows.append((display, value))
                    seen_labels.add(display)

    batch_id = inputs.get("batch_id", "unknown")

    # Create DataFrame
    index = [row[0] for row in rows]
    values = [row[1] for row in rows]
    new_data = pd.DataFrame({batch_id: values}, index=index)
    new_data.index.name = "Parameter / Batch ID"

    # Merge with existing file if present
    if os.path.exists(filename):
        try:
            existing = pd.read_excel(filename, index_col=0)
            updated = pd.concat([existing, new_data], axis=1)
        except Exception as e:
            print(f"Could not read existing file ({e}). Creating new.")
            updated = new_data
    else:
        updated = new_data

    # Write with robust error handling
    try:
        updated.to_excel(filename, engine='openpyxl')
        print(f"Successfully saved: {filename} (clean readable labels, all data preserved)")
    except PermissionError as e:
        print(f"[WARN] Permission denied on {filename} - close any open Excel!")
        print(f"  Error: {e}")
        # Automatic backup save
        alt_filename = filename.replace('.xlsx', '_backup.xlsx')
        print(f"  Trying backup save to: {alt_filename}")
        try:
            updated.to_excel(alt_filename, engine='openpyxl')
            print(f"Backup saved successfully to {alt_filename}")
        except Exception as backup_e:
            print(f"Backup save also failed: {backup_e}")
    except Exception as e:
        print(f"Error writing Excel: {e}")
        traceback.print_exc()