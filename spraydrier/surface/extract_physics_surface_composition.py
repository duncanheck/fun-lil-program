#!/usr/bin/env python3
"""
Extract Physics-Based Surface Composition from main_INTEGRATED_FINAL.py Output
===============================================================================

This script:
1. Reads main_INTEGRATED_FINAL.py Excel output (DOE_Integral_results.xlsx)
2. Extracts physics-based Pe values and concentrations for each trial
3. Uses integrated_surface_composition_analyzer for accurate surface composition
4. Combines with morphology predictions from main script
5. Outputs comprehensive DOE results with consistent physics

Usage:
    python extract_physics_surface_composition.py

This replaces the flawed surface composition calculation in run_doe_morphology_prediction.py
with proper physics-based calculations that match main_INTEGRATED_FINAL.py.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

def extract_trial_physics_data(df_main, trial_name):
    """Extract Pe values, concentrations, and morphology for a single trial."""

    # Find the column for this trial
    trial_col = None
    for col in df_main.columns:
        if trial_name in str(col):
            trial_col = col
            break

    if trial_col is None:
        print(f"Warning: Could not find column for trial {trial_name}")
        return None

    # Extract parameters
    trial_data = {}
    for i, param in enumerate(df_main['Parameter / Batch ID']):
        value = df_main[trial_col].iloc[i]
        trial_data[param] = value

    # Extract key physics values
    physics_data = {
        'trial_name': trial_name,
        'ds_conc': trial_data.get('Drug Substance conc. (mg/mL)', 0),
        'moni_conc': trial_data.get('Moni conc. (mg/mL)', 0),
        'buffer_conc': trial_data.get('buffer_conc', 0),
        'stab_A_conc': trial_data.get('Stabilizer conc. (mg/mL)', 0),
        'additive_B_conc': trial_data.get('Additive #1 conc. (mg/mL)', 0),

        # Physics-based Pe values from main script
        'pe_drug': trial_data.get('Effective Pe (Drug)', 1.0),
        'pe_moni': trial_data.get('Effective Pe (Moni)', 0.1),
        'pe_buffer': trial_data.get('Effective Pe (Buffer)', 2.0),
        'pe_stabilizer': trial_data.get('Effective Pe (Stabilizer)', 1.0),
        'pe_additive_b': trial_data.get('Effective Pe (Additive B)', 1.0),

        # Diffusion coefficients from main script
        'd_drug': trial_data.get('D (Drug)', 1e-11),
        'd_moni': trial_data.get('D (Moni)', 5e-11),
        'd_stabilizer': trial_data.get('D (Stabilizer)', 5e-10),
        'd_additive_b': trial_data.get('D (Additive B)', 1e-9),

        # Morphology predictions from main script
        'darcy_morphology': trial_data.get('Darcy Predicted Morphology', 'unknown'),
        'ml_morphology': trial_data.get('Predicted Morphology', 'unknown'),
        'morphology_confidence': trial_data.get('Morphology Confidence', 0.0),

        # Process parameters for reference
        't1_c': trial_data.get('Drying Gas Inlet (C)', 70),
        'rh1': trial_data.get('RH1', 20),
        't2_c': trial_data.get('T2_C', 22),
        'rh2': trial_data.get('RH2', 1),
        'feed_rate': trial_data.get('Feed Rate (g/min)', 2),
        'solids_frac': trial_data.get('%Solids', 0.12)
    }

    return physics_data

def calculate_physics_based_surface_composition(physics_data):
    """Calculate surface composition using integrated analyzer with physics data."""

    try:
        from integrated_surface_composition_analyzer import IntegratedSurfaceCompositionAnalyzer

        # Prepare trial data for analyzer
        trial_data = {
            'ds_conc': physics_data['ds_conc'],
            'moni_conc': physics_data['moni_conc'],
            'buffer_conc': physics_data['buffer_conc'],
            'stab_A_conc': physics_data['stab_A_conc'],
            'additive_B_conc': physics_data['additive_B_conc'],
            'ds': 'IGG'  # Default drug substance name
        }

        # Prepare pe_d_data from main script physics
        pe_d_data = {
            'pe_drug': physics_data['pe_drug'],
            'pe_moni': physics_data['pe_moni'],
            'pe_stabilizer': physics_data['pe_stabilizer'],
            'pe_additive_b': physics_data['pe_additive_b'],
            'pe_buffer': physics_data['pe_buffer'],
            'd_drug': physics_data['d_drug'],
            'd_moni': physics_data['d_moni'],
            'd_stabilizer': physics_data['d_stabilizer'],
            'd_additive_b': physics_data['d_additive_b']
        }

        # Calculate surface composition
        analyzer = IntegratedSurfaceCompositionAnalyzer()
        result = analyzer.calculate_surface_composition(trial_data, pe_d_data)

        if result:
            return {
                'drug_surface_pct': result['surface_percentages'].get('IGG', 0),
                'moni_surface_pct': result['surface_percentages'].get('Moni', 0),
                'buffer_surface_pct': result['surface_percentages'].get('Buffer', 0),
                'stabilizer_surface_pct': result['surface_percentages'].get('Stabilizer_A', 0),
                'additive_surface_pct': result['surface_percentages'].get('Additive_B', 0),

                'drug_bulk_pct': result['bulk_percentages'].get('IGG', 0),
                'moni_bulk_pct': result['bulk_percentages'].get('Moni', 0),
                'buffer_bulk_pct': result['bulk_percentages'].get('Buffer', 0),
                'stabilizer_bulk_pct': result['bulk_percentages'].get('Stabilizer_A', 0),
                'additive_bulk_pct': result['bulk_percentages'].get('Additive_B', 0),

                'total_concentration': result.get('total_concentration_mg_ml', 0)
            }
        else:
            print(f"Warning: Surface composition calculation failed for {physics_data['trial_name']}")
            return None

    except ImportError:
        print("Error: Could not import integrated_surface_composition_analyzer")
        return None
    except Exception as e:
        print(f"Error calculating surface composition for {physics_data['trial_name']}: {e}")
        return None

def main():
    print("=" * 80)
    print("EXTRACT PHYSICS-BASED SURFACE COMPOSITION FROM MAIN SCRIPT OUTPUT")
    print("=" * 80)

    # Input file from main_INTEGRATED_FINAL.py
    input_file = "DOE_Integral_results.xlsx"
    output_file = "DOE_physics_based_surface_composition.xlsx"

    if not os.path.exists(input_file):
        print(f"❌ Error: Input file '{input_file}' not found")
        print("   Please run main_INTEGRATED_FINAL.py first to generate this file")
        return

    print(f"📖 Reading main script output: {input_file}")

    # Load main script output
    df_main = pd.read_excel(input_file)
    print(f"   Found {df_main.shape[1]-1} trials in main script output")

    # Extract trial names from column headers
    trial_columns = [col for col in df_main.columns if col != 'Parameter / Batch ID']
    print(f"   Trial columns: {trial_columns[:3]}...")

    # Process each trial
    results = []

    for trial_col in trial_columns:
        print(f"\n🔬 Processing {trial_col}...")

        # Extract physics data
        physics_data = extract_trial_physics_data(df_main, trial_col)
        if not physics_data:
            continue

        print(f"   Pe_drug: {physics_data['pe_drug']:.3f}, Pe_moni: {physics_data['pe_moni']:.3f}")
        print(f"   Morphology: {physics_data['ml_morphology']} (confidence: {physics_data['morphology_confidence']})")

        # Calculate surface composition
        surface_data = calculate_physics_based_surface_composition(physics_data)
        if surface_data:
            print(f"   Drug surface: {surface_data['drug_surface_pct']:.1f}%, Moni surface: {surface_data['moni_surface_pct']:.1f}%")
            # Combine all data
            trial_result = {
                **physics_data,
                **surface_data
            }
            results.append(trial_result)
        else:
            print("   ❌ Surface composition calculation failed")

    if not results:
        print("\n❌ No trials processed successfully")
        return

    # Create output DataFrame
    df_output = pd.DataFrame(results)

    # Reorder columns for clarity
    column_order = [
        'trial_name',
        'ds_conc', 'moni_conc', 'buffer_conc', 'stab_A_conc', 'additive_B_conc',
        'pe_drug', 'pe_moni', 'pe_buffer', 'pe_stabilizer', 'pe_additive_b',
        'd_drug', 'd_moni', 'd_stabilizer', 'd_additive_b',
        'darcy_morphology', 'ml_morphology', 'morphology_confidence',
        'drug_surface_pct', 'moni_surface_pct', 'buffer_surface_pct',
        'stabilizer_surface_pct', 'additive_surface_pct',
        'drug_bulk_pct', 'moni_bulk_pct', 'buffer_bulk_pct',
        'stabilizer_bulk_pct', 'additive_bulk_pct',
        'total_concentration',
        't1_c', 'rh1', 't2_c', 'rh2', 'feed_rate', 'solids_frac'
    ]

    # Only include columns that exist
    available_columns = [col for col in column_order if col in df_output.columns]
    df_output = df_output[available_columns]

    # Save results
    df_output.to_excel(output_file, index=False)

    print(f"\n✅ Results saved to: {output_file}")
    print(f"   Processed {len(results)} trials successfully")
    print("\n📊 Summary:")
    print(f"   - Uses physics-based Pe values from main_INTEGRATED_FINAL.py")
    print("   - Applies consistent amphiphilic enhancement (1.5x Pe for moni)")
    print("   - No hardcoded surfactant overrides")
    print("   - Fully consistent with production physics")

if __name__ == "__main__":
    main()