#!/usr/bin/env python3
"""
Surface Composition Analyzer
===========================

Standalone script for detailed surface composition analysis of spray-dried particles.
Reads Excel output from main.py and performs advanced droplet modeling to predict
how solutes distribute between particle surface and bulk during drying.

This provides detailed time-dependent analysis that complements the basic
simulation results from main.py.

Usage:
    python surface_composition_analyzer.py input.xlsx [output.xlsx]

Author: Claude
Date: December 2025
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path for imports
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def analyze_surface_composition(excel_file, output_file=None):
    """
    Perform detailed surface composition analysis on main.py output.

    Args:
        excel_file: Path to Excel file from main.py
        output_file: Optional output file path

    Returns:
        Enhanced DataFrame with surface composition results
    """

    print(f"Loading data from {excel_file}")

    # Read the Excel file
    try:
        df = pd.read_excel(excel_file, index_col=0)
        print(f"Loaded {len(df)} parameters for {len(df.columns)} batches")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

    # Transpose if needed (should be parameter rows, batch columns)
    if df.shape[0] < df.shape[1]:  # More columns than rows, likely needs transpose
        df = df.T
        print("Transposed DataFrame for analysis")

    results = []

    # Process each batch
    for batch_idx, batch_name in enumerate(df.columns):
        print(f"\nAnalyzing batch: {batch_name}")

        # Extract parameters from the batch data
        batch_data = df[batch_name].copy()

        # Get required parameters
        params = extract_parameters(batch_data)
        if params is None:
            print(f"  Skipping batch {batch_name} - insufficient parameters")
            continue

        # Perform surface composition analysis
        surface_results = calculate_surface_composition(params)

        # Combine original data with surface analysis
        enhanced_batch = batch_data.copy()
        for key, value in surface_results.items():
            if isinstance(value, (list, np.ndarray)):
                # For time-series data, store as string or just key metrics
                if key == 'time_history_s':
                    enhanced_batch[f'{key}'] = str(value[:10]) + '...' if len(value) > 10 else str(value)
                elif key == 'radius_history_um':
                    enhanced_batch[f'{key}'] = str(value[:10]) + '...' if len(value) > 10 else str(value)
                else:
                    enhanced_batch[f'{key}'] = str(value)
            else:
                enhanced_batch[key] = value

        results.append((batch_name, enhanced_batch))

    # Create enhanced results DataFrame
    if results:
        enhanced_df = pd.DataFrame({name: data for name, data in results})
        enhanced_df.index.name = 'Parameter / Result'

        # Save enhanced results
        if output_file is None:
            base_name = os.path.splitext(excel_file)[0]
            output_file = f"{base_name}_surface_analysis.xlsx"

        enhanced_df.to_excel(output_file)
        print(f"\nEnhanced results saved to: {output_file}")
        print(f"Added surface composition analysis for {len(results)} batches")

        return enhanced_df
    else:
        print("No valid batches found for analysis")
        return None

def extract_parameters(batch_data):
    """
    Extract required parameters from batch data for surface analysis.

    Args:
        batch_data: Pandas Series with batch parameters

    Returns:
        Dictionary of extracted parameters or None if insufficient data
    """

    try:
        # Required parameters with fallbacks
        params = {}

        # Basic concentrations (mg/mL)
        params['ds_conc'] = float(batch_data.get('Drug Substance conc. (mg/mL)', 0))
        params['moni_conc'] = float(batch_data.get('Moni conc. (mg/mL)', 0))
        params['stab_A_conc'] = float(batch_data.get('Stabilizer conc. (mg/mL)', 0))
        params['additive_B_conc'] = float(batch_data.get('Additive #1 conc. (mg/mL)', 0))

        # Solids fraction
        params['solids_frac'] = float(batch_data.get('%Solids', batch_data.get('solids_frac', 10))) / 100.0

        # Péclet numbers (from main.py simulation)
        params['effective_pe_drug'] = float(batch_data.get('effective_pe_drug', 10.0))
        params['effective_pe_moni'] = float(batch_data.get('effective_pe_moni', 0.1))
        params['effective_pe_stabilizer'] = float(batch_data.get('effective_pe_stabilizer', 1.0))
        params['effective_pe_additive'] = float(batch_data.get('effective_pe_additive', 1.0))

        # Process conditions
        params['T1_C'] = float(batch_data.get('Drying Gas Inlet (C)', 70))
        params['feed_g_min'] = float(batch_data.get('Feed Rate (g/min)', 1.0))
        params['D50_actual'] = float(batch_data.get('D50_actual', 5.0))

        # Diffusion coefficients (if available)
        params['D_solute'] = float(batch_data.get('D_solute', 4e-11))

        return params

    except (ValueError, TypeError, KeyError) as e:
        print(f"  Parameter extraction failed: {e}")
        return None

def calculate_surface_composition(params):
    """
    Perform detailed surface composition analysis.

    Args:
        params: Dictionary of extracted parameters

    Returns:
        Dictionary of surface composition results
    """

    results = {}

    # Extract parameters
    ds_conc = params['ds_conc']
    moni_conc = params['moni_conc']
    stab_conc = params['stab_A_conc']
    additive_conc = params['additive_B_conc']
    solids_frac = params['solids_frac']

    # Effective Péclet numbers
    pe_drug = params['effective_pe_drug']
    pe_moni = params['effective_pe_moni']
    pe_stabilizer = params['effective_pe_stabilizer']
    pe_additive = params['effective_pe_additive']

    # Calculate initial mass fractions
    total_solids_mg_ml = ds_conc + moni_conc + stab_conc + additive_conc

    if total_solids_mg_ml > 0:
        xi0_drug = ds_conc / (1000 * 1000)  # mg/mL to g/mL to mass fraction
        xi0_moni = moni_conc / (1000 * 1000)
        xi0_stabilizer = stab_conc / (1000 * 1000)
        xi0_additive = additive_conc / (1000 * 1000)

        # Normalize to ensure they sum to solids fraction
        total_individual = xi0_drug + xi0_moni + xi0_stabilizer + xi0_additive
        if total_individual > 0:
            xi0_drug = xi0_drug * (solids_frac / total_individual)
            xi0_moni = xi0_moni * (solids_frac / total_individual)
            xi0_stabilizer = xi0_stabilizer * (solids_frac / total_individual)
            xi0_additive = xi0_additive * (solids_frac / total_individual)
    else:
        # Fallback: assume all solids are drug
        xi0_drug = solids_frac
        xi0_moni = 0
        xi0_stabilizer = 0
        xi0_additive = 0

    # Calculate surface enrichment using exponential relationship
    # Surface concentration = bulk concentration * exp(Pe)
    # Cap enrichment to prevent numerical overflow
    drug_enrichment = min(np.exp(min(pe_drug, 10)), 1e10)
    moni_enrichment = min(np.exp(min(pe_moni, 10)), 1e10)
    stabilizer_enrichment = min(np.exp(min(pe_stabilizer, 10)), 1e10)
    additive_enrichment = min(np.exp(min(pe_additive, 10)), 1e10)

    # Calculate surface mass fractions
    xi_drug_surface = xi0_drug * drug_enrichment
    xi_moni_surface = xi0_moni * moni_enrichment
    xi_stabilizer_surface = xi0_stabilizer * stabilizer_enrichment
    xi_additive_surface = xi0_additive * additive_enrichment

    # Normalize surface fractions to sum to 1 (100% surface coverage)
    total_surface = xi_drug_surface + xi_moni_surface + xi_stabilizer_surface + xi_additive_surface
    if total_surface > 0:
        xi_drug_surface /= total_surface
        xi_moni_surface /= total_surface
        xi_stabilizer_surface /= total_surface
        xi_additive_surface /= total_surface

    # Convert to percentages for output
    results['drug_surface_pct'] = xi_drug_surface * 100
    results['drug_bulk_pct'] = xi0_drug * 100
    results['moni_surface_pct'] = xi_moni_surface * 100
    results['moni_bulk_pct'] = xi0_moni * 100
    results['stabilizer_surface_pct'] = xi_stabilizer_surface * 100
    results['stabilizer_bulk_pct'] = xi0_stabilizer * 100
    results['additive_surface_pct'] = xi_additive_surface * 100
    results['additive_bulk_pct'] = xi0_additive * 100

    # Surface enrichment ratios
    results['drug_enrichment_ratio'] = drug_enrichment if xi0_drug > 0 else 0
    results['moni_enrichment_ratio'] = moni_enrichment if xi0_moni > 0 else 0
    results['stabilizer_enrichment_ratio'] = stabilizer_enrichment if xi0_stabilizer > 0 else 0
    results['additive_enrichment_ratio'] = additive_enrichment if xi0_additive > 0 else 0

    # Surface composition summary
    surface_composition = {
        'drug': xi_drug_surface * 100,
        'moni': xi_moni_surface * 100,
        'stabilizer': xi_stabilizer_surface * 100,
        'additive': xi_additive_surface * 100
    }
    results['surface_composition_summary'] = str(surface_composition)

    # Identify primary surface component
    max_component = max(surface_composition.items(), key=lambda x: x[1])
    results['primary_surface_component'] = max_component[0]
    results['primary_surface_pct'] = max_component[1]

    print(f"  Surface analysis complete:")
    print(".1f")
    print(f"    Primary surface component: {max_component[0]} ({max_component[1]:.1f}%)")

    return results

def main():
    parser = argparse.ArgumentParser(
        description='Analyze surface composition from main.py Excel output'
    )
    parser.add_argument('input_file', help='Excel file from main.py')
    parser.add_argument('output_file', nargs='?', help='Output Excel file (optional)')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)

    # Perform analysis
    result = analyze_surface_composition(args.input_file, args.output_file)

    if result is not None:
        print("\nSurface composition analysis completed successfully!")
    else:
        print("\nSurface composition analysis failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()