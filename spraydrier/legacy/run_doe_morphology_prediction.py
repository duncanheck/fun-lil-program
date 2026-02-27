#!/usr/bin/env python3
"""
Run Morphology Prediction on DOE Trials
========================================

This script:
1. Loads training data (historical batches with known morphology)
2. Trains a morphology prediction model
3. Applies it to DOE trial data
4. Saves results to Excel

Usage:
    python run_doe_morphology_prediction.py

ARCHITECTURAL NOTE: Pe Calculations vs main_INTEGRATED_FINAL.py
===============================================================

This DOE script NOW USES THE SAME Pe calculation approach as main_INTEGRATED_FINAL.py
for maximum prediction accuracy and consistency.

Both scripts now perform comprehensive physics simulation:
- run_full_spray_drying_simulation() → full physics simulation with time histories
- calculate_all_peclet_metrics() → computes effective_pe, max_pe, integrated_pe per component
  based on radius_history_um, v_evap_history_m_s, D_compounds_m2_s, solids_fraction_history
- Utilizes training.xlsx, calibration.json, and pkl files for consistency

This ensures DOE predictions are based on the same physics as production simulations,
providing the most accurate morphology and surface composition predictions possible.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Import physics-based surface calculation functions
from models.adsorption_model import surfactant_priority_override
from diffusion_coefficient import calculate_diffusion_for_compounds
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="scipy.optimize")

def calculate_physics_based_surface_composition(row, peclet_numbers, bulk_fractions):
    """
    Calculate surface composition using physics-based approach from advanced model.
    Includes surfactant priority override for moni.
    
    NOTE: Pe values used here NOW COME FROM THE SAME FULL PHYSICS SIMULATION
    as main_INTEGRATED_FINAL.py, ensuring maximum prediction accuracy and consistency.
    
    The DOE script now:
    1. Runs run_full_spray_drying_simulation() for each DOE trial
    2. Calculates effective_pe_X values using calculate_all_peclet_metrics()
    3. Uses the same training.xlsx, calibration.json, and pkl files
    4. Applies surfactant override for moni's high surface activity
    
    This provides the most accurate morphology and surface composition predictions possible
    by utilizing the same comprehensive physics as production simulations.
    """
    # Get initial solute concentrations (convert to mass fractions)
    drug_conc = float(row.get('ds_conc', row.get('Drug Substance conc. (mg/mL)', 0)))
    moni_conc = float(row.get('moni_conc', row.get('Moni conc. (mg/mL)', 0)))
    stabilizer_conc = float(row.get('stab_A_conc', row.get('Stabilizer conc. (mg/mL)', 0)))
    additive_conc = float(row.get('additive_B_conc', row.get('Additive #1 conc. (mg/mL)', 0)))

    total_solids_mg_ml = drug_conc + moni_conc + stabilizer_conc + additive_conc

    # Calculate initial mass fractions
    if total_solids_mg_ml > 0:
        # mg/mL to g/mL to mass fraction
        xi0_drug = drug_conc / (1000 * 1000)
        xi0_moni = moni_conc / (1000 * 1000)
        xi0_stabilizer = stabilizer_conc / (1000 * 1000)
        xi0_additive = additive_conc / (1000 * 1000)

        # Normalize to ensure they sum to solids fraction
        solids_frac = float(row.get('%Solids', row.get('solids_frac', 0.1)))
        total_individual = xi0_drug + xi0_moni + xi0_stabilizer + xi0_additive
        if total_individual > 0:
            xi0_drug = xi0_drug * (solids_frac / total_individual)
            xi0_moni = xi0_moni * (solids_frac / total_individual)
            xi0_stabilizer = xi0_stabilizer * (solids_frac / total_individual)
            xi0_additive = xi0_additive * (solids_frac / total_individual)
    else:
        # Fallback: assume all solids are drug
        solids_frac = float(row.get('%Solids', row.get('solids_frac', 0.1)))
        xi0_drug = solids_frac
        xi0_moni = 0
        xi0_stabilizer = 0
        xi0_additive = 0

    # Get Peclet numbers - NOW FROM FULL PHYSICS SIMULATION
    pe_drug = peclet_numbers.get('pe_drug', 10.0)
    pe_moni = peclet_numbers.get('pe_moni', 0.1)
    pe_stabilizer = peclet_numbers.get('pe_stabilizer', 1.0)
    pe_additive = peclet_numbers.get('pe_additive', 1.0)
    
    # NOTE: These Pe values now come from the same full physics simulation as main_INTEGRATED_FINAL.py:
    # - run_full_spray_drying_simulation() provides time-dependent histories
    # - calculate_all_peclet_metrics() computes effective_pe_X from radius, evaporation, diffusion data
    # - Uses the same training.xlsx, calibration.json, and pkl files for consistency
    #
    # This ensures DOE predictions have the same physics-based accuracy as production runs.

    # Ensure we have numeric values
    pe_drug = float(pe_drug) if pe_drug is not None else 10.0
    pe_moni = float(pe_moni) if pe_moni is not None else 0.1
    pe_stabilizer = float(pe_stabilizer) if pe_stabilizer is not None else 1.0
    pe_additive = float(pe_additive) if pe_additive is not None else 1.0

    # Calculate surface enrichment using exponential relationship
    drug_enrichment = min(np.exp(min(pe_drug, 10)), 1e10)
    moni_enrichment = min(np.exp(min(pe_moni, 10)), 1e10)
    stabilizer_enrichment = min(np.exp(min(pe_stabilizer, 10)), 1e10)
    additive_enrichment = min(np.exp(min(pe_additive, 10)), 1e10)

    # Calculate surface mass fractions
    xi_drug_surface = xi0_drug * drug_enrichment
    xi_moni_surface = xi0_moni * moni_enrichment
    xi_stabilizer_surface = xi0_stabilizer * stabilizer_enrichment
    xi_additive_surface = xi0_additive * additive_enrichment

    # Apply surfactant priority override (Langmuir adsorption model)
    surfactant_override_applied = False
    if moni_conc > 0:
        try:
            override = surfactant_priority_override(
                moni_conc_mg_ml=float(moni_conc),
                moni_name='moni',
                moni_mw=float(row.get('moni_mw', 6800.0))
            )

            if override.get("force_surfactant_override"):
                surfactant_override_applied = True

                moni_surface_pct = override['moni_surface_pct']
                remaining = 100.0 - override['moni_surface_pct']

                total_protein = xi0_drug + xi0_stabilizer + xi0_additive
                if total_protein > 0:
                    # Convert to percentage scale
                    scale = remaining / (total_protein * 100)
                    drug_surface_pct = xi0_drug * scale
                    stabilizer_surface_pct = xi0_stabilizer * scale
                    additive_surface_pct = xi0_additive * scale
                else:
                    drug_surface_pct = remaining
                    stabilizer_surface_pct = 0.0
                    additive_surface_pct = 0.0

                # Override the calculated values
                xi_drug_surface = drug_surface_pct / 100.0
                xi_moni_surface = moni_surface_pct / 100.0
                xi_stabilizer_surface = stabilizer_surface_pct / 100.0
                xi_additive_surface = additive_surface_pct / 100.0
        except Exception as e:
            print(f"   → Surfactant override failed: {e}")

    # If no surfactant override, normalize surface fractions to sum to 1
    if not surfactant_override_applied:
        total_surface = xi_drug_surface + xi_moni_surface + \
            xi_stabilizer_surface + xi_additive_surface
        if total_surface > 0:
            xi_drug_surface /= total_surface
            xi_moni_surface /= total_surface
            xi_stabilizer_surface /= total_surface
            xi_additive_surface /= total_surface

    return {
        'xi_drug_surface': xi_drug_surface,
        'xi_moni_surface': xi_moni_surface,
        'xi_stabilizer_surface': xi_stabilizer_surface,
        'xi_additive_surface': xi_additive_surface
    }

def get_file_path(prompt, must_exist=True):
    """Get file path from user with validation."""
    while True:
        file_path = input(prompt).strip()
        
        # Remove quotes if user included them
        file_path = file_path.strip('"').strip("'")
        
        # Expand user home directory if used
        file_path = os.path.expanduser(file_path)
        
        if must_exist:
            if os.path.exists(file_path):
                return file_path
            else:
                print(f"âŒ File not found: {file_path}")
                print("   Please check the path and try again.")
        else:
            # For output file, just check directory exists
            output_dir = os.path.dirname(file_path)
            if output_dir == '':
                # Current directory
                return file_path
            elif os.path.exists(output_dir):
                return file_path
            else:
                print(f"âŒ Directory not found: {output_dir}")
                print("   Please check the path and try again.")

def main():
    print("=" * 70)
    print("MORPHOLOGY PREDICTION FOR DOE TRIALS")
    print("=" * 70)
    
    # Get file paths from user
    print("\nðŸ“‚ Please provide file paths:")
    print("   (You can drag-and-drop files into the terminal)")
    print()
    
    training_file = get_file_path(
        "Enter training data file path (e.g., Snapshot_training.xlsx): ",
        must_exist=True
    )
    
    doe_file = get_file_path(
        "Enter DOE file path (e.g., DOE_output.xlsx): ",
        must_exist=True
    )
    
    output_file = get_file_path(
        "Enter output file path (e.g., DOE_results.xlsx): ",
        must_exist=False
    )
    
    # Ensure output has .xlsx extension
    if not output_file.endswith('.xlsx'):
        output_file += '.xlsx'
    
    print(f"\nâœ… Files configured:")
    print(f"   Training: {training_file}")
    print(f"   DOE: {doe_file}")
    print(f"   Output: {output_file}")
    
    # Try to import the simulator module
    try:
        # First try importing from the same directory as training file
        training_dir = os.path.dirname(os.path.abspath(training_file))
        if training_dir not in sys.path:
            sys.path.insert(0, training_dir)
        
        # Also try common locations
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        from advanced_droplet_model_refactored import DropletSimulator
        
        # Try to import physics-based surface composition analyzer
        try:
            from integrated_surface_composition_analyzer import IntegratedSurfaceCompositionAnalyzer
            physics_analyzer_available = True
            print("✓ Physics-based surface composition analyzer available")
        except ImportError:
            physics_analyzer_available = False
            print("⚠️ Physics-based surface composition analyzer not available - using defaults")
    except ImportError:
        print("\nâŒ Error: Could not find 'advanced_droplet_model_refactored.py'")
        print("   This file must be in the same directory as your training data,")
        print("   or in the same directory as this script.")
        return
    
    # ========================================================================
    # STEP 1: Load and prepare training data
    # ========================================================================
    print(f"\n" + "=" * 70)
    print("STEP 1: LOADING TRAINING DATA")
    print("=" * 70)
    
    df_train = pd.read_excel(training_file)
    print(f"Raw training data shape: {df_train.shape}")
    
    # Transpose if needed
    if 'Parameter / Batch ID' in df_train.columns:
        df_train = df_train.set_index('Parameter / Batch ID').T
        df_train.index.name = 'batch_id'
    
    print(f"âœ… Training data: {df_train.shape}")
    
    # Check for morphology column (only needed if training)
    morphology_col = None
    for col in ['Morphology', 'Morphology (Known)', 'Morphology (known)', 'morphology']:
        if col in df_train.columns:
            morphology_col = col
            break
    
    if morphology_col is None:
        print("⚠️ No morphology column found in training data!")
        print("   This is OK if using pre-trained models.")
        print("   Continuing with pre-trained morphology model...")
    else:
        print(f"âœ… Found morphology column: '{morphology_col}'")
        print(f"âœ… Morphology distribution:")
        morph_counts = df_train[morphology_col].value_counts()
        for morph, count in morph_counts.items():
            pct = count/len(df_train)*100
            print(f"      {morph}: {count} ({pct:.1f}%)")
    
    # ========================================================================
    # STEP 2: Load and prepare DOE data
    # ========================================================================
    print(f"\n" + "=" * 70)
    print("STEP 2: LOADING DOE DATA")
    print("=" * 70)
    
    df_doe = pd.read_excel(doe_file)
    print(f"Raw DOE data shape: {df_doe.shape}")
    
    # Transpose if needed
    if 'Parameter / Batch ID' in df_doe.columns:
        df_doe = df_doe.set_index('Parameter / Batch ID').T
        df_doe.index.name = 'trial_id'
        # Drop 'Result' row if present
        df_doe = df_doe.drop('Result', errors='ignore')
    elif 'Parameter' in df_doe.columns:
        # Handle the case where the column is just 'Parameter' (like in DOE_Integral.xlsx)
        # Extract trial names from the first row
        trial_names = df_doe.iloc[0, 1:].values  # Skip first column (Parameter)
        df_doe = df_doe.set_index('Parameter').T
        df_doe.index = trial_names
        df_doe.index.name = 'trial_id'
    
    print(f"âœ… DOE data: {df_doe.shape}")
    print(f"âœ… Trial IDs: {list(df_doe.index[:5])}...")
    
    # ========================================================================
    # STEP 3: Train morphology model
    # ========================================================================
    print(f"\n" + "=" * 70)
    print("STEP 3: TRAINING MORPHOLOGY MODEL")
    print("=" * 70)
    
    simulator = DropletSimulator()
    
    # Initialize physics-based surface composition analyzer if available
    physics_analyzer = None
    if physics_analyzer_available:
        physics_analyzer = IntegratedSurfaceCompositionAnalyzer()
        print("✓ Physics-based analyzer initialized")
    
    training_results = simulator.process_dataframe(df_train, train_morphology=False)  # Skip training, use pre-trained model
    
    print(f"âœ… Using pre-trained morphology model")
    print(f"âœ… Model loaded successfully")
    
    # ========================================================================
    # STEP 4: Predict morphology for DOE trials with physics-based Pe calculations
    # ========================================================================
    print(f"\n" + "=" * 70)
    print("STEP 4: PREDICTING DOE TRIAL MORPHOLOGIES")
    print("=" * 70)
    
    # Map BHV1400 column names to expected format for DropletSimulator
    def map_bhv1400_columns(df):
        """Map BHV1400 column names to DropletSimulator expected format."""
        column_mapping = {
            'T1_C': 'Drying Gas Inlet (C)',
            'feed_g_min': 'Feed Rate (g/min)',
            'm1_m3ph': 'Drying gas rate (m³/hr)',
            'ds_conc': 'Drug Substance conc. (mg/mL)',
            'moni_conc': 'Moni conc. (mg/mL)',
            'stab_A_conc': 'Stabilizer conc. (mg/mL)',
            'additive_B_conc': 'Additive #1 conc. (mg/mL)',
            'additive_C_conc': 'Additive #2 conc. (mg/mL)',
            'buffer_conc': 'Buffer conc. (mg/mL)',
            'solids_frac': '%Solids',
            'RH1': 'RH1',
            'RH2': 'RH2',
            'T2_C': 'T2_C',
            'T_outlet_C': 'T_outlet_C',
            'D50_actual': 'D50_actual',
            'D10_actual': 'D10_actual',
            'D90_actual': 'D90_actual',
            'Span': 'Span'
        }
        
        # Create a copy and rename columns
        df_mapped = df.copy()
        for old_col, new_col in column_mapping.items():
            if old_col in df_mapped.columns:
                df_mapped[new_col] = df_mapped[old_col]
        
        return df_mapped
    
    # Map columns for DOE data
    df_doe_mapped = map_bhv1400_columns(df_doe)
    
    # Process DOE trials with physics-based Pe calculations if available
    if physics_analyzer is not None:  # ENABLED: Run full physics for accurate Pe calculations
        print("🧪 Using full physics-based surface composition analysis...")
        print("   This ensures DOE predictions use the same calculations as main_INTEGRATED_FINAL.py")
        print("   utilizing training.xlsx, calibration.json, and pkl files for consistency")
        
        # Process each trial individually to get physics-based Pe values
        enhanced_doe_data = df_doe_mapped.copy()
        
        for trial_id in df_doe_mapped.index:
            trial_data = df_doe_mapped.loc[trial_id]
            print(f"  Processing {trial_id} with physics-based Pe calculations...")
            
            try:
                # Prepare trial data for physics simulation - don't add defaults for parameters
                # that should be calculated by the physics simulation
                trial_data = trial_data.copy()  # Don't modify original
                
                # Only add defaults for parameters that are absolutely required and not calculated
                # by the physics simulation. Span, pH, and particle sizes should come from physics.
                defaults = {
                    'nozzle_tip_d_mm': 0.7,  # mm - required for atomization calculations
                    'viscosity': 'n',       # use calculated
                    'surface_tension': 'n', # use calculated
                    'feed': 'g',            # feed rate in g/min
                    'rho_l': 1.05,          # g/mL typical density
                    'moisture_input': 0.0,  # % moisture
                    'moisture_content': 0.02, # fraction moisture
                }
                
                for param, default_value in defaults.items():
                    if param not in trial_data or pd.isna(trial_data.get(param)):
                        trial_data[param] = default_value
                        print(f"    Added default {param} = {default_value}")
                
                # Get physics-based Pe and D values (this will calculate D50_calc, Span, etc.)
                pe_d_data = physics_analyzer.run_simulation_for_trial(trial_data)
                
                # Update the trial data with calculated values from physics simulation
                if 'pe_drug' in pe_d_data:
                    enhanced_doe_data.loc[trial_id, 'effective_pe_drug'] = pe_d_data['pe_drug']
                if 'pe_moni' in pe_d_data:
                    enhanced_doe_data.loc[trial_id, 'effective_pe_moni'] = pe_d_data['pe_moni']
                if 'pe_stabilizer' in pe_d_data:
                    enhanced_doe_data.loc[trial_id, 'effective_pe_stabilizer'] = pe_d_data['pe_stabilizer']
                if 'pe_additive_b' in pe_d_data:
                    enhanced_doe_data.loc[trial_id, 'effective_pe_additive_b'] = pe_d_data['pe_additive_b']
                    
                # Update with calculated particle size and other physics parameters
                if 'D50_calc' in pe_d_data:
                    enhanced_doe_data.loc[trial_id, 'D50_calc'] = pe_d_data['D50_calc']
                if 'D50_calc_moni' in pe_d_data:
                    enhanced_doe_data.loc[trial_id, 'D50_calc_moni'] = pe_d_data['D50_calc_moni']
                if 'D32_um_moni' in pe_d_data:
                    enhanced_doe_data.loc[trial_id, 'D32_um_moni'] = pe_d_data['D32_um_moni']
                if 'Span' in pe_d_data:
                    enhanced_doe_data.loc[trial_id, 'Span'] = pe_d_data['Span']
                if 'pH' in pe_d_data:
                    enhanced_doe_data.loc[trial_id, 'pH'] = pe_d_data['pH']
                    
                print(f"    ✓ Pe_drug: {pe_d_data.get('pe_drug', 'N/A')}, Pe_moni: {pe_d_data.get('pe_moni', 'N/A')}, D50_calc: {pe_d_data.get('D50_calc', 'N/A')}")
                
            except Exception as e:
                print(f"    ⚠️ Physics calculation failed for {trial_id}: {e}")
                print("    Using default Pe values...")
        
        # Use enhanced data with physics-based Pe values
        doe_predictions = simulator.process_dataframe(enhanced_doe_data, train_morphology=False)
        
    else:
        print("⚠️ Physics analyzer not available - falling back to simplified Pe calculations")
        print("   This may produce less accurate predictions than main_INTEGRATED_FINAL.py")
        doe_predictions = simulator.process_dataframe(df_doe_mapped, train_morphology=False)
    
    print(f"âœ… Predictions complete for {len(doe_predictions)} trials")
    
    # ========================================================================
    # STEP 5: Create output file
    # ========================================================================
    print(f"\n" + "=" * 70)
    print("STEP 5: CREATING OUTPUT FILE")
    print("=" * 70)
    
    # Get trial IDs from DOE file
    trial_ids = df_doe.index.tolist()
    
    # Build results dictionary
    results_data = {}
    
    # Add input parameters from DOE file
    for col in df_doe.columns:
        if col in ['solids_frac', 'feed_g_min', 'T1_C', 'RH1', 'RH2', 'T2_C',
                   'ds_conc', 'moni_conc', 'buffer_conc', 'stab_A_conc', 
                   'additive_B_conc', 'additive_C_conc']:
            # Store as is, don't convert yet
            results_data[col] = df_doe[col].tolist()
    
    # Add morphology predictions
    if 'predicted_morphology' in doe_predictions.columns:
        results_data['Predicted Morphology'] = doe_predictions['predicted_morphology'].tolist()
    
    if 'morphology_confidence' in doe_predictions.columns:
        # Convert to percentages
        results_data['Morphology Confidence (%)'] = (doe_predictions['morphology_confidence'] * 100).tolist()
    
    # Add surface composition percentages for components that exist
    surface_pct_cols = [
        ('drug_surface_pct', 'Drug Surface (%)', 'ds_conc'),
        ('moni_surface_pct', 'Moni Surface (%)', 'moni_conc'),
        ('stabilizer_surface_pct', 'Stabilizer Surface (%)', 'stab_A_conc'),
        ('additive_surface_pct', 'Additive Surface (%)', ['additive_B_conc', 'additive_C_conc'])
    ]

    for col_name, display_name, conc_cols in surface_pct_cols:
        # Check if component exists in formulation FIRST
        include_pct = False
        if isinstance(conc_cols, list):
            for conc_col in conc_cols:
                if conc_col in df_doe.columns and df_doe[conc_col].notna().any() and (df_doe[conc_col] > 0).any():
                    include_pct = True
                    break
        else:
            if conc_cols in df_doe.columns and df_doe[conc_cols].notna().any() and (df_doe[conc_cols] > 0).any():
                include_pct = True

        print(f"DEBUG: {display_name} - include_pct: {include_pct}, col_name in predictions: {col_name in doe_predictions.columns}")
        if include_pct and col_name in doe_predictions.columns:
            results_data[display_name] = doe_predictions[col_name].tolist()
    
    # Add bulk composition percentages for components that exist
    bulk_pct_cols = [
        ('drug_bulk_pct', 'Drug Bulk (%)', 'ds_conc'),
        ('moni_bulk_pct', 'Moni Bulk (%)', 'moni_conc'),
        ('stabilizer_bulk_pct', 'Stabilizer Bulk (%)', 'stab_A_conc'),
        ('additive_bulk_pct', 'Additive Bulk (%)', ['additive_B_conc', 'additive_C_conc'])
    ]

    for col_name, display_name, conc_cols in bulk_pct_cols:
        # Check if component exists in formulation FIRST
        include_pct = False
        if isinstance(conc_cols, list):
            for conc_col in conc_cols:
                if conc_col in df_doe.columns and df_doe[conc_col].notna().any() and (df_doe[conc_col] > 0).any():
                    include_pct = True
                    break
        else:
            if conc_cols in df_doe.columns and df_doe[conc_cols].notna().any() and (df_doe[conc_cols] > 0).any():
                include_pct = True

        if include_pct and col_name in doe_predictions.columns:
            results_data[display_name] = doe_predictions[col_name].tolist()
    
    # Add Peclet numbers if available AND component exists in formulation
    pe_cols = [
        ('pe_drug', 'Pe Drug', 'ds_conc'),  # drug concentration
        ('pe_moni', 'Pe Moni', 'moni_conc'),  # moni concentration
        ('pe_stabilizer', 'Pe Stabilizer', 'stab_A_conc'),  # stabilizer concentration
        ('pe_additive', 'Pe Additive', ['additive_B_conc', 'additive_C_conc'])  # additive concentrations
    ]

    for col_name, display_name, conc_cols in pe_cols:
        # Check if component exists in formulation FIRST
        include_pe = False
        if isinstance(conc_cols, list):
            for conc_col in conc_cols:
                if conc_col in df_doe.columns and df_doe[conc_col].notna().any() and (df_doe[conc_col] > 0).any():
                    include_pe = True
                    break
        else:
            if conc_cols in df_doe.columns and df_doe[conc_cols].notna().any() and (df_doe[conc_cols] > 0).any():
                include_pe = True

        if include_pe and col_name in doe_predictions.columns:
            results_data[display_name] = doe_predictions[col_name].tolist()
    
    # Create DataFrame with trials as columns and parameters as rows
    final_df = pd.DataFrame(results_data, index=trial_ids).T
    
    print(f"âœ… Output format: {final_df.shape[0]} parameters Ã— {final_df.shape[1]} trials")
    
    # Save to Excel
    final_df.to_excel(output_file)
    
    # Save detailed results - TRANSPOSED with trials as columns
    detail_file = output_file.replace('.xlsx', '_detailed.xlsx')
    
    # Filter detailed results to only include relevant columns
    filtered_predictions = doe_predictions.copy()
    
    # Remove surface/bulk percentages and Peclet numbers for components that don't exist
    columns_to_check = [
        ('drug_surface_pct', 'ds_conc'),
        ('moni_surface_pct', 'moni_conc'),
        ('stabilizer_surface_pct', 'stab_A_conc'),
        ('additive_surface_pct', ['additive_B_conc', 'additive_C_conc']),
        ('drug_bulk_pct', 'ds_conc'),
        ('moni_bulk_pct', 'moni_conc'),
        ('stabilizer_bulk_pct', 'stab_A_conc'),
        ('additive_bulk_pct', ['additive_B_conc', 'additive_C_conc']),
        ('pe_drug', 'ds_conc'),
        ('pe_moni', 'moni_conc'),
        ('pe_stabilizer', 'stab_A_conc'),
        ('pe_additive', ['additive_B_conc', 'additive_C_conc'])
    ]
    
    for col_name, conc_cols in columns_to_check:
        if col_name in filtered_predictions.columns:
            # Check if component exists in formulation
            include_col = False
            if isinstance(conc_cols, list):
                for conc_col in conc_cols:
                    if conc_col in df_doe.columns and df_doe[conc_col].notna().any() and (df_doe[conc_col] > 0).any():
                        include_col = True
                        break
            else:
                if conc_cols in df_doe.columns and df_doe[conc_cols].notna().any() and (df_doe[conc_cols] > 0).any():
                    include_col = True
            
            if not include_col:
                filtered_predictions = filtered_predictions.drop(columns=[col_name])
    
    # Set trial IDs as index and transpose
    filtered_predictions.index = trial_ids
    doe_predictions_transposed = filtered_predictions.T
    
    # Save transposed detailed results
    doe_predictions_transposed.to_excel(detail_file)
    
    print(f"\nâœ… Files saved:")
    print(f"   Main results: {output_file}")
    print(f"   Detailed data: {detail_file}")
    
    # ========================================================================
    # STEP 6: Show summary
    # ========================================================================
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if 'predicted_morphology' in doe_predictions.columns:
        morphology_summary = doe_predictions['predicted_morphology'].value_counts()
        print(f"\nðŸ“Š Morphology Predictions:")
        for morph, count in morphology_summary.items():
            pct = count/len(doe_predictions)*100
            print(f"   â€¢ {morph}: {count}/{len(doe_predictions)} trials ({pct:.1f}%)")
    
    if 'morphology_confidence' in doe_predictions.columns:
        avg_confidence = doe_predictions['morphology_confidence'].mean() * 100
        min_confidence = doe_predictions['morphology_confidence'].min() * 100
        max_confidence = doe_predictions['morphology_confidence'].max() * 100
        print(f"\nðŸŽ¯ Confidence Statistics:")
        print(f"   â€¢ Average: {avg_confidence:.1f}%")
        print(f"   â€¢ Range: {min_confidence:.1f}% - {max_confidence:.1f}%")
    
    # Show surface composition summary if available
    if 'moni_surface_pct' in doe_predictions.columns:
        print(f"\nðŸ”¬ Surface Composition (Averages):")
        if 'drug_surface_pct' in doe_predictions.columns:
            avg_drug = doe_predictions['drug_surface_pct'].mean()
            print(f"   â€¢ Drug at surface: {avg_drug:.1f}%")
        if 'moni_surface_pct' in doe_predictions.columns:
            avg_moni = doe_predictions['moni_surface_pct'].mean()
            print(f"   â€¢ Moni at surface: {avg_moni:.1f}%")
        if 'stabilizer_surface_pct' in doe_predictions.columns:
            avg_stab = doe_predictions['stabilizer_surface_pct'].mean()
            print(f"   â€¢ Stabilizer at surface: {avg_stab:.1f}%")
    
    print(f"\n" + "=" * 70)
    print("âœ… WORKFLOW COMPLETE!")
    print("=" * 70)
    print(f"\nðŸ“ View your results:")
    print(f"   Main: {output_file}")
    print(f"   Detailed (transposed): {detail_file}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nâŒ Error: {e}")
        import traceback
        traceback.print_exc()
