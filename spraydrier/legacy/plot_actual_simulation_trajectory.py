#!/usr/bin/env python3
"""
Plot Actual Simulation Trajectory
==================================

Plots particle size evolution using the ACTUAL data from simulation.py
(not recreated or approximated).

This script runs simulation.py and plots the radius_history_um array
that contains the full physics-based trajectory.

Usage:
    python plot_actual_simulation_trajectory.py
    python plot_actual_simulation_trajectory.py last_inputs_batch_1.json
    python plot_actual_simulation_trajectory.py --excel DOE_Integral.xlsx --trial 1

Author: Claude
Date: December 2024
"""

import sys
import os
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path


def plot_trajectory_from_result(result, title="Particle Size Evolution"):
    """Plot the actual trajectory from simulation result."""
    
    # Extract history arrays
    time_history = result.get('time_history_s', [])
    radius_history = result.get('radius_history_um', [])
    
    if not time_history or not radius_history:
        print("ERROR: No history data found in simulation result!")
        print("Available keys:", list(result.keys())[:20])
        return None
    
    # Convert to useful units
    time_ms = np.array(time_history) * 1000  # seconds â†’ milliseconds
    diameter_um = np.array(radius_history) * 2  # radius â†’ diameter
    
    # Calculate shrinking rate
    if len(time_ms) > 1:
        shrink_rate = np.diff(diameter_um) / np.diff(time_ms)
        shrink_rate_time = time_ms[1:]
    else:
        shrink_rate = []
        shrink_rate_time = []
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # Main plot: Diameter evolution
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(time_ms, diameter_um, 'b-', linewidth=2, alpha=0.8)
    
    # Mark key points
    D_initial = diameter_um[0]
    D_final = diameter_um[-1]
    t_final = time_ms[-1]
    
    ax1.plot(0, D_initial, 'go', markersize=12, zorder=5, 
            label=f'Initial: {D_initial:.2f} Î¼m')
    ax1.plot(t_final, D_final, 'ro', markersize=12, zorder=5,
            label=f'Final: {D_final:.2f} Î¼m')
    
    # Find shrinkage completion (when D reaches ~95% of final)
    target_D = D_initial - 0.95 * (D_initial - D_final)
    idx_95 = np.argmin(np.abs(diameter_um - target_D))
    t_95 = time_ms[idx_95]
    ax1.axvline(t_95, color='red', linestyle='--', alpha=0.5,
               label=f'95% shrunk: {t_95:.1f} ms')
    
    ax1.set_xlabel('Time (milliseconds)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Particle Diameter (Î¼m)', fontweight='bold', fontsize=12)
    ax1.set_title(f'{title}\nFull Physics Trajectory from simulation.py', 
                 fontweight='bold', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Shrinking rate plot
    ax2 = plt.subplot(2, 2, 2)
    if len(shrink_rate) > 0:
        ax2.plot(shrink_rate_time, shrink_rate, 'r-', linewidth=1.5, alpha=0.8)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Time (milliseconds)', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Shrinking Rate (Î¼m/ms)', fontweight='bold', fontsize=12)
        ax2.set_title('Instantaneous Shrinking Rate', fontweight='bold', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    # Diameter vs fraction of time
    ax3 = plt.subplot(2, 2, 3)
    time_fraction = time_ms / time_ms[-1]
    ax3.plot(time_fraction, diameter_um, 'g-', linewidth=2, alpha=0.8)
    ax3.axhline(D_final, color='r', linestyle=':', alpha=0.5, label='Final size')
    ax3.set_xlabel('Fraction of Total Time', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Particle Diameter (Î¼m)', fontweight='bold', fontsize=12)
    ax3.set_title('Normalized Time Evolution', fontweight='bold', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Statistics
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    # Calculate statistics
    total_time = time_ms[-1]
    size_reduction = (D_initial - D_final) / D_initial * 100
    avg_rate = (D_initial - D_final) / total_time
    max_rate = np.min(shrink_rate) if len(shrink_rate) > 0 else 0
    
    # Additional physics from result
    Pe = result.get('Peclet Number', 'N/A')
    Pe_max = result.get('Max Peclet Number', 'N/A')
    v_s = result.get('Surface recession velocity', 'N/A')
    
    stats_text = f"""
SIMULATION STATISTICS

Data Points: {len(time_history):,}
Total Time: {total_time:.2f} ms

Initial Diameter: {D_initial:.3f} Î¼m
Final Diameter: {D_final:.3f} Î¼m
Size Reduction: {size_reduction:.1f}%

Average Shrinking Rate: {avg_rate:.4f} Î¼m/ms
Maximum Shrinking Rate: {max_rate:.4f} Î¼m/ms

Time to 95% Shrinkage: {t_95:.2f} ms
Fraction of Total: {t_95/total_time*100:.1f}%

PHYSICS PARAMETERS

PÃ©clet Number: {Pe}
Max PÃ©clet: {Pe_max}
Surface Recession: {v_s}

SOURCE: simulation.py
Full dÂ²-law with:
  â€¢ Concentration evolution
  â€¢ Diffusion attenuation
  â€¢ Tg-aware physics
"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Plot actual simulation.py trajectory with full physics'
    )
    parser.add_argument('input_file', nargs='?', 
                       help='JSON file with inputs (optional)')
    parser.add_argument('--excel', type=str,
                       help='Excel file with DOE parameters')
    parser.add_argument('--trial', type=int, default=0,
                       help='Trial number to plot (0-based, default: 0)')
    parser.add_argument('--output', type=str, default='simulation_actual_trajectory.png',
                       help='Output filename for plot (PNG)')
    parser.add_argument('--save-data', action='store_true',
                       help='Save trajectory data to Excel/CSV file')
    parser.add_argument('--data-format', type=str, choices=['xlsx', 'csv'], default='xlsx',
                       help='Format for trajectory data file (default: xlsx)')
    
    args = parser.parse_args()
    
    # Get inputs
    if args.excel:
        # Load from Excel
        print(f"Loading inputs from Excel: {args.excel}")
        try:
            df = pd.read_excel(args.excel, header=None)
            
            # Check if transposed (parameters in rows, trials in columns)
            # Look for 'Parameter' or 'Parameter / Batch ID' in first cell
            first_cell = str(df.iloc[0, 0]).strip().lower()
            if 'parameter' in first_cell:
                # Transposed format: parameters in column 0, trials in other columns
                # Row 0 is just headers ("Parameter", NaN, NaN...) - skip it
                # Then set column 0 (parameter names) as index and transpose
                df = df.iloc[1:].set_index(0).T  # Skip row 0, set col 0 as index, transpose
                df.index = range(len(df))  # Reset index to 0, 1, 2...
            
            # Get specific trial
            if args.trial >= len(df):
                print(f"ERROR: Trial {args.trial} not found. File has {len(df)} trials (0-{len(df)-1})")
                return
            
            trial_idx = args.trial
            batch_id = df.iloc[trial_idx].get('batch_id', f'Trial_{trial_idx}')
            print(f"Using trial {trial_idx}: {batch_id}")
            
            inputs = df.iloc[trial_idx].to_dict()
            
            # Convert to proper types and handle NaN
            for key, val in list(inputs.items()):
                if pd.isna(val) or val == 'nan':
                    inputs[key] = None
                else:
                    try:
                        # Try to convert to number if it looks like a number
                        val_str = str(val).strip()
                        if val_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                            inputs[key] = float(val_str) if '.' in val_str else int(float(val_str))
                        else:
                            inputs[key] = val_str
                    except:
                        inputs[key] = val
            
            # Check if this is an OUTPUT file (has results) or INPUT file (has parameters)
            # REQUIRED inputs that simulation.py directly requires (lines 130-142):
            # - Core processing: T1_C, feed_g_min, ds_conc, ds_mw, solids_frac
            # - Formulation: moni_conc, ds (drug substance name)
            # - Equipment: V_chamber_m3, m1_m3ph, nozzle_tip_d_mm, etc.
            # 
            # OPTIONAL/CALCULATED:
            # - D_solute: calculated from compounds (line 1199-1212) or defaults to 1e-10
            # - pH, viscosity, surface_tension: simulation can calculate/default
            # - D50_calc: this is an OUTPUT, not an input (can be improved by ML training)
            
            has_results = any(key in inputs for key in ['D50 (calculated with Moni)', 'Surface recession velocity', 'D50_calc', 'Peclet Number'])
            has_required_inputs = any(key in inputs for key in ['T1_C', 'feed_g_min', 'ds_conc', 'ds_mw', 'solids_frac', 'moni_conc', 'ds'])
            
            if has_results and not has_required_inputs:
                print("\n" + "="*70)
                print("âš ï¸  ERROR: WRONG FILE TYPE")
                print("="*70)
                print(f"\nYou provided: {args.excel}")
                print("This appears to be a RESULTS file (like DOE_output.xlsx)")
                print("\nResults files contain simulation OUTPUTS:")
                print("  âœ… D50_calc, Peclet Number, Surface recession velocity")
                print("\nBut they DON'T contain REQUIRED simulation INPUTS:")
                print("  âŒ T1_C, feed_g_min, ds_conc, ds_mw, solids_frac, moni_conc, ds")
                print("\nNote: These are calculated/optional:")
                print("  âš ï¸ D_solute (calculated from compounds)")
                print("  âš ï¸ pH, viscosity, surface_tension (simulation can default)")
                print("  âš ï¸ D50_calc (OUTPUT, not input - ML can improve estimate)")
                print("\nThis script needs to RE-RUN the simulation to get the full trajectory,")
                print("so it needs the original INPUT parameters.")
                print("\n" + "="*70)
                print("SOLUTIONS:")
                print("="*70)
                print("\n1. Use the INPUT file instead:")
                print(f"   python plot_actual_simulation_trajectory.py --excel DOE_Integral.xlsx --trial {args.trial}")
                print("\n2. Or use physics_based_evolution_plotter.py (works with results files):")
                print(f"   python physics_based_evolution_plotter.py {args.excel} --batch '{trial_idx}'")
                print("\n3. Or provide a JSON input file:")
                print(f"   python plot_actual_simulation_trajectory.py last_inputs_batch_{args.trial+1}.json")
                return
                    
        except Exception as e:
            print(f"Error loading Excel: {e}")
            return
            
    elif args.input_file and Path(args.input_file).exists():
        # Load from JSON
        print(f"Loading inputs from JSON: {args.input_file}")
        with open(args.input_file) as f:
            inputs = json.load(f)
    else:
        # Interactive mode - ask user for file
        print("\n" + "="*70)
        print("INTERACTIVE MODE")
        print("="*70)
        
        # Ask for input file
        excel_file = input("\nEnter input Excel filename (e.g., DOE_Integral.xlsx): ").strip()
        
        if not Path(excel_file).exists():
            print(f"Error: File '{excel_file}' not found")
            return
        
        print(f"Selected: {excel_file}")
        
        # Now load the Excel file
        print(f"\nLoading: {excel_file}")
        try:
            df = pd.read_excel(excel_file, header=None)
            
            # Check if transposed (parameters in rows, trials in columns)
            first_cell = str(df.iloc[0, 0]).strip().lower()
            if 'parameter' in first_cell:
                # Transposed format: parameters in column 0, trials in other columns
                # Row 0 is just headers ("Parameter", NaN, NaN...) - skip it
                df = df.iloc[1:].set_index(0).T  # Skip row 0, set col 0 as index, transpose
                df.index = range(len(df))  # Reset index to 0, 1, 2...
            
            # Show available trials with their batch IDs
            print(f"\nFound {len(df)} trials:")
            for i in range(len(df)):
                batch_id = df.iloc[i].get('batch_id', f'Trial_{i}')
                print(f"  {i}. {batch_id}")
            
            # Ask for trial
            try:
                trial_input = input(f"\nEnter trial number (0-{len(df)-1}): ").strip()
                trial_num = int(trial_input)
                
                if not (0 <= trial_num < len(df)):
                    print(f"Error: Trial number must be between 0 and {len(df)-1}")
                    return
                
                batch_id = df.iloc[trial_num].get('batch_id', f'Trial_{trial_num}')
                print(f"Selected: {trial_num} ({batch_id})")
                
            except (ValueError, KeyboardInterrupt, EOFError):
                print("\nInvalid input or cancelled by user")
                return
            
            inputs = df.iloc[trial_num].to_dict()
            
            # Convert to proper types and handle NaN
            for key, val in list(inputs.items()):
                if pd.isna(val) or val == 'nan':
                    inputs[key] = None
                else:
                    try:
                        # Try to convert to number if it looks like a number
                        val_str = str(val).strip()
                        if val_str.replace('.', '', 1).replace('-', '', 1).isdigit():
                            inputs[key] = float(val_str) if '.' in val_str else int(float(val_str))
                        else:
                            inputs[key] = val_str
                    except:
                        inputs[key] = val
            
            # Check if this is an OUTPUT file or INPUT file
            # REQUIRED: T1_C, feed_g_min, ds_conc, ds_mw, solids_frac, moni_conc, ds
            # CALCULATED: D_solute (from compounds), D50_calc (OUTPUT not input)
            # OPTIONAL: pH, viscosity, surface_tension
            has_results = any(key in inputs for key in ['D50 (calculated with Moni)', 'Surface recession velocity', 'D50_calc', 'Peclet Number'])
            has_required_inputs = any(key in inputs for key in ['T1_C', 'feed_g_min', 'ds_conc', 'ds_mw', 'solids_frac', 'moni_conc', 'ds'])
            
            if has_results and not has_required_inputs:
                print("\n" + "="*70)
                print("âš ï¸  ERROR: WRONG FILE TYPE")
                print("="*70)
                print(f"\nYou selected: {excel_file}")
                print("This appears to be a RESULTS file (simulation outputs)")
                print("\nResults files contain:")
                print("  âœ… D50_calc, Peclet Number, Surface recession velocity")
                print("\nBut they DON'T contain REQUIRED inputs:")
                print("  âŒ T1_C, feed_g_min, ds_conc, ds_mw, solids_frac, moni_conc, ds")
                print("\nNote: These are calculated/optional:")
                print("  âš ï¸ D_solute (calculated from compounds)")
                print("  âš ï¸ pH, viscosity, surface_tension (defaults available)")
                print("  âš ï¸ D50_calc (OUTPUT - ML training can improve)")
                print("\nThis script needs to RE-RUN the simulation to get the full trajectory.")
                print("\n" + "="*70)
                print("PLEASE SELECT AN INPUT FILE INSTEAD")
                print("="*70)
                print("\nLook for files like:")
                print("  â€¢ DOE_Integral.xlsx")
                print("  â€¢ full_input_sample.xlsx")
                print("  â€¢ Snapshot_training.xlsx")
                print("\nOr use: python physics_based_evolution_plotter.py (works with results files)")
                return
            
            # Ask for output filenames
            print("\n" + "="*70)
            print("OUTPUT FILES")
            print("="*70)
            
            # Ask for PNG output filename
            default_png = f"{Path(excel_file).stem}_Trial{trial_num}_trajectory.png"
            png_output = input(f"\nEnter PNG output filename (default: {default_png}): ").strip()
            if not png_output:
                png_output = default_png
            
            # Ask if they want to save data to Excel
            save_data = input("\nSave trajectory data to Excel? (y/n, default: n): ").strip().lower()
            excel_output = None
            if save_data == 'y':
                default_excel = png_output.replace('.png', '_data.xlsx')
                excel_output = input(f"Enter Excel output filename (default: {default_excel}): ").strip()
                if not excel_output:
                    excel_output = default_excel
            
            # Update args
            args.output = png_output
            args.save_data = (save_data == 'y')
            if excel_output:
                args.data_output = excel_output
            
            print(f"\nPNG output: {png_output}")
            if excel_output:
                print(f"Excel output: {excel_output}")
            
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # Run simulation
    print("\n" + "="*70)
    print("CHECKING REQUIRED PARAMETERS")
    print("="*70)
    
    # Check for TRULY required fields (no defaults available)
    # These are the parameters that simulation.py MUST have
    required_fields = {
        'batch_id': 'Batch identifier',
        'dryer': 'Dryer type (e.g., B290, B300, ProCept)',
        'V_chamber_m3': 'Chamber volume (mÂ³)',
        'cyclone_type': 'Cyclone type (e.g., high, std)',
        'gas1': 'Drying gas (air or nitrogen)',
        'T1_C': 'Drying gas temperature (Â°C)',
        'RH1': 'Drying gas RH (%)',
        'm1_m3ph': 'Drying gas flow (mÂ³/h)',
        'gas2': 'Atomizing gas (air or nitrogen)',
        'T2_C': 'Atomizing gas temperature (Â°C)',
        'RH2': 'Atomizing gas RH (%)',
        'atom_pressure': 'Atomizing pressure (bar)',
        'nozzle_tip_d_mm': 'Nozzle tip diameter (mm)',
        'nozzle_cap_d_mm': 'Nozzle cap diameter (mm)',
        'nozzle_level': 'Nozzle level (Y/N)',
        'ds': 'Drug substance name',
        'ds_conc': 'Drug concentration (mg/mL)',
        'ds_mw': 'Drug molecular weight (kDa)',
        'moni_conc': 'Moni concentration (mg/mL)',
        'solids_frac': 'Solids fraction (0-1)',
        'feed_g_min': 'Feed rate (g/min)',
        'rho_l': 'Solution density (kg/mÂ³)',
    }
    
    # These are OPTIONAL (have defaults in simulation.py)
    optional_fields = {
        'T_outlet_C': 'Outlet temperature (can be estimated)',
        'pH': 'pH (defaults to "not known")',
        'buffer': 'Buffer type (defaults to "none")',
        'buffer_conc': 'Buffer concentration (defaults to 0)',
        'stabilizer_A': 'Stabilizer (defaults to "none")',
        'stab_A_conc': 'Stabilizer conc (defaults to 0)',
        'additive_B': 'Additive B (defaults to "none")',
        'additive_B_conc': 'Additive B conc (defaults to 0)',
        'additive_C': 'Additive C (defaults to "none")',
        'additive_C_conc': 'Additive C conc (defaults to 0)',
        'D_solute': 'Diffusion coefficient (calculated from compounds)',
        'surface_tension': 'Surface tension (calculated from formulation)',
        'viscosity': 'Viscosity (calculated from ds and concentration)',
    }
    
    missing_fields = []
    for field, description in required_fields.items():
        if field not in inputs or inputs[field] is None or inputs[field] == '':
            missing_fields.append((field, description))
    
    if missing_fields:
        print(f"\nâš ï¸  MISSING {len(missing_fields)} REQUIRED PARAMETER(S):\n")
        for field, description in missing_fields:
            print(f"  âŒ {field:20s} - {description}")
        
        print("\n" + "="*70)
        print("WHAT TO DO:")
        print("="*70)
        print("\nYour Excel file is missing required parameters.")
        print("\nOptions:")
        print("1. Add missing columns to your Excel file")
        print("2. Provide defaults (I can add them if you tell me what values)")
        print("3. Use a complete input file (like full_input_sample.xlsx)")
        print("\n" + "="*70)
        return
    
    print("âœ… All required parameters present")
    
    # Show which optional parameters are missing (informational only)
    missing_optional = []
    for field, description in optional_fields.items():
        if field not in inputs or inputs[field] is None or inputs[field] == '':
            missing_optional.append((field, description))
    
    if missing_optional:
        print(f"\nâ„¹ï¸  Note: {len(missing_optional)} optional parameter(s) missing (will use defaults):")
        for field, description in missing_optional[:5]:  # Show first 5
            print(f"  âš ï¸  {field:20s} - {description}")
        if len(missing_optional) > 5:
            print(f"  ... and {len(missing_optional) - 5} more")
    
    print("\n" + "="*70)
    print("RUNNING FULL PHYSICS SIMULATION")
    print("="*70)
    
    try:
        from simulation import run_full_spray_drying_simulation
    except ImportError as e:
        print("ERROR: Cannot import simulation.py")
        print(f"Import error: {e}")
        print(f"\nCurrent directory: {os.getcwd()}")
        print(f"\nLooking for simulation.py in:")
        print(f"  {os.path.join(os.getcwd(), 'simulation.py')}")
        
        if os.path.exists('simulation.py'):
            print("\nâœ“ simulation.py EXISTS in current directory")
            print("  But Python cannot import it. Possible issues:")
            print("    - Syntax error in simulation.py")
            print("    - Missing dependencies")
        else:
            print("\nâœ— simulation.py NOT FOUND in current directory")
            print("  Make sure simulation.py is in the same directory as this script")
        return
    
    result = run_full_spray_drying_simulation(inputs)
    
    if result is None:
        print("\nERROR: Simulation failed")
        return
    
    # Plot trajectory
    print("\n" + "="*70)
    print("PLOTTING ACTUAL TRAJECTORY")
    print("="*70)
    
    batch_id = inputs.get('batch_id', 'Unknown')
    fig = plot_trajectory_from_result(result, title=f"Batch: {batch_id}")
    
    if fig is None:
        return
    
    # Save plot
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"\nâœ… Plot saved: {args.output}")
    
    # Save trajectory data if requested
    if args.save_data:
        # Create data filename from plot filename
        data_filename = args.output.replace('.png', f'_data.{args.data_format}')
        
        # Prepare data
        time_history = result['time_history_s']
        radius_history = result['radius_history_um']
        diameter_history = [r * 2 for r in radius_history]
        
        df_trajectory = pd.DataFrame({
            'Time_s': time_history,
            'Time_ms': [t * 1000 for t in time_history],
            'Radius_um': radius_history,
            'Diameter_um': diameter_history
        })
        
        # Add derived columns
        if len(time_history) > 1:
            # Shrinking rate
            shrink_rate = np.diff(diameter_history) / np.diff([t*1000 for t in time_history])
            shrink_rate = np.concatenate([[np.nan], shrink_rate])  # Pad first value
            df_trajectory['Shrinking_Rate_um_per_ms'] = shrink_rate
        
        # Add time fraction
        df_trajectory['Time_Fraction'] = df_trajectory['Time_s'] / df_trajectory['Time_s'].max()
        
        # Add physics parameters as metadata
        metadata = {
            'Batch_ID': inputs.get('batch_id', 'Unknown'),
            'Initial_Diameter_um': diameter_history[0],
            'Final_Diameter_um': diameter_history[-1],
            'Total_Time_ms': time_history[-1] * 1000,
            'Data_Points': len(time_history),
            'Peclet_Number': result.get('Peclet Number', 'N/A'),
            'Max_Peclet_Number': result.get('Max Peclet Number', 'N/A'),
            'Surface_Recession_Velocity': result.get('Surface recession velocity', 'N/A'),
        }
        
        # Save based on format
        if args.data_format == 'xlsx':
            with pd.ExcelWriter(data_filename, engine='openpyxl') as writer:
                # Save trajectory data
                df_trajectory.to_excel(writer, sheet_name='Trajectory', index=False)
                
                # Save metadata
                df_metadata = pd.DataFrame([metadata]).T
                df_metadata.columns = ['Value']
                df_metadata.to_excel(writer, sheet_name='Metadata')
            
            print(f"âœ… Trajectory data saved: {data_filename}")
            print(f"   Sheet 1: Trajectory ({len(df_trajectory):,} rows)")
            print(f"   Sheet 2: Metadata")
        else:  # CSV
            df_trajectory.to_csv(data_filename, index=False)
            
            # Save metadata to separate file
            metadata_filename = data_filename.replace('.csv', '_metadata.csv')
            pd.DataFrame([metadata]).T.to_csv(metadata_filename)
            
            print(f"âœ… Trajectory data saved: {data_filename} ({len(df_trajectory):,} rows)")
            print(f"âœ… Metadata saved: {metadata_filename}")
    
    # Print summary
    time_history = result['time_history_s']
    radius_history = result['radius_history_um']
    
    print(f"\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Data points: {len(time_history):,}")
    print(f"Total time: {time_history[-1]*1000:.2f} ms")
    print(f"Initial diameter: {radius_history[0]*2:.3f} Î¼m")
    print(f"Final diameter: {radius_history[-1]*2:.3f} Î¼m")
    print(f"Size reduction: {(1 - radius_history[-1]/radius_history[0])*100:.1f}%")
    print(f"\nThis is the ACTUAL simulation.py trajectory with full physics!")
    print(f"Not recreated, not approximated - this is the real dÂ²-law calculation.")
    
    # Show plot
    plt.show()
    

if __name__ == "__main__":
    main()
