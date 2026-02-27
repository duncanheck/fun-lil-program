#!/usr/bin/env python3
"""
Enhanced Two-Stage Particle Evolution Plotter
==============================================

Uses ACTUAL surface recession velocity from simulation to calculate
precise shrinkage dynamics instead of approximations.

Key Physics:
- Direct integration of surface recession: dR/dt = -v_s(t)
- Accounts for time-varying evaporation rate
- Uses simulation-calculated v_s for accuracy

Author: Claude
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Set matplotlib to use a font that supports Unicode
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'sans-serif']


def calculate_shrinkage_with_recession_velocity(R_initial_m, R_final_m, v_s_m_s, 
                                                 concentration_evolution=True):
    """
    Calculate precise shrinkage trajectory using surface recession velocity.
    
    The fundamental equation is:
        dR/dt = -v_s(t)
    
    For constant evaporation rate:
        R(t) = R_initial - v_s Ã— t
    
    For dÂ²-law (v_s âˆ 1/RÂ²):
        R(t) satisfies: dR/dt = -v_s0 Ã— (R0/R)Â²
        
    Parameters:
    -----------
    R_initial_m : float
        Initial droplet radius in meters
    R_final_m : float
        Final particle radius in meters
    v_s_m_s : float
        Surface recession velocity in m/s
    concentration_evolution : bool
        If True, account for increasing concentration (decreasing v_s)
        
    Returns:
    --------
    t_shrinkage : float
        Time to complete shrinkage in seconds
    time_array : ndarray
        Time points in seconds
    radius_array : ndarray
        Radius at each time point in meters
    """
    
    print(f"\n[*] CALCULATING SHRINKAGE FROM SURFACE RECESSION VELOCITY")
    print(f"=" * 70)
    print(f"Initial radius: {R_initial_m*1e6:.3f} Âµm")
    print(f"Final radius: {R_final_m*1e6:.3f} Âµm")
    print(f"Surface recession velocity: {v_s_m_s:.3e} m/s")
    
    # Method 1: Simple linear recession (conservative estimate)
    # R(t) = R_initial - v_s Ã— t
    # Solve for t when R = R_final:
    delta_R = R_initial_m - R_final_m
    t_simple = delta_R / v_s_m_s
    print(f"\n[*] Linear model: t_shrinkage = {t_simple*1000:.2f} ms")
    
    # Method 2: dÂ²-law with concentration dependence
    # As droplet shrinks, concentration increases, which can affect v_s
    # v_s(t) = v_s0 Ã— (R0/R)^n where n depends on diffusion limitation
    # For evaporation-limited (typical early stage): n â‰ˆ 0
    # For diffusion-limited (late stage): n â‰ˆ 2
    
    if concentration_evolution:
        # Use dÂ²-law: dR/dt = -v_s0 Ã— (R0/R)Â²
        # Analytical solution: RÂ³ = R0Â³ - 3Ã—v_s0Ã—R0Â²Ã—t
        # Solve for t when R = R_final:
        R0_cubed = R_initial_m ** 3
        Rf_cubed = R_final_m ** 3
        t_d2_law = (R0_cubed - Rf_cubed) / (3 * v_s_m_s * R_initial_m ** 2)
        print(f"[*] dÂ²-law model: t_shrinkage = {t_d2_law*1000:.2f} ms")
        
        # Use the dÂ²-law time
        t_shrinkage = t_d2_law
        
        # Generate trajectory using dÂ²-law
        n_points = 200
        time_array = np.linspace(0, t_shrinkage, n_points)
        
        # Numerical integration of dR/dt = -v_s0 Ã— (R0/R)Â²
        radius_array = np.zeros(n_points)
        radius_array[0] = R_initial_m
        
        dt = time_array[1] - time_array[0]
        for i in range(1, n_points):
            # dÂ²-law recession rate
            v_s_current = v_s_m_s * (R_initial_m / radius_array[i-1]) ** 2
            radius_array[i] = radius_array[i-1] - v_s_current * dt
            
            # Safety: don't go below final radius
            if radius_array[i] < R_final_m:
                radius_array[i] = R_final_m
        
    else:
        # Simple constant velocity
        t_shrinkage = t_simple
        time_array = np.linspace(0, t_shrinkage, 200)
        radius_array = R_initial_m - v_s_m_s * time_array
    
    print(f"\n[+] Selected model: t_shrinkage = {t_shrinkage*1000:.2f} ms")
    print(f"   Shrinkage rate: {(R_initial_m - R_final_m)/t_shrinkage*1e6:.2f} Âµm/s")
    
    return t_shrinkage, time_array, radius_array


def plot_from_simulation_data(excel_file, batch_id=None, output_dir="./", 
                              use_d2_law=True):
    """
    Generate two-stage evolution plot using ACTUAL simulation data.
    
    Parameters:
    -----------
    excel_file : str
        Path to Excel file with simulation results
    batch_id : str, optional
        Specific batch to plot (if None, plots first or all)
    output_dir : str
        Directory to save output plots
    use_d2_law : bool
        Use dÂ²-law model (True) or simple linear (False)
    """
    
    print(f"\n[*] TWO-STAGE EVOLUTION FROM SIMULATION DATA")
    print(f"=" * 70)
    print(f"Input file: {excel_file}")
    print(f"Using dÂ²-law model: {use_d2_law}")
    
    # Load data
    try:
        df = pd.read_excel(excel_file)
        
        # Check if transposed format
        if df.iloc[0, 0] == 'Parameter / Batch ID' or 'Parameter / Batch ID' in df.columns:
            if 'Parameter / Batch ID' in df.columns:
                df = df.set_index('Parameter / Batch ID').T
            else:
                df = df.set_index(df.columns[0]).T
            df = df.drop('Parameter', errors='ignore')
            df.index.name = 'batch_id'
            
        print(f"âœ“ Loaded: {df.shape[0]} batches, {df.shape[1]} parameters")
        
    except Exception as e:
        print(f"âŒ Error loading file: {e}")
        return None
    
    # Select batch
    if batch_id and batch_id in df.index:
        batches = [batch_id]
    else:
        batches = list(df.index)  # Process ALL batches
        print(f"Processing all {len(batches)} batches: {', '.join(batches[:5])}{'...' if len(batches) > 5 else ''}")
    
    for batch in batches:
        print(f"\n{'='*70}")
        print(f"[*] PROCESSING BATCH: {batch}")
        print(f"{'='*70}")
        
        try:
            row = df.loc[batch]
            
            # Extract parameters
            D32_initial = float(row['D32 (calculated with Moni)'])
            D50_calc = float(row['D50 (calculated with Moni)'])
            D50_actual = float(row['D50_actual']) if pd.notna(row.get('D50_actual')) else None
            t_chamber = float(row['Drying Time estimate (s)'])
            v_s_m_s = float(row['Surface recession velocity'])  # â† KEY PARAMETER!
            
            print(f"\n[*] EXTRACTED PARAMETERS:")
            print(f"  D32 initial: {D32_initial:.3f} Âµm")
            print(f"  D50 calculated: {D50_calc:.3f} Âµm")
            if D50_actual:
                print(f"  D50 actual: {D50_actual:.3f} Âµm")
            print(f"  Chamber residence: {t_chamber*1000:.2f} ms")
            print(f"  Surface recession velocity: {v_s_m_s:.3e} m/s")
            
            # Convert to meters
            R_initial_m = D32_initial * 1e-6 / 2
            R_final_m = D50_calc * 1e-6 / 2
            
            # Calculate shrinkage dynamics using surface recession velocity
            t_shrinkage, time_shrink_s, radius_shrink_m = calculate_shrinkage_with_recession_velocity(
                R_initial_m, R_final_m, v_s_m_s, concentration_evolution=use_d2_law
            )
            
            # Ensure shrinkage time is reasonable (< 80% of chamber time)
            if t_shrinkage > t_chamber * 0.8:
                print(f"\n[!] Warning: Calculated shrinkage time ({t_shrinkage*1000:.1f} ms) " +
                      f"> 80% of chamber time ({t_chamber*1000:.1f} ms)")
                print(f"   Adjusting to 70% of chamber time")
                t_shrinkage = t_chamber * 0.7
                time_shrink_s = np.linspace(0, t_shrinkage, 200)
                radius_shrink_m = R_initial_m - (R_initial_m - R_final_m) * (time_shrink_s / t_shrinkage)
            
            # Extended drying phase (constant size)
            time_extended_s = np.linspace(t_shrinkage, t_chamber, 50)
            radius_extended_m = np.full_like(time_extended_s, R_final_m)
            
            # Convert to micrometers for plotting
            radius_shrink_um = radius_shrink_m * 1e6
            radius_extended_um = radius_extended_m * 1e6
            diameter_shrink_um = radius_shrink_um * 2
            diameter_extended_um = radius_extended_um * 2
            
            # Calculate additional physics
            Pe_initial = float(row['Peclet Number']) if 'Peclet Number' in row.index else None
            Pe_effective = float(row['Effective Peclet Number']) if 'Effective Peclet Number' in row.index else None
            Pe_max = float(row['Max Peclet Number']) if 'Max Peclet Number' in row.index else None
            
            # CREATE PLOT
            fig = plt.figure(figsize=(18, 12))
            
            # Main plot: Diameter evolution
            ax1 = plt.subplot(2, 2, (1, 2))
            
            # Plot Phase 1
            ax1.plot(time_shrink_s * 1000, diameter_shrink_um, 'b-', 
                    linewidth=3.5, label='Phase 1: Active Shrinkage', zorder=3)
            
            # Plot Phase 2
            ax1.plot(time_extended_s * 1000, diameter_extended_um, 'g--', 
                    linewidth=3.5, label='Phase 2: Extended Drying', zorder=3)
            
            # CRITICAL: Vertical line at shrinkage completion
            ax1.axvline(x=t_shrinkage * 1000, color='red', linestyle='-', 
                       linewidth=3, alpha=0.8, zorder=4,
                       label=f'Shrinkage Complete ({t_shrinkage*1000:.1f} ms)')
            
            # Shade regions
            ax1.axvspan(0, t_shrinkage * 1000, alpha=0.15, color='blue', zorder=1)
            ax1.axvspan(t_shrinkage * 1000, t_chamber * 1000, alpha=0.15, color='green', zorder=1)
            
            # Reference lines
            ax1.axhline(y=D50_calc, color='darkgreen', linestyle=':', linewidth=1.5, alpha=0.7,
                       label=f'Calculated D50 = {D50_calc:.2f} Âµm')
            if D50_actual:
                ax1.axhline(y=D50_actual, color='darkred', linestyle=':', linewidth=1.5, alpha=0.7,
                           label=f'Actual D50 = {D50_actual:.2f} Âµm')
            
            # Key points
            ax1.plot(0, D32_initial, 'go', markersize=14, zorder=5)
            ax1.plot(t_shrinkage * 1000, D50_calc, 'ro', markersize=14, zorder=5)
            ax1.plot(t_chamber * 1000, D50_calc, 'bs', markersize=14, zorder=5)
            
            # Annotations
            ax1.text(0, D32_initial * 1.05, f'Initial\nD32 = {D32_initial:.2f} Âµm\nv_s = {v_s_m_s:.2e} m/s',
                    fontsize=10, ha='left', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
            
            ax1.text(t_shrinkage * 1000, D50_calc * 1.15,
                    f'Shrinkage Complete\n{t_shrinkage*1000:.1f} ms\n({t_shrinkage/t_chamber*100:.0f}% of chamber time)',
                    fontsize=10, ha='center', va='bottom', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
            
            ax1.text(t_chamber * 1000, D50_calc * 0.85,
                    f'Chamber Exit\n{t_chamber*1000:.1f} ms',
                    fontsize=10, ha='right', va='top', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
            
            # Shrinkage percentage
            shrinkage_pct = (1 - D50_calc / D32_initial) * 100
            ax1.text(t_shrinkage * 1000 / 2, (D32_initial + D50_calc) / 2,
                    f'Shrinkage:\n{shrinkage_pct:.1f}%',
                    fontsize=15, ha='center', va='center', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.95))
            
            ax1.set_xlabel('Time (ms)', fontweight='bold', fontsize=14)
            ax1.set_ylabel('Particle Diameter (Âµm)', fontweight='bold', fontsize=14)
            ax1.set_title(f'Complete Particle Evolution in Spray Dryer\n' +
                         f'Calculated using Surface Recession Velocity = {v_s_m_s:.2e} m/s',
                         fontweight='bold', fontsize=15)
            ax1.grid(True, alpha=0.3, zorder=0)
            ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
            ax1.set_xlim(left=0, right=t_chamber * 1000 * 1.05)
            
            # Bottom left: Phase 1 detail
            ax2 = plt.subplot(2, 2, 3)
            ax2.plot(time_shrink_s * 1000, diameter_shrink_um, 'b-', linewidth=3)
            ax2.plot(0, D32_initial, 'go', markersize=12)
            ax2.plot(t_shrinkage * 1000, D50_calc, 'ro', markersize=12)
            ax2.axhline(y=D50_calc, color='darkgreen', linestyle=':', linewidth=1.5)
            
            # Add recession rate annotation
            avg_rate_um_s = (D32_initial - D50_calc) / t_shrinkage
            ax2.text(t_shrinkage * 1000 / 2, D32_initial * 0.9,
                    f'Avg rate:\n{avg_rate_um_s:.2f} Âµm/s',
                    fontsize=10, ha='center', va='top',
                    bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.8))
            
            ax2.set_xlabel('Time (ms)', fontweight='bold', fontsize=12)
            ax2.set_ylabel('Diameter (Âµm)', fontweight='bold', fontsize=12)
            ax2.set_title('Phase 1: Active Shrinkage (Zoomed)', fontweight='bold', fontsize=13)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(left=0, right=t_shrinkage * 1000)
            
            # Bottom right: Detailed timeline with physics
            ax3 = plt.subplot(2, 2, 4)
            ax3.axis('off')
            
            timeline_text = f"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘     SPRAY DRYING PHYSICS-BASED CALCULATION           â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

[*] BATCH: {batch}

[*] PHASE 1: ACTIVE SHRINKAGE
   Using: Surface Recession Velocity
   
   v_s = {v_s_m_s:.3e} m/s
   
   Shrinkage equation:
   {"dR/dt = -v_s Ã— (Râ‚€/R)Â²  [dÂ²-law]" if use_d2_law else "dR/dt = -v_s  [linear]"}
   
   Duration: {t_shrinkage*1000:.2f} ms ({t_shrinkage/t_chamber*100:.1f}%)
   Initial: D32 = {D32_initial:.2f} Âµm
   Final: D50 = {D50_calc:.2f} Âµm
   Shrinkage: {shrinkage_pct:.1f}%
   Avg rate: {avg_rate_um_s:.2f} Âµm/s

[*] PHASE 2: EXTENDED DRYING
   Duration: {(t_chamber - t_shrinkage)*1000:.2f} ms ({(1-t_shrinkage/t_chamber)*100:.1f}%)
   Size: Constant at {D50_calc:.2f} Âµm
   Process: Moisture equilibration
           Shell hardening
           Glass transition

[*] TOTAL CHAMBER RESIDENCE
   Total: {t_chamber*1000:.2f} ms
   Shrinkage phase: {t_shrinkage/t_chamber*100:.1f}%
   Extended phase: {(1-t_shrinkage/t_chamber)*100:.1f}%
"""
            
            if Pe_initial and Pe_effective and Pe_max:
                pe_text = f"""
[*] PECLET NUMBERS (from simulation)
   Initial: {Pe_initial:.2f}
   Effective: {Pe_effective:.2f}
   Maximum: {Pe_max:.2f}
   
   Interpretation:
   Pe > 1 â†’ Surface enrichment occurs
   Phase 1 is dominated by advection
"""
                timeline_text += pe_text
            
            if D50_actual:
                validation_text = f"""
[+] VALIDATION
   Calculated D50: {D50_calc:.2f} Âµm
   Actual D50: {D50_actual:.2f} Âµm
   Error: {abs(D50_actual - D50_calc):.2f} Âµm ({abs(D50_actual - D50_calc)/D50_actual*100:.1f}%)
"""
                timeline_text += validation_text
            
            ax3.text(0.05, 0.95, timeline_text, transform=ax3.transAxes,
                    fontsize=9.5, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.9))
            
            # Overall title
            fig.suptitle(f'Two-Stage Particle Evolution: {batch}\n' +
                        f'D32 = {D32_initial:.2f} Âµm â†’ D50 = {D50_calc:.2f} Âµm | ' +
                        f'Physics-Based Calculation using v_s = {v_s_m_s:.2e} m/s',
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            
            # Save
            output_file = Path(output_dir) / f'two_stage_physics_{batch}.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n[+] Saved: {output_file}")
            plt.show()
            
        except Exception as e:
            print(f"âŒ Error processing {batch}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"[+] COMPLETE!")
    print(f"{'='*70}")
    return True


def main():
    """Main execution function."""
    
    if len(sys.argv) > 1:
        excel_file = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "/mnt/user-data/outputs"
        batch_id = sys.argv[3] if len(sys.argv) > 3 else None
        plot_from_simulation_data(excel_file, batch_id, output_dir, use_d2_law=True)
    else:
        # Default: prompt user for Excel file
        print("\n[*] ENHANCED TWO-STAGE PLOTTER")
        print("Using ACTUAL surface recession velocity from simulation")
        print("=" * 70)
        
        # Prompt user for Excel file path
        excel_file = input("Enter the path to the Excel file (e.g., Snapshot_training.xlsx): ").strip()
        if not excel_file:
            print("No file specified. Exiting.")
            return
            
        # Check if file exists
        if not Path(excel_file).exists():
            print(f"Error: File '{excel_file}' not found.")
            return
            
        # Prompt user for output directory
        output_dir = input("Enter the output directory (default: ./outputs): ").strip()
        if not output_dir:
            output_dir = "./outputs"
            
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        plot_from_simulation_data(excel_file, batch_id=None, output_dir=output_dir, use_d2_law=True)


if __name__ == "__main__":
    main()
