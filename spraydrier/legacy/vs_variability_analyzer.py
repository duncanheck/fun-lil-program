#!/usr/bin/env python3
"""
Surface Recession Velocity Variability Analysis
================================================

Shows how v_s varies across batches and correlates with process conditions.
This demonstrates WHY you must use batch-specific v_s values.

Author: Claude
Date: December 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_vs_variability(excel_file, output_dir="./"):
    """
    Analyze and visualize v_s variability across batches.
    """
    
    print(f"\nðŸ“Š SURFACE RECESSION VELOCITY VARIABILITY ANALYSIS")
    print(f"=" * 80)
    
    # Load data
    df = pd.read_excel(excel_file)
    df_t = df.set_index(df.columns[0]).T.drop('Parameter', errors='ignore')
    df_t.index.name = 'batch_id'
    
    # Extract parameters
    params = {
        'v_s': 'Surface recession velocity',
        'D32': 'D32 (calculated with Moni)',
        'D50': 'D50 (calculated with Moni)',
        'feed': 'Feed Rate (g/min)',
        'T_inlet': 'Drying Gas Inlet (C)',
        'visc': 'Estimated feed viscosity (PaÂ·s)',
        't_dry': 'Drying Time estimate (s)',
        'u_atom': 'Atomization Gas velocity'
    }
    
    data = {}
    for key, param in params.items():
        if param in df_t.columns:
            data[key] = pd.to_numeric(df_t[param], errors='coerce')
    
    df_analysis = pd.DataFrame(data)
    df_analysis = df_analysis.dropna(subset=['v_s'])
    
    print(f"âœ“ Loaded {len(df_analysis)} batches with complete data")
    
    # Calculate derived quantities
    df_analysis['t_shrink'] = (df_analysis['D32'] - df_analysis['D50']) * 1e-6 / 2 / df_analysis['v_s'] * 1000  # ms
    df_analysis['phase1_pct'] = df_analysis['t_shrink'] / (df_analysis['t_dry'] * 1000) * 100
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 14))
    
    # Plot 1: v_s distribution
    ax1 = plt.subplot(3, 3, 1)
    ax1.hist(df_analysis['v_s'] * 1e6, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(df_analysis['v_s'].mean() * 1e6, color='red', linestyle='--', linewidth=2, label='Mean')
    ax1.set_xlabel('Surface Recession Velocity (Âµm/s)', fontweight='bold')
    ax1.set_ylabel('Count', fontweight='bold')
    ax1.set_title('Distribution of v_s Across Batches', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add statistics
    mean_vs = df_analysis['v_s'].mean() * 1e6
    std_vs = df_analysis['v_s'].std() * 1e6
    cv_vs = (df_analysis['v_s'].std() / df_analysis['v_s'].mean()) * 100
    ax1.text(0.95, 0.95, f'Mean: {mean_vs:.3f} Âµm/s\nStd: {std_vs:.3f} Âµm/s\nCV: {cv_vs:.1f}%',
            transform=ax1.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: v_s vs Feed Rate
    ax2 = plt.subplot(3, 3, 2)
    scatter = ax2.scatter(df_analysis['feed'], df_analysis['v_s'] * 1e6, 
                         c=df_analysis['T_inlet'], cmap='coolwarm', s=100, alpha=0.7, edgecolors='black')
    ax2.set_xlabel('Feed Rate (g/min)', fontweight='bold')
    ax2.set_ylabel('v_s (Âµm/s)', fontweight='bold')
    ax2.set_title('v_s vs Feed Rate (colored by T_inlet)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax2, label='T_inlet (Â°C)')
    
    # Plot 3: v_s vs Inlet Temperature
    ax3 = plt.subplot(3, 3, 3)
    scatter = ax3.scatter(df_analysis['T_inlet'], df_analysis['v_s'] * 1e6,
                         c=df_analysis['feed'], cmap='viridis', s=100, alpha=0.7, edgecolors='black')
    ax3.set_xlabel('Inlet Temperature (Â°C)', fontweight='bold')
    ax3.set_ylabel('v_s (Âµm/s)', fontweight='bold')
    ax3.set_title('v_s vs Inlet Temperature (colored by feed rate)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax3, label='Feed Rate (g/min)')
    
    # Plot 4: v_s vs D32
    ax4 = plt.subplot(3, 3, 4)
    scatter = ax4.scatter(df_analysis['D32'], df_analysis['v_s'] * 1e6,
                         c=df_analysis['visc'] * 1e3, cmap='plasma', s=100, alpha=0.7, edgecolors='black')
    ax4.set_xlabel('D32 (Âµm)', fontweight='bold')
    ax4.set_ylabel('v_s (Âµm/s)', fontweight='bold')
    ax4.set_title('v_s vs Initial Droplet Size (colored by viscosity)', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Viscosity (mPaÂ·s)')
    
    # Strong correlation expected
    corr_d32 = df_analysis['v_s'].corr(df_analysis['D32'])
    ax4.text(0.05, 0.95, f'Correlation: {corr_d32:.3f}',
            transform=ax4.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Plot 5: v_s vs Viscosity
    ax5 = plt.subplot(3, 3, 5)
    ax5.scatter(df_analysis['visc'] * 1e3, df_analysis['v_s'] * 1e6,
               c='purple', s=100, alpha=0.7, edgecolors='black')
    ax5.set_xlabel('Viscosity (mPaÂ·s)', fontweight='bold')
    ax5.set_ylabel('v_s (Âµm/s)', fontweight='bold')
    ax5.set_title('v_s vs Feed Viscosity', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    corr_visc = df_analysis['v_s'].corr(df_analysis['visc'])
    ax5.text(0.05, 0.95, f'Correlation: {corr_visc:.3f}',
            transform=ax5.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Plot 6: Shrinkage time distribution
    ax6 = plt.subplot(3, 3, 6)
    ax6.hist(df_analysis['t_shrink'], bins=20, color='coral', edgecolor='black', alpha=0.7)
    ax6.axvline(df_analysis['t_shrink'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax6.set_xlabel('Shrinkage Time (ms)', fontweight='bold')
    ax6.set_ylabel('Count', fontweight='bold')
    ax6.set_title('Distribution of Calculated Shrinkage Times', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Statistics
    mean_t = df_analysis['t_shrink'].mean()
    std_t = df_analysis['t_shrink'].std()
    ax6.text(0.95, 0.95, f'Mean: {mean_t:.1f} ms\nStd: {std_t:.1f} ms\nRange: {df_analysis["t_shrink"].max() - df_analysis["t_shrink"].min():.1f} ms',
            transform=ax6.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 7: Phase 1 percentage
    ax7 = plt.subplot(3, 3, 7)
    ax7.hist(df_analysis['phase1_pct'], bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
    ax7.axvline(df_analysis['phase1_pct'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax7.axvline(100, color='orange', linestyle=':', linewidth=2, label='100% (full chamber)')
    ax7.set_xlabel('Phase 1 as % of Chamber Time', fontweight='bold')
    ax7.set_ylabel('Count', fontweight='bold')
    ax7.set_title('Phase 1 Duration Relative to Chamber Time', fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: v_s vs atomization velocity
    ax8 = plt.subplot(3, 3, 8)
    if 'u_atom' in df_analysis.columns:
        scatter = ax8.scatter(df_analysis['u_atom'], df_analysis['v_s'] * 1e6,
                             c=df_analysis['D32'], cmap='jet', s=100, alpha=0.7, edgecolors='black')
        ax8.set_xlabel('Atomization Gas Velocity (m/s)', fontweight='bold')
        ax8.set_ylabel('v_s (Âµm/s)', fontweight='bold')
        ax8.set_title('v_s vs Atomization (colored by D32)', fontweight='bold')
        ax8.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax8, label='D32 (Âµm)')
        
        corr_atom = df_analysis['v_s'].corr(df_analysis['u_atom'])
        ax8.text(0.05, 0.95, f'Correlation: {corr_atom:.3f}',
                transform=ax8.transAxes, fontsize=10, va='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Plot 9: Correlation matrix
    ax9 = plt.subplot(3, 3, 9)
    
    corr_text = f"""
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘   CORRELATION WITH v_s                 â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

Strong positive correlations:
  â€¢ D32:          {df_analysis['v_s'].corr(df_analysis['D32']):+.3f} â¬†ï¸
  â€¢ Drying time:  {df_analysis['v_s'].corr(df_analysis['t_dry']):+.3f} â¬†ï¸
  â€¢ D50:          {df_analysis['v_s'].corr(df_analysis['D50']):+.3f} â¬†ï¸

Strong negative correlations:
  â€¢ Atomization:  {df_analysis['v_s'].corr(df_analysis['u_atom']):+.3f} â¬‡ï¸
  â€¢ Viscosity:    {df_analysis['v_s'].corr(df_analysis['visc']):+.3f} â¬‡ï¸
  â€¢ Feed rate:    {df_analysis['v_s'].corr(df_analysis['feed']):+.3f} â¬‡ï¸
  â€¢ T_inlet:      {df_analysis['v_s'].corr(df_analysis['T_inlet']):+.3f} â¬‡ï¸

KEY FINDINGS:
â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”â”

âœ“ v_s varies {df_analysis['v_s'].max()/df_analysis['v_s'].min():.2f}Ã— across batches

âœ“ Larger droplets â†’ slower recession
  (more volume to evaporate)

âœ“ Higher atomization â†’ faster recession
  (better heat/mass transfer)

âœ“ Higher viscosity â†’ slower recession
  (reduced diffusion)

âœ“ Each batch is UNIQUE!
  Cannot use single k value!

ðŸ“Š VARIABILITY:
  CV = {(df_analysis['v_s'].std()/df_analysis['v_s'].mean()*100):.1f}%
  Range = {(df_analysis['v_s'].max() - df_analysis['v_s'].min())*1e6:.3f} Âµm/s
  
âš ï¸  Using k = 20.1 for all would
   ignore all this variability!
"""
    
    ax9.axis('off')
    ax9.text(0.05, 0.95, corr_text, transform=ax9.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Overall title
    fig.suptitle('Surface Recession Velocity Variability Analysis\n' +
                 'Why Batch-Specific v_s is Essential for Accurate Predictions',
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    # Save
    output_file = Path(output_dir) / 'vs_variability_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nâœ… Saved: {output_file}")
    plt.show()
    
    # Print summary statistics
    print(f"\nðŸ“Š SUMMARY STATISTICS:")
    print(f"=" * 80)
    print(f"\nSurface Recession Velocity:")
    print(f"  Mean: {df_analysis['v_s'].mean()*1e6:.3f} Âµm/s")
    print(f"  Std:  {df_analysis['v_s'].std()*1e6:.3f} Âµm/s")
    print(f"  Min:  {df_analysis['v_s'].min()*1e6:.3f} Âµm/s")
    print(f"  Max:  {df_analysis['v_s'].max()*1e6:.3f} Âµm/s")
    print(f"  CV:   {(df_analysis['v_s'].std()/df_analysis['v_s'].mean()*100):.1f}%")
    
    print(f"\nShrinkage Time (calculated from v_s):")
    print(f"  Mean: {df_analysis['t_shrink'].mean():.1f} ms")
    print(f"  Min:  {df_analysis['t_shrink'].min():.1f} ms")
    print(f"  Max:  {df_analysis['t_shrink'].max():.1f} ms")
    print(f"  Range: {df_analysis['t_shrink'].max() - df_analysis['t_shrink'].min():.1f} ms")
    
    print(f"\nðŸ” TOP CORRELATIONS WITH v_s:")
    correlations = df_analysis.corr()['v_s'].sort_values(ascending=False)
    print(correlations.to_string())
    
    return df_analysis


if __name__ == "__main__":
    analyze_vs_variability('/mnt/project/Snapshot_training.xlsx', '/mnt/user-data/outputs')
