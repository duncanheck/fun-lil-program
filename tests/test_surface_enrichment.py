#!/usr/bin/env python3
"""
Test script to demonstrate surface enrichment with different diffusion coefficients
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
D_drug = 5e-10  # m²/s - higher diffusion coefficient
D_moni = 2e-10  # m²/s - lower diffusion coefficient (should enrich more)
D_stabilizer = 1e-9  # m²/s - highest diffusion coefficient
D_additive = 1e-10   # m²/s - lowest diffusion coefficient (should enrich most)

# Initial conditions for slow drying
initial_radius = 50e-6  # 50 μm droplet
solids_fraction = 0.001  # 0.1% solids - very dilute
xi_critical = 0.95  # High threshold to allow more drying

# Initial solute concentrations (mass fractions)
xi_drug_0 = 0.40
xi_moni_0 = 0.40
xi_stabilizer_0 = 0.15
xi_additive_0 = 0.05

# Drying conditions
surface_recession = 1e-6  # m/s - 1 μm/s - much faster drying for demonstration
dt = 0.001  # s
t_max = 1.0  # s

# Simulation arrays
t = np.arange(0, t_max, dt)
n_steps = len(t)

xi_drug = np.zeros(n_steps)
xi_moni = np.zeros(n_steps)
xi_stabilizer = np.zeros(n_steps)
xi_additive = np.zeros(n_steps)

xi_drug[0] = xi_drug_0
xi_moni[0] = xi_moni_0
xi_stabilizer[0] = xi_stabilizer_0
xi_additive[0] = xi_additive_0

# Surface concentrations
xi_drug_surface = np.zeros(n_steps)
xi_moni_surface = np.zeros(n_steps)
xi_stabilizer_surface = np.zeros(n_steps)
xi_additive_surface = np.zeros(n_steps)

xi_drug_surface[0] = xi_drug_0
xi_moni_surface[0] = xi_moni_0
xi_stabilizer_surface[0] = xi_stabilizer_0
xi_additive_surface[0] = xi_additive_0

# Track radius (simplified - constant recession rate)
radius = initial_radius - surface_recession * t

print("Demonstrating surface enrichment with different diffusion coefficients")
print(f"Initial radius: {initial_radius*1e6:.1f} μm")
print(f"Solids fraction: {solids_fraction*100:.1f}%")
print(f"Surface recession rate: {surface_recession*1e9:.1f} nm/s")
print(f"Diffusion coefficients: Drug={D_drug*1e10:.1f}e-10, Moni={D_moni*1e10:.1f}e-10, Stabilizer={D_stabilizer*1e10:.1f}e-10, Additive={D_additive*1e10:.1f}e-10 m²/s")
print()

crust_formed = False
crust_time = None

for i in range(1, n_steps):
    # Calculate Peclet numbers
    Pe_drug = surface_recession * radius[i-1] / D_drug
    Pe_moni = surface_recession * radius[i-1] / D_moni
    Pe_stabilizer = surface_recession * radius[i-1] / D_stabilizer
    Pe_additive = surface_recession * radius[i-1] / D_additive

    # Cap Peclet numbers
    Pe_drug = min(Pe_drug, 10)
    Pe_moni = min(Pe_moni, 10)
    Pe_stabilizer = min(Pe_stabilizer, 10)
    Pe_additive = min(Pe_additive, 10)

    # Update bulk concentrations (simplified - just track)
    # In reality, this would be more complex with mass balance
    xi_drug[i] = xi_drug[i-1]
    xi_moni[i] = xi_moni[i-1]
    xi_stabilizer[i] = xi_stabilizer[i-1]
    xi_additive[i] = xi_additive[i-1]

    # Update surface concentrations
    xi_drug_surface[i] = xi_drug[i-1] * np.exp(Pe_drug)
    xi_moni_surface[i] = xi_moni[i-1] * np.exp(Pe_moni)
    xi_stabilizer_surface[i] = xi_stabilizer[i-1] * np.exp(Pe_stabilizer)
    xi_additive_surface[i] = xi_additive[i-1] * np.exp(Pe_additive)

    # Check for crust formation (simplified - based on surface concentration)
    total_surface = xi_drug_surface[i] + xi_moni_surface[i] + xi_stabilizer_surface[i] + xi_additive_surface[i]

    if total_surface >= xi_critical and not crust_formed:
        crust_formed = True
        crust_time = t[i]
        print(f"Crust formed at t = {crust_time:.3f}s")
        print(f"Surface concentrations at crust formation:")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".3f")
        print(".1f")

        # Calculate enrichment factors (surface/bulk)
        enrichment_drug = xi_drug_surface[i] / xi_drug[i]
        enrichment_moni = xi_moni_surface[i] / xi_moni[i]
        enrichment_stabilizer = xi_stabilizer_surface[i] / xi_stabilizer[i]
        enrichment_additive = xi_additive_surface[i] / xi_additive[i]

        print(f"\nEnrichment factors (surface/bulk):")
        print(f"Drug: {enrichment_drug:.2f}x")
        print(f"Moni: {enrichment_moni:.2f}x")
        print(f"Stabilizer: {enrichment_stabilizer:.2f}x")
        print(f"Additive: {enrichment_additive:.2f}x")

        # Normalize to percentages
        total = xi_drug_surface[i] + xi_moni_surface[i] + xi_stabilizer_surface[i] + xi_additive_surface[i]
        pct_drug = xi_drug_surface[i] / total * 100
        pct_moni = xi_moni_surface[i] / total * 100
        pct_stabilizer = xi_stabilizer_surface[i] / total * 100
        pct_additive = xi_additive_surface[i] / total * 100

        print(f"\nSurface composition (%):")
        print(f"Drug: {pct_drug:.1f}%")
        print(f"Moni: {pct_moni:.1f}%")
        print(f"Stabilizer: {pct_stabilizer:.1f}%")
        print(f"Additive: {pct_additive:.1f}%")
        break

if not crust_formed:
    print("No crust formed within simulation time")
    print("Final surface concentrations:")
    i = -1
    print(f"Drug: {xi_drug_surface[i]:.3f}")
    print(f"Moni: {xi_moni_surface[i]:.3f}")
    print(f"Stabilizer: {xi_stabilizer_surface[i]:.3f}")
    print(f"Additive: {xi_additive_surface[i]:.3f}")

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(t[:i+1], xi_drug_surface[:i+1], 'b-', label='Drug surface')
plt.plot(t[:i+1], xi_moni_surface[:i+1], 'r-', label='Moni surface')
plt.plot(t[:i+1], xi_stabilizer_surface[:i+1], 'g-', label='Stabilizer surface')
plt.plot(t[:i+1], xi_additive_surface[:i+1], 'm-', label='Additive surface')
plt.xlabel('Time (s)')
plt.ylabel('Surface concentration')
plt.title('Surface Concentrations vs Time')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t[:i+1], radius[:i+1]*1e6, 'k-')
plt.xlabel('Time (s)')
plt.ylabel('Radius (μm)')
plt.title('Droplet Radius vs Time')
plt.grid(True)

plt.subplot(2, 2, 3)
Pe_drug_plot = surface_recession * radius[:i+1] / D_drug
Pe_moni_plot = surface_recession * radius[:i+1] / D_moni
Pe_stabilizer_plot = surface_recession * radius[:i+1] / D_stabilizer
Pe_additive_plot = surface_recession * radius[:i+1] / D_additive

plt.plot(t[:i+1], np.minimum(Pe_drug_plot, 10), 'b-', label='Drug Pe')
plt.plot(t[:i+1], np.minimum(Pe_moni_plot, 10), 'r-', label='Moni Pe')
plt.plot(t[:i+1], np.minimum(Pe_stabilizer_plot, 10), 'g-', label='Stabilizer Pe')
plt.plot(t[:i+1], np.minimum(Pe_additive_plot, 10), 'm-', label='Additive Pe')
plt.xlabel('Time (s)')
plt.ylabel('Peclet Number')
plt.title('Peclet Numbers vs Time')
plt.legend()
plt.grid(True)

plt.subplot(2, 2, 4)
if crust_formed:
    enrichment_drug = xi_drug_surface[i] / xi_drug[i]
    enrichment_moni = xi_moni_surface[i] / xi_moni[i]
    enrichment_stabilizer = xi_stabilizer_surface[i] / xi_stabilizer[i]
    enrichment_additive = xi_additive_surface[i] / xi_additive[i]

    solutes = ['Drug', 'Moni', 'Stabilizer', 'Additive']
    enrichments = [enrichment_drug, enrichment_moni, enrichment_stabilizer, enrichment_additive]
    colors = ['blue', 'red', 'green', 'magenta']

    plt.bar(solutes, enrichments, color=colors)
    plt.ylabel('Enrichment Factor')
    plt.title('Surface Enrichment at Crust Formation')
    plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig('surface_enrichment_demo.png', dpi=150, bbox_inches='tight')
print("\nPlot saved as 'surface_enrichment_demo.png'")