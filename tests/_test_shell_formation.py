#!/usr/bin/env python3

import sys
import os
sys.path.append('/Users/doughecker/Desktop/Python')

from spraydrier.legacy.enhanced_physics_evolution_plotter_FINAL import (
    calculate_enhanced_shrinkage_with_glass_transition
)


# Test batch data with composition
test_batch_data = {
    'batch_id': 'TEST_001',
    'ds': 'BSA',
    'ds_conc': 100.0,
    'moni_conc': 5.0,
    'buffer': 'phosphate',
    'buffer_conc': 10.0,
    'pH': 7.0,
    'solids_frac': 0.1,
    'T_outlet_C': 65.0,
    'feed_g_min': 2.0,
    'T_inlet_C': 78.0,
    'RH1': 55.0,
    'D50_actual': 1.5
}

print("Testing enhanced shrinkage calculation with glass transition...")
print(f"Batch: {test_batch_data['batch_id']}")
print(f"Composition: {test_batch_data['ds']} {test_batch_data['ds_conc']} mg/mL, Moni {test_batch_data['moni_conc']} mg/mL")

try:
    results = calculate_enhanced_shrinkage_with_glass_transition(
        R_initial_m=1.5e-6,  # 1.5 μm radius
        v_s_initial=2e-6,    # Initial evaporation rate
        t_chamber=0.25,      # 250 ms
        batch_data=test_batch_data,
        T_inlet=78.0,
        D50_measured=1.5
    )

    print("\nResults:")
    for key, value in results.items():
        if 'shell' in key.lower():
            print(f"  {key}: {value}")
        elif 'formation' in key.lower():
            print(f"  {key}: {value}")

    shell_time = results.get('shell_formation_time')
    if shell_time is not None and shell_time > 0:
        print(f"\n✅ SUCCESS: Shell formation detected at {shell_time*1000:.1f} ms!")
    else:
        print(f"\n❌ FAILED: No shell formation detected (shell_formation_time = {shell_time})")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()