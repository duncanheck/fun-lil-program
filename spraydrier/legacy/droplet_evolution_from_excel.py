# droplet_evolution_from_excel.py
# Run this with: python droplet_evolution_from_excel.py DOE_Integral.xlsx

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def calculate_initial_droplet_radius(batch_id, input_df):
    """Calculate initial droplet radius using atomization parameters"""
    try:
        # Find the row for this batch
        batch_col = None
        for col in input_df.columns[1:]:  # Skip first column
            if str(col) == str(batch_id):
                batch_col = col
                break
        
        if batch_col is None:
            return None
            
        # Extract parameters
        params = {}
        param_mappings = {
            'atom_pressure': ['atom_pressure', 'Atom. Pressure (bar)', 'Atomization Gas velocity'],
            'nozzle_diameter': ['nozzle_tip_d_mm', 'Nozzle tip', 'nozzle_diameter'],
            'viscosity': ['viscosity_user_input', 'viscosity', 'Estimated feed viscosity (Pa·s)'],
            'surface_tension': ['surface_tension_user_input', 'surface_tension', 'Estimated Feed Surface Tension (N/m)'],
            'density': ['rho_l', 'density', 'calc_density']
        }
        
        for param_name, possible_names in param_mappings.items():
            for name in possible_names:
                if name in input_df.iloc[:,0].values:
                    row_idx = input_df[input_df.iloc[:,0] == name].index
                    if len(row_idx) > 0:
                        value = input_df.loc[row_idx[0], batch_col]
                        if pd.notna(value):
                            # Try to convert to float, skip non-numeric values like 'y'
                            try:
                                numeric_value = float(value)
                                params[param_name] = numeric_value
                                break
                            except (ValueError, TypeError):
                                continue
        
        # Check if we have minimum required parameters
        required = ['atom_pressure', 'nozzle_diameter', 'viscosity', 'surface_tension']
        if not all(key in params for key in required):
            return None
            
        # Use default density if not available (typical protein solution ~1000 kg/m³)
        if 'density' not in params:
            params['density'] = 1000.0  # kg/m³
            
        # Convert units
        P_atom = params['atom_pressure'] * 1e5  # bar to Pa
        d_nozzle = params['nozzle_diameter'] / 1000  # mm to m
        mu = params['viscosity']  # Pa·s
        sigma = params['surface_tension']  # Already in N/m (mN/m = N/m)
        
        # Handle density units - if rho_l is ~1, it's likely in g/cm³, convert to kg/m³
        rho = params['density']
        if rho < 10:  # Likely in g/cm³ if < 10
            rho = rho * 1000  # Convert g/cm³ to kg/m³
        # kg/m³
        
        # Using a simplified correlation for pressure atomization
        # D32 = k * (σ/ρ)^{0.5} * (μ/ρ)^{0.25} * (1/ΔP)^{0.25} * d_nozzle^{0.5}
        # where k is an empirical constant
        
        k = 150.0  # Empirical constant for pressure atomization (calibrated for realistic sizes)
        D32 = k * (sigma / rho)**0.5 * (mu / rho)**0.25 * (1 / P_atom)**0.25 * d_nozzle**0.5
        
        # Convert from meters to microns
        D32_um = D32 * 1e6
        
        # Sanity check: typical spray drying droplets are 10-100 μm
        if D32_um < 10 or D32_um > 200:
            print(f"Warning: Calculated droplet size {D32_um:.1f} μm seems unrealistic, using fallback")
            return None
        
        return D32_um / 2  # Return radius, not diameter
        
    except Exception as e:
        print(f"Error calculating droplet size for {batch_id}: {e}")
        return None

def generate_droplet_evolution_graph(excel_file, output_file=None):
    # Read the Excel file
    df = pd.read_excel(excel_file, header=0)
    
    # Check file type
    has_final_radius = any('final_radius_um' in str(col).lower() for col in df.iloc[:,0].values)
    has_inlet_temp = any('Drying Gas Inlet (C)' in str(col) for col in df.iloc[:,0].values)
    has_d50 = any('D50_actual' in str(col) for col in df.iloc[:,0].values)
    
    if has_final_radius and not has_inlet_temp:
        # This is an output file - we need inlet temps from corresponding input file
        print("Output file detected. Looking for corresponding input file...")
        return generate_from_output_file(df, excel_file, output_file)
    
    elif has_inlet_temp and has_d50:
        # This is a data file with inlet temps and D50 - can generate graph directly
        print("Data file detected with inlet temperatures and D50 data.")
        return generate_from_data_file(df, excel_file, output_file)
    
    else:
        print(f"File type not recognized. File has:")
        print(f"  - Inlet temperatures: {has_inlet_temp}")
        print(f"  - Final radius: {has_final_radius}")
        print(f"  - D50 data: {has_d50}")
        return

def generate_from_output_file(df, excel_file, output_file=None):
    """Generate graph from output file (needs to look up input data)"""
    # Extract batch IDs from output file first
    batch_ids = df.columns[1:]  # Skip first column
    
    # Determine which input file to use based on the output file name
    if 'doe' in excel_file.lower():
        input_filename = 'DOE_Integral.xlsx'
    else:
        input_filename = 'data.xlsx'
    
    print(f"Using input file: {input_filename}")
    
    # Try to find inlet temperatures and load input data
    try:
        input_df = pd.read_excel(input_filename, header=0)
        inlet_temp_row = input_df[input_df.iloc[:,0] == 'Drying Gas Inlet (C)']
        if len(inlet_temp_row) > 0:
            input_batch_ids = input_df.columns[1:]  # Skip first column
            inlet_temp_values = inlet_temp_row.iloc[0,1:].values
            
            # Match batch IDs between input and output files
            inlet_temps = []
            for batch_id in batch_ids:
                if batch_id in input_batch_ids:
                    idx = list(input_batch_ids).index(batch_id)
                    inlet_temps.append(inlet_temp_values[idx])
                else:
                    inlet_temps.append(100.0)  # Default if not found
            print(f"Matched inlet temperatures for {len(inlet_temps)} batches from {input_filename}")
        else:
            raise Exception("No inlet temp data found")
    except Exception as e:
        print(f"Could not load inlet temperatures: {e}. Using default 100°C for all batches.")
        inlet_temps = [100.0] * len(batch_ids)
    
    # Extract final radii from D50_actual in input file, with fallback to final_radius_um
    if input_df is not None:
        d50_row = input_df[input_df.iloc[:,0] == 'D50_actual']
        final_radius_row = df[df.iloc[:,0].str.contains('final_radius_um', na=False, case=False)]
        
        if len(d50_row) > 0 and len(final_radius_row) > 0:
            d50_values = d50_row.iloc[0,1:].values
            fallback_radii = final_radius_row.iloc[0,1:].values
            
            # Use D50 where available, otherwise use final_radius_um
            final_radii = []
            for d50, fallback in zip(d50_values, fallback_radii):
                if pd.notna(d50) and d50 > 0:
                    final_radii.append(d50 / 2)  # Convert diameter to radius
                elif pd.notna(fallback) and fallback > 0:
                    final_radii.append(fallback)  # Use final_radius_um as radius
                else:
                    final_radii.append(np.nan)  # Keep as NaN for filtering
            
            final_radii = np.array(final_radii)
            valid_count = np.sum(np.isfinite(final_radii))
            print(f"Using D50_actual data with final_radius_um fallback ({valid_count} valid values)")
        else:
            print("No particle size data found")
            return
    else:
        final_radii = df[df.iloc[:,0].str.contains('final_radius_um', na=False, case=False)].iloc[0,1:].values
    
    # Filter out batches with invalid data
    valid_indices = []
    for i, (batch, inlet_temp, final_r) in enumerate(zip(batch_ids, inlet_temps, final_radii)):
        if pd.notna(final_r) and pd.notna(inlet_temp) and final_r > 0:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        print("No valid data found for plotting.")
        return
    
    # Keep only valid batches
    batch_ids = [batch_ids[i] for i in valid_indices]
    inlet_temps = [inlet_temps[i] for i in valid_indices] 
    final_radii = [final_radii[i] for i in valid_indices]
    
    # Use the input_df for atomization calculations
    plot_evolution_graph(batch_ids, inlet_temps, final_radii, input_df, output_file)

def generate_from_data_file(df, excel_file, output_file=None):
    """Generate graph directly from data file (has all needed data)"""
    # Extract inlet temperatures
    inlet_temps = df[df.iloc[:,0] == 'Drying Gas Inlet (C)'].iloc[0,1:].values
    
    # Extract final radii from D50 data (try multiple column names)
    d50_values = [np.nan] * (len(df.columns) - 1)  # Initialize with NaN
    
    # First try D50_actual
    d50_actual_row = df[df.iloc[:,0] == 'D50_actual']
    if len(d50_actual_row) > 0:
        actual_values = d50_actual_row.iloc[0,1:].values
        for i, val in enumerate(actual_values):
            if pd.notna(val) and val > 0:
                d50_values[i] = val
    
    # Then try alternative names to fill in missing values
    alt_names = ['D50 (calculated with Moni)', 'D50_calculated', 'D50']
    for alt_name in alt_names:
        alt_row = df[df.iloc[:,0] == alt_name]
        if len(alt_row) > 0:
            alt_values = alt_row.iloc[0,1:].values
            for i, val in enumerate(alt_values):
                if pd.notna(val) and val > 0 and pd.isna(d50_values[i]):
                    d50_values[i] = val
                    print(f"Using {alt_name} for batch {df.columns[i+1]}")
            break
    
    # Convert to radii
    final_radii = []
    for d50 in d50_values:
        if pd.notna(d50) and d50 > 0:
            final_radii.append(d50 / 2)  # Convert diameter to radius
        else:
            final_radii.append(np.nan)
    
    final_radii = np.array(final_radii)
    valid_count = np.sum(np.isfinite(final_radii))
    if valid_count > 0:
        print(f"Using D50 data ({valid_count} valid values)")
    else:
        print("No D50 data found in file")
        return
    
    # Extract batch IDs
    batch_ids = df.columns[1:]  # Skip first column
    
    # Filter out batches with invalid data
    valid_indices = []
    for i, (batch, inlet_temp, final_r) in enumerate(zip(batch_ids, inlet_temps, final_radii)):
        if pd.notna(final_r) and pd.notna(inlet_temp) and final_r > 0:
            valid_indices.append(i)
    
    if len(valid_indices) == 0:
        print("No valid data found for plotting.")
        return
    
    # Keep only valid batches
    batch_ids = [batch_ids[i] for i in valid_indices]
    inlet_temps = [inlet_temps[i] for i in valid_indices] 
    final_radii = [final_radii[i] for i in valid_indices]
    
    # Use the same data file for atomization calculations
    plot_evolution_graph(batch_ids, inlet_temps, final_radii, df, output_file)

def plot_evolution_graph(batch_ids, inlet_temps, final_radii, input_df, output_file=None):
    """Common plotting function used by both file type handlers"""
    # Time vector: 0 to 10 seconds in 0.02s steps (501 points) - higher resolution for smooth curves
    time_s = np.linspace(0, 10.0, 501)
    
    # Colors matching your approved graph
    colors = ['blue', 'orange', 'red', 'green', 'purple', 'brown', 'pink', 'gray']
    
    plt.figure(figsize=(11, 7))
    
    # Track completion times for each batch
    completion_times = []
    initial_radii = []  # Collect initial radii for y-axis scaling
    data_collection = []  # Collect radius data for Excel output
    
    for i, (batch, inlet_temp, final_r) in enumerate(zip(batch_ids, inlet_temps, final_radii)):
        color = colors[i % len(colors)]
        
        if "bsa" in str(batch).lower():
            label = f"{batch} (BSA ref)"
        else:
            label = f"{batch} (Inlet {inlet_temp:.0f}°C)"
        
        # Calculate initial droplet radius using atomization parameters
        r0 = None
        surface_recession = None
        if input_df is not None:
            r0 = calculate_initial_droplet_radius(batch, input_df)
            
            # Extract surface recession velocity for physics-based evolution
            recession_row = input_df[input_df.iloc[:,0] == 'Surface recession velocity']
            if len(recession_row) > 0:
                try:
                    batch_idx = list(input_df.columns[1:]).index(batch) + 1  # +1 because columns[0] is the parameter name
                    recession_value = recession_row.iloc[0, batch_idx]
                    if pd.notna(recession_value) and recession_value > 0:
                        # Detect units: if value < 1, assume m/s; if > 1, assume μm/s and convert
                        if recession_value < 1:
                            surface_recession = recession_value  # Already in m/s
                        else:
                            surface_recession = recession_value * 1e-6  # Convert from μm/s to m/s
                except (ValueError, IndexError):
                    pass  # Batch not found, will use fallback
        
        if r0 is None or r0 <= final_r:
            # Fallback: estimate based on final particle size
            r0 = final_r * 25  # Start with droplet 25x larger than final particle
            print(f"Using estimated initial radius for {batch}: {r0:.1f} μm (final_r={final_r:.2f})")
        else:
            print(f"Calculated initial radius for {batch}: {r0:.1f} μm")
        
        initial_radii.append(r0)  # Collect for y-axis scaling
        
        # Use physics-based evolution if surface recession data is available
        if surface_recession is not None and False:  # Temporarily disable physics-based, use d²-law
            print(f"Using physics-based evolution for {batch} (surface recession: {surface_recession:.2e} m/s)")
            
            # Physics-based evolution using surface recession velocity
            radius = []
            current_r = r0
            batch_completion_time = None
            for t in time_s:
                if current_r <= final_r:
                    # Reached final size
                    radius.append(final_r)
                    if batch_completion_time is None:
                        batch_completion_time = t
                else:
                    # Calculate how much radius has been lost
                    # Using simplified constant surface recession (valid for early drying)
                    radius_lost = surface_recession * t * 1e6  # Convert to μm
                    current_r_calc = max(r0 - radius_lost, final_r)
                    radius.append(current_r_calc)
                    current_r = current_r_calc
        else:
            # Fallback to simplified d²-law
            print(f"Using simplified d²-law evolution for {batch}")
            
            # Calculate shrinkage rate to reach final radius in ~1-2 seconds
            # Using d²-law: r² = r0² - k*t
            # At t_final: final_r² = r0² - k*t_final
            # So: k = (r0² - final_r²) / t_final
            t_final = 1.5  # seconds to reach final size (adjusted for realistic spray drying times)
            k = (r0**2 - final_r**2) / t_final
            
            # Adjust k based on inlet temperature (higher T = faster drying)
            temp_factor = 1.0 + (inlet_temp - 40) * 0.05  # 5% increase per °C above 40°C (more aggressive)
            k = k * temp_factor
            
            # Generate radius over time
            radius = []
            batch_completion_time = None
            for t in time_s:
                if t >= t_final:
                    # Reached final size
                    r = final_r
                    if batch_completion_time is None:
                        batch_completion_time = t_final
                else:
                    # d²-law shrinkage
                    r_squared = r0**2 - k * t
                    if r_squared <= final_r**2:
                        r = final_r
                        if batch_completion_time is None:
                            # Solve for t when r_squared = final_r**2
                            batch_completion_time = (r0**2 - final_r**2) / k
                    else:
                        r = np.sqrt(r_squared)
                
                radius.append(r)
        
        # Store completion time for this batch
        if batch_completion_time is not None:
            completion_times.append(batch_completion_time)
        
        plt.plot(time_s, radius, color=color, linewidth=3, label=label)
        data_collection.append(radius)  # Collect for Excel output
    
    # Set x-axis limit based on completion times
    if completion_times:
        max_completion_time = max(completion_times)
        x_limit = max_completion_time + 0.1  # 0.1 seconds beyond time to reach D50 radius
        print(f"Setting x-axis limit to {x_limit:.2f} seconds (max completion time: {max_completion_time:.1f}s)")
    else:
        x_limit = 10.0  # Fallback if no completion times found
    
    plt.xlim(0, x_limit)
    plt.xticks(np.arange(0, x_limit + 0.5, 0.5))  # Tick marks every 0.5 seconds
    plt.ylim(0, max(initial_radii) * 1.1)  # Scale y-axis to max initial radius + 10%
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Particle radius (µm)', fontsize=12)
    plt.title('Droplet → Particle evolution for your data', fontsize=14, pad=20)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('droplet_evolution_graph.png', dpi=300, bbox_inches='tight')
    print("Graph saved as 'droplet_evolution_graph.png'")
    plt.show()
    
    # Save data to Excel if output file specified
    if output_file:
        df_out = pd.DataFrame()
        df_out['Time (s)'] = time_s
        for batch, rad in zip(batch_ids, data_collection):
            df_out[f'{batch} radius (μm)'] = rad
        df_out.to_excel(output_file, index=False)
        print(f"Evolution data saved to '{output_file}'")

# === RUN IT ===
if __name__ == "__main__":
    if len(sys.argv) == 2:
        # Command line usage: input file only
        excel_file = sys.argv[1]
        output_file = None
    elif len(sys.argv) == 3:
        # Command line usage: input and output files
        excel_file = sys.argv[1]
        output_file = sys.argv[2]
    else:
        # Interactive mode
        excel_file = input("Enter the Excel file name (including .xlsx extension): ").strip()
        if not excel_file:
            print("No file specified. Exiting.")
            sys.exit(1)
        output_file = input("Enter output Excel file name (or press enter to skip): ").strip()
        if not output_file:
            output_file = None
    
    generate_droplet_evolution_graph(excel_file, output_file)