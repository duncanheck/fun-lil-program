import numpy as np
import plotly.graph_objects as go
import math

def get_user_input():
    print("Enter details for up to 5 compounds.")
    num_compounds = int(input("Number of compounds (1-5): "))
    if not 1 <= num_compounds <= 5:
        raise ValueError("Number of compounds must be between 1 and 5.")
    
    compounds = []
    for i in range(num_compounds):
        print(f"\nCompound {i+1}:")
        name = input("Name: ")
        conc = float(input("Initial concentration (mg/mL): "))
        solubility = float(input("Solubility at drying temperature (mg/mL): "))
        diffusion = float(input("Diffusion coefficient (cm²/s, e.g., 1e-5): "))
        compounds.append({
            'name': name,
            'conc': conc,
            'solubility': solubility,
            'diffusion': diffusion
        })
    
    density = float(input("\nSolution density (g/cm³, e.g., 1.0 for water): "))
    viscosity = float(input("Solution viscosity (Pa·s, e.g., 0.001 for water): "))
    surface_tension = float(input("Solution surface tension (N/m, e.g., 0.072 for water): "))
    gas_temp = float(input("Drying gas temperature (°C, e.g., 40): "))
    droplet_diameter = float(input("Droplet diameter (microns, e.g., 30): "))
    K = float(input("Hydrogen bonding equilibrium constant (mL/mg, e.g., 50, or 0 for none): "))
    
    return compounds, density, viscosity, surface_tension, gas_temp, droplet_diameter, K

def calculate_droplet_properties(droplet_diameter, compounds, density):
    radius = droplet_diameter * 1e-4 / 2  # microns to cm
    volume = (4/3) * math.pi * radius**3  # cm³
    volume_ml = volume * 1000  # mL
    masses = [c['conc'] * volume_ml for c in compounds]  # µg
    return volume_ml, masses, radius

def evaporation_rate(radius, gas_temp, density):
    D_vap = 0.26  # cm²/s
    P_sat = 2338.1 * math.exp(17.2694 - 4102.99 / (gas_temp + 237.431))  # Pa
    rho = density  # g/cm³
    R = 8.314  # J/(mol·K)
    T = gas_temp + 273.15  # K
    RH = 0.2
    
    dV_dt = -4 * math.pi * radius**2 * (D_vap * P_sat / (rho * R * T)) * (1 - RH)
    dV_dt *= 1000  # cm³/s to mL/s
    v_evap = abs(dV_dt) / (4 * math.pi * radius**2)  # cm/s
    return dV_dt, v_evap

def calculate_peclet(radius, v_evap, diffusion):
    Pe = (radius * v_evap) / diffusion
    return Pe

def hydrogen_bonding_concentrations(masses, volume, K, idx_low, idx_high):
    if K == 0:
        return [m / volume for m in masses], 0, 0
    m_low, m_high = masses[idx_low], masses[idx_high]
    a = K
    b = -K * (m_low + m_high) / volume
    c = m_low * m_high / (volume**2)
    AD = (-b - math.sqrt(b**2 - 4 * a * c)) / (2 * a)
    conc = [m / volume for m in masses]
    conc[idx_low] -= AD
    conc[idx_high] -= AD
    return conc, AD, 55  # A-D solubility

def simulate_precipitation(compounds, masses, volume_initial, radius, gas_temp, K):
    sorted_indices = sorted(range(len(compounds)), key=lambda i: compounds[i]['solubility'])
    idx_low, idx_high = sorted_indices[0], sorted_indices[-1]
    
    # Calculate Peclet number
    _, v_evap = evaporation_rate(radius, gas_temp, 1.0)
    avg_diffusion = sum(c['diffusion'] for c in compounds) / len(compounds)
    Pe = calculate_peclet(radius, v_evap, avg_diffusion)
    print(f"Peclet number: {Pe:.4f}")
    
    # Low Pe: Uniform precipitation
    critical_volumes = []
    conc, AD, S_AD = hydrogen_bonding_concentrations(masses, volume_initial, K, idx_low, idx_high)
    if K > 0 and AD > S_AD:
        V_AD = math.sqrt(masses[idx_low] * masses[idx_high] * K / S_AD)
        critical_volumes.append(('A-D', V_AD))
    for i, c in enumerate(compounds):
        V_i = masses[i] / c['solubility'] if conc[i] > c['solubility'] else volume_initial
        critical_volumes.append((c['name'], V_i))
    
    critical_volumes = sorted(critical_volumes, key=lambda x: x[1], reverse=True)
    critical_volumes = [(name, max(0, min(v, volume_initial))) for name, v in critical_volumes]
    
    V_steps = np.linspace(volume_initial, 0, 100)
    surface_mass = {c['name']: 0 for c in compounds}
    core_mass = {c['name']: 0 for c in compounds}
    AD_surface, AD_core = 0, 0
    epsilon = 0.05  # Reduced mixing for low Pe
    
    for i in range(1, len(V_steps)):
        V = V_steps[i]
        precipitating = [name for name, V_crit in critical_volumes if V < V_crit <= V_steps[i-1]]
        
        for name in precipitating:
            idx = next(i for i, c in enumerate(compounds) if c['name'] == name)
            if name == 'A-D':
                dV = V_steps[i-1] - V
                m_precip = 55 * dV
                if V > volume_initial / 2:
                    AD_surface += m_precip
                else:
                    AD_core += m_precip
            else:
                m_precip = max(0, masses[idx] - compounds[idx]['solubility'] * V)
                if m_precip > 0:
                    if V > volume_initial / 2:
                        surface_mass[name] += m_precip * 0.6  # Slight A enrichment
                        core_mass[name] += m_precip * 0.4
                    else:
                        core_mass[name] += m_precip
                    masses[idx] -= m_precip
        
        total_precip = sum(surface_mass.values()) + AD_surface if V > volume_initial / 2 else sum(core_mass.values()) + AD_core
        for c in compounds:
            name = c['name']
            if name not in precipitating:
                m_mix = epsilon * total_precip / max(1, len(compounds) - len(precipitating))
                if V > volume_initial / 2:
                    surface_mass[name] += m_mix
                    core_mass[name] += m_mix * 0.67
                else:
                    core_mass[name] += m_mix
                masses[compounds.index(c)] -= m_mix
    
    for AD_mass, zone in [(AD_surface, surface_mass), (AD_core, core_mass)]:
        zone[compounds[idx_low]['name']] += AD_mass / 2
        zone[compounds[idx_high]['name']] += AD_mass / 2
    
    return surface_mass, core_mass

def plot_results(shell_mass, core_mass, compounds):
    names = [c['name'] for c in compounds]
    shell_values = [shell_mass[name] for name in names]
    core_values = [core_mass[name] for name in names]
    
    fig = go.Figure(data=[
        go.Bar(name=name, x=['Surface', 'Core'], y=[shell_values[i], core_values[i]])
        for i, name in enumerate(names)
    ])
    
    fig.update_layout(
        barmode='stack',
        title='Mass of Compounds in Spherical Particle Zones (Low Pe)',
        xaxis_title='Zone',
        yaxis_title='Mass (µg)',
        legend_title='Compound'
    )
    
    fig.show()

def main():
    try:
        compounds, density, viscosity, surface_tension, gas_temp, droplet_diameter, K = get_user_input()
        volume_initial, masses, radius = calculate_droplet_properties(droplet_diameter, compounds, density)
        dV_dt, _ = evaporation_rate(radius, gas_temp, density)
        evap_time = volume_initial / abs(dV_dt)
        print(f"\nDroplet volume: {volume_initial:.2e} mL")
        print(f"Evaporation rate: {dV_dt:.2e} mL/s")
        print(f"Evaporation time: {evap_time:.2e} s")
        
        surface_mass, core_mass = simulate_precipitation(compounds, masses.copy(), volume_initial, radius, gas_temp, K)
        
        print("\nSurface Zone Composition (µg):")
        for c in compounds:
            print(f"{c['name']}: {surface_mass[c['name']]:.4f}")
        print("\nCore Zone Composition (µg):")
        for c in compounds:
            print(f"{c['name']}: {core_mass[c['name']]:.4f}")
        
        plot_results(surface_mass, core_mass, compounds)
        
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()