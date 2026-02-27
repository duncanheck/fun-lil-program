import numpy as np
import logging
import argparse
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False
    import matplotlib.pyplot as plt
from scipy.integrate import odeint
import math

def get_user_input():
    """Collect user input for compounds and solution properties."""
    print("Enter details for up to 5 compounds.")
    compounds = []
    num_compounds = int(input("Number of compounds (1-5): "))
    if not 1 <= num_compounds <= 5:
        raise ValueError("Number of compounds must be between 1 and 5.")

    for i in range(num_compounds):
        print(f"\nCompound {i+1}:")
        name = input("Name: ")
        conc = float(input("Initial concentration (mg/mL): "))
        solubility = float(input("Solubility at drying temperature (mg/mL): "))
        diffusion = float(input("Diffusion coefficient (m^2/s, e.g., 1e-10): "))
        # Adsorption kinetics (optional) - prompt with default placeholders
        try:
            k_ads = input("k_ads (mL/(mg·s) or press Enter for default 1e-3): ")
            k_ads = float(k_ads) if k_ads.strip() != '' else 1e-3
        except Exception:
            k_ads = 1e-3
        try:
            k_des = input("k_des (1/s) or press Enter for default 1e-3): ")
            k_des = float(k_des) if k_des.strip() != '' else 1e-3
        except Exception:
            k_des = 1e-3
        compounds.append({
            'name': name,
            'conc': conc,
            'solubility': solubility,
            'diffusion': diffusion
            , 'k_ads': k_ads, 'k_des': k_des
        })

    # Solution properties
    density = float(input("\nSolution density (g/mL, e.g., 1.0 for water): "))
    viscosity_cP = float(input("Solution viscosity (cP, centipoise, e.g., 1.0 for water): "))
    # convert cP -> Pa·s (1 cP = 0.001 Pa·s)
    viscosity = viscosity_cP * 0.001
    surface_tension_mNm = float(input("Solution surface tension (mN/m, e.g., 72 for water): "))
    # convert mN/m -> N/m
    surface_tension = surface_tension_mNm * 0.001

    # Drying conditions
    gas_temp = float(input("Drying gas temperature (°C, e.g., 40): "))
    gas_velocity = float(input("Drying gas velocity (m/s, e.g., 0.5): "))
    droplet_diameter = float(input("Droplet diameter (microns, e.g., 30): "))

    # Hydrogen bonding (between least and most soluble compounds)
    K = float(input("Hydrogen bonding equilibrium constant (mL/mg, e.g., 50, or 0 for none): "))
    try:
        Gamma_max = float(input("Surface capacity Gamma_max (mg/m^2) or press Enter for default 1e-3): "))
    except Exception:
        Gamma_max = 1e-3

    return compounds, density, viscosity, surface_tension, gas_temp, gas_velocity, droplet_diameter, K, Gamma_max

def calculate_droplet_properties(droplet_diameter, compounds, density):
    """Calculate droplet volume and initial masses."""
    # droplet_diameter is in microns; convert to cm for volume calculation
    radius = droplet_diameter * 1e-4 / 2  # microns -> cm
    volume_cm3 = (4/3) * math.pi * radius**3  # cm³
    # 1 cm³ == 1 mL
    volume_ml = volume_cm3
    # masses: concentration (mg/mL) * volume (mL) = mass in mg
    masses = [c['conc'] * volume_ml for c in compounds]
    return volume_ml, masses, radius

def evaporation_rate(radius, gas_temp, density, surface_tension, viscosity, gas_velocity):
    """Estimate evaporation rate (mL/s) using diffusion-limited rate with a Sherwood (convective) correction.

    This is an approximate model: compute a base diffusion-limited flux and multiply by Sh/2 to account
    for convective enhancement (Sh >= 2). Units are approximate but consistent for qualitative use.
    """
    # Constants and conversions
    # radius is provided in cm (from calculate_droplet_properties); convert to meters
    radius_m = radius * 1e-2
    D_vap = 2.6e-5  # water vapor diffusivity in air (m^2/s)
    P_sat = 2338.1 * math.exp(17.2694 - 4102.99 / (gas_temp + 237.431))  # Pa, at gas_temp
    rho_liq = density * 1000.0  # g/mL -> kg/m^3 (1 g/mL = 1000 kg/m^3)
    R_gas = 8.314  # J/(mol·K)
    T = gas_temp + 273.15  # K
    RH = 0.2

    # Air properties for Re/Sc
    rho_air = 1.2  # kg/m^3
    mu_air = 1.81e-5  # Pa·s

    # Reynolds and Schmidt numbers for a sphere in airflow
    U = max(0.0, gas_velocity)
    Re = rho_air * U * (2 * radius_m) / mu_air if U > 0 else 0.0
    Sc = mu_air / (rho_air * D_vap)
    Sh = 2.0 + 0.6 * (Re ** 0.5) * (Sc ** (1/3)) if Re > 0 else 2.0

    # Base diffusion-limited volume flux (m^3/s): approximate using D_vap and saturation pressure
    # This follows the form used previously but converted to SI and then corrected by Sh/2.
    # Note: this is an approximation for qualitative behavior.
    base_flux_m3s = 4 * math.pi * (radius_m ** 2) * (D_vap * P_sat / (R_gas * T)) / (rho_liq) * (1.0 - RH)

    # Apply convective enhancement
    flux_m3s = base_flux_m3s * (Sh / 2.0)

    # Convert m^3/s to mL/s (1 m^3 = 1e6 mL)
    dV_dt_mL_s = flux_m3s * 1e6
    # Return negative to indicate volume loss
    return -abs(dV_dt_mL_s)

def hydrogen_bonding_concentrations(masses, volume, K, idx_low, idx_high):
    """Calculate concentrations of free and complexed species."""
    if K == 0:
        return [m / volume for m in masses], 0, 0

    m_low, m_high = masses[idx_low], masses[idx_high]
    # Solve quadratic for [A-D]: K [A-D]^2 - K (m_A + m_D)/V [A-D] + m_A m_D / V^2 = 0
    a = K
    b = -K * (m_low + m_high) / volume
    c = m_low * m_high / (volume**2)
    AD = (-b - math.sqrt(b**2 - 4 * a * c)) / (2 * a)

    conc = [m / volume for m in masses]
    conc[idx_low] -= AD
    conc[idx_high] -= AD
    return conc, AD, 55  # Assume A-D solubility = 55 mg/mL

def simulate_precipitation(compounds, masses, volume_initial, radius, gas_temp, K,
                           density, surface_tension, viscosity, gas_velocity, Gamma_max,
                           max_steps=10000, min_volume_frac=0.001):
    """Simulate precipitation and zone formation with time-dependent evaporation.

    masses: list of initial dissolved masses in mg
    volume_initial: initial droplet volume in mL
    radius: initial radius in cm
    Returns shell_mass and core_mass (both dicts, mass units same as input mg)
    """
    # Sort compounds by solubility to identify low/high for H-bonding
    sorted_indices = sorted(range(len(compounds)), key=lambda i: compounds[i]['solubility'])
    idx_low, idx_high = sorted_indices[0], sorted_indices[-1]

    # Compute critical volumes where concentration equals solubility: V_crit = mass / solubility
    critical_volumes = []
    for i, c in enumerate(compounds):
        sol = c.get('solubility', 0.0)
        if sol > 0:
            V_i = masses[i] / sol
        else:
            V_i = float('inf')
        critical_volumes.append((c['name'], V_i))

    # A-D complex critical volume (approx) if hydrogen bonding present
    conc0, AD0, S_AD = hydrogen_bonding_concentrations(masses, volume_initial, K, idx_low, idx_high)
    if K > 0 and S_AD > 0:
        V_AD = math.sqrt(max(0.0, masses[idx_low] * masses[idx_high] * K / S_AD))
        critical_volumes.append(('A-D', V_AD))

    critical_volumes = sorted(critical_volumes, key=lambda x: x[1], reverse=True)
    critical_volumes = [(name, max(0.0, min(v, volume_initial))) for name, v in critical_volumes]

    # Initialize
    shell_mass = {c['name']: 0.0 for c in compounds}
    core_mass = {c['name']: 0.0 for c in compounds}
    # Adsorbed surface coverage (mass per area, mg/m^2)
    Gamma = {c['name']: 0.0 for c in compounds}
    # Adsorbed total mass (mg) per species
    adsorbed_mass = {c['name']: 0.0 for c in compounds}
    AD_shell, AD_core = 0.0, 0.0
    epsilon = 0.1  # Marangoni mixing
    initial_masses = masses.copy()  # Save initial masses for percentage calculations

    # Estimate evaporation time and timestep
    # Use initial evaporation rate to get approximate time scale
    dV_dt0 = evaporation_rate(radius, gas_temp, density, surface_tension, viscosity, gas_velocity)
    if dV_dt0 == 0:
        est_time = 1.0
    else:
        est_time = abs(volume_initial / dV_dt0)
    nsteps = min(max_steps, max(100, int(est_time * 1000)))

    V = volume_initial
    V_prev = V
    dt = est_time / float(nsteps) if est_time > 0 else 1e-6

    for step in range(nsteps):
        # compute current radius from volume (volume in mL -> convert to m^3? but relative scale OK)
        # radius in m from V (mL): 1 mL = 1e-6 m^3
        V_m3 = V * 1e-6
        if V_m3 <= 0:
            break
        r_m = ((3.0 * V_m3) / (4.0 * math.pi)) ** (1.0 / 3.0)
        # convert to cm for evaporation_rate which expects radius in cm
        r_cm = r_m * 100.0

        # Evaporation: compute dV/dt (mL/s)
        dV_dt = evaporation_rate(r_cm, gas_temp, density, surface_tension, viscosity, gas_velocity)
        V_new = max(0.0, V + dV_dt * dt)

        # Surface area in m^2 (V in mL -> m^3)
        surface_area = 4.0 * math.pi * (r_m ** 2)

        # Adsorption kinetics: Langmuir-like competitive adsorption
        # dGamma_i/dt = k_ads_i * c_bulk * (1 - sum(Gamma)/Gamma_max) - k_des_i * Gamma_i
        # c_bulk in mg/mL, convert to mg/m^3 for consistent units? We'll use mg/mL with k_ads in mL/(mg*s)
        sum_Gamma = sum(Gamma.values())
        avail_fraction = max(0.0, 1.0 - sum_Gamma / Gamma_max) if Gamma_max > 0 else 0.0
        previous_adsorbed = adsorbed_mass.copy()
        for i_c, c in enumerate(compounds):
            name = c['name']
            k_ads_i = c.get('k_ads', 1e-3)
            k_des_i = c.get('k_des', 1e-3)
            # bulk concentration in mg/mL = mass (mg) / volume (mL)
            c_bulk = masses[i_c] / V if V > 0 else 0.0
            dGamma_dt = k_ads_i * c_bulk * avail_fraction - k_des_i * Gamma[name]
            # Euler step
            Gamma[name] = max(0.0, Gamma[name] + dGamma_dt * dt)
            # update adsorbed mass (mg) = Gamma (mg/m^2) * area (m^2)
            adsorbed_mass[name] = Gamma[name] * surface_area
            # remove the adsorbed amount from dissolved pool (conserve mass)
            delta_adsorbed = adsorbed_mass[name] - previous_adsorbed[name]
            remove_amt = min(masses[i_c], delta_adsorbed)
            masses[i_c] = max(0.0, masses[i_c] - remove_amt)

        # Check which species precipitate when passing from V -> V_new
        precipitating = [name for name, V_crit in critical_volumes if V_new < V_crit <= V]

        for name in precipitating:
            if name == 'A-D':
                dV = V - V_new
                m_precip = 55.0 * dV  # approximate mass for complex (mg scale approximation)
                if V > volume_initial / 2.0:
                    AD_shell += m_precip
                else:
                    AD_core += m_precip
            else:
                idx = next(i for i, c in enumerate(compounds) if c['name'] == name)
                sol = compounds[idx]['solubility']
                # mass that exceeds solubility at new volume (clamped)
                m_precip = max(0.0, masses[idx] - sol * V_new)
                if m_precip > 0:
                    # only transfer up to the available dissolved mass
                    transfer = min(m_precip, masses[idx])
                    if V_new > volume_initial / 2.0:
                        shell_mass[name] += transfer
                    else:
                        core_mass[name] += transfer
                    masses[idx] = max(0.0, masses[idx] - transfer)

        # Marangoni mixing: move a small fraction of each species' own dissolved mass to the shell/core
        for i_c, c in enumerate(compounds):
            name = c['name']
            if name not in precipitating:
                # move a fraction of this species' own dissolved mass (no cross-species transfer)
                own_available = masses[i_c]
                m_mix = min(own_available, epsilon * own_available)
                if m_mix > 0:
                    if V > volume_initial / 2.0:
                        shell_mass[name] += m_mix
                    else:
                        core_mass[name] += m_mix
                    masses[i_c] = max(0.0, masses[i_c] - m_mix)

        V_prev = V
        V = V_new

        # Stop early if volume is very small
        if V <= volume_initial * min_volume_frac:
            break

    # Distribute remaining dissolved mass to core
    for i, c in enumerate(compounds):
        name = c['name']
        remaining = masses[i]
        if remaining > 0:
            core_mass[name] += remaining

    # Distribute A-D mass (1:1 molar, assume equal mass for simplicity)
    for AD_mass, zone in [(AD_shell, shell_mass), (AD_core, core_mass)]:
        zone[compounds[idx_low]['name']] += AD_mass / 2.0
        zone[compounds[idx_high]['name']] += AD_mass / 2.0

    return shell_mass, core_mass, adsorbed_mass, initial_masses

def plot_results(shell_mass, core_mass, compounds, out_prefix='surface_layer'):
    """Generate a stacked bar chart using Plotly."""
    names = [c['name'] for c in compounds]
    shell_values = [shell_mass[name] for name in names]
    core_values = [core_mass[name] for name in names]
    if _HAS_PLOTLY:
        fig = go.Figure(data=[
            go.Bar(name=name, x=['Shell', 'Core'], y=[shell_values[i], core_values[i]])
            for i, name in enumerate(names)
        ])
        fig.update_layout(
            barmode='stack',
            title='Mass of Compounds in Spherical Particle Zones',
            xaxis_title='Zone',
            yaxis_title='Mass (µg)',
            legend_title='Compound'
        )
        # save an interactive HTML so it's viewable later
        out_html = f"{out_prefix}.html"
        fig.write_html(out_html)
        print(f"Plot saved to {out_html}")
    else:
        # Simple matplotlib stacked bar plot
        ind = np.arange(2)  # Shell, Core
        width = 0.5
        bottom = np.zeros(2)
        fig, ax = plt.subplots()
        for i, name in enumerate(names):
            vals = [shell_values[i], core_values[i]]
            ax.bar(ind, vals, width, bottom=bottom, label=name)
            bottom += np.array(vals)
        ax.set_xticks(ind)
        ax.set_xticklabels(['Shell', 'Core'])
        ax.set_title('Mass of Compounds in Spherical Particle Zones')
        ax.set_ylabel('Mass (µg)')
        ax.legend(title='Compound')
    out_png = f"{out_prefix}.png"
    fig.savefig(out_png, bbox_inches='tight')
    print(f"Plot saved to {out_png}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description='Surface layer precipitation simulator')
    parser.add_argument('--no-plot', action='store_true', help='Do not create or save plots')
    parser.add_argument('--out-prefix', default='surface_layer', help='Prefix for output plot files')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    try:
        # Get user input (interactive)
        compounds, density, viscosity, surface_tension, gas_temp, gas_velocity, droplet_diameter, K, Gamma_max = get_user_input()

        # Calculate droplet properties
        volume_initial, masses, radius = calculate_droplet_properties(droplet_diameter, compounds, density)

        # Calculate evaporation rate
        dV_dt = evaporation_rate(radius, gas_temp, density, surface_tension, viscosity, gas_velocity)
        evap_time = volume_initial / abs(dV_dt) if dV_dt != 0 else float('inf')
        logging.info(f"Droplet volume: {volume_initial:.2e} mL")
        logging.info(f"Evaporation rate: {dV_dt:.2e} mL/s")
        logging.info(f"Evaporation time: {evap_time:.2e} s")

        # Print initial masses (mg and µg)
        logging.info("Initial masses:")
        for i, c in enumerate(compounds):
            m_mg = masses[i]
            m_ug = m_mg * 1000.0
            logging.info(f"  {c['name']}: {m_mg:.3e} mg ({m_ug:.3e} µg)")

        # Simulate precipitation (pass fluid & gas properties)
        shell_mass, core_mass, adsorbed, initial_masses = simulate_precipitation(
            compounds, masses.copy(), volume_initial, radius, gas_temp, K,
            density, surface_tension, viscosity, gas_velocity, Gamma_max
        )

        # Check if any precipitation occurred (non-zero shell/core)
        any_precip = any(v != 0 for v in list(shell_mass.values()) + list(core_mass.values()))
        if not any_precip:
            logging.info("No precipitation detected: initial concentrations did not exceed solubility for any compound.")

        # Output results in µg (scientific notation) and as percent of each compound
        logging.info("Shell Composition:")
        for i, c in enumerate(compounds):
            name = c['name']
            shell_mg = shell_mass.get(name, 0.0)
            core_mg = core_mass.get(name, 0.0)
            shell_ug = shell_mg * 1000.0
            core_ug = core_mg * 1000.0
            initial_mg = initial_masses[i]
            pct_shell = (shell_mg / initial_mg * 100.0) if initial_mg > 0 else 0.0
            pct_core = (core_mg / initial_mg * 100.0) if initial_mg > 0 else 0.0
            logging.info(f"  {name}: {shell_ug:.3e} µg ({pct_shell:.2f} % of initial)")
        logging.info("Core Composition:")
        for i, c in enumerate(compounds):
            name = c['name']
            core_mg = core_mass.get(name, 0.0)
            core_ug = core_mg * 1000.0
            initial_mg = initial_masses[i]
            pct_core = (core_mg / initial_mg * 100.0) if initial_mg > 0 else 0.0
            logging.info(f"  {name}: {core_ug:.3e} µg ({pct_core:.2f} % of initial)")

        # Adsorbed mass report
        logging.info("Adsorbed Mass (interface):")
        for i, c in enumerate(compounds):
            name = c['name']
            ad_mg = adsorbed.get(name, 0.0)
            ad_ug = ad_mg * 1000.0
            initial_mg = initial_masses[i]
            pct_ad = (ad_mg / initial_mg * 100.0) if initial_mg > 0 else 0.0
            logging.info(f"  {name}: {ad_ug:.3e} µg ({pct_ad:.2f} % of initial)")

        # Plot results (skip if requested)
        if not args.no_plot:
            plot_results(shell_mass, core_mass, compounds, out_prefix=args.out_prefix)
        else:
            logging.info('Skipping plot generation (--no-plot)')

    except ValueError as e:
        logging.error(f"Error: {e}")
    except Exception as e:
        logging.exception(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()