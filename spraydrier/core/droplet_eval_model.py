import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def droplet_evaporation_model(
    a0_um=100,              # initial radius in μm
    xi0=0.10,               # initial solids mass fraction
    RH=50,                  # relative humidity %
    T=25,                   # temperature °C
    xi_crust=0.35,          # surface fraction when crust forms
    shell_permeability_factor=1e-3,
    E=1e9,                  # ← NEW: Young's modulus of the crust in Pa
    nu=0.3,                 # ← NEW: Poisson's ratio (0.3 typical)
    rho_solid=1500,         # ← NEW: density of the solid material in kg/m³
    plot=True
):
    a0 = a0_um * 1e-6
    rho_water = 1000
    rho_droplet = 1000
    
    K1 = 1.2e-10 * (1 - RH/100)   # m²/s
    
    a_crust = a0 * (xi0 / xi_crust)**(1/3)
    t_crust = (a0**2 - a_crust**2) / K1
    
    K2 = K1 * a_crust**2 * shell_permeability_factor
    
    m_solids = xi0 * (4/3)*np.pi*a0**3 * rho_droplet
    gamma = 0.072  # surface tension water

    # ==============================================================
    #     NEW: BUCKLING PREDICTION (inserted here)
    # ==============================================================
    def predict_final_morphology():
        vol_solids = m_solids / rho_solid
        # Approximate average shell thickness at crust formation
        inner_void_at_crust = (a_crust**3 - 3*vol_solids/(4*np.pi))**(1/3)
        h_avg = a_crust - inner_void_at_crust if inner_void_at_crust > 0 else a_crust*0.02

        # Critical buckling pressure for thin spherical shell (Zoelly formula, simplified)
        def P_crit(r_void):
            h = a_crust - r_void
            if h <= 0: return 1e15
            return 8 * E * (h/a_crust)**2 / np.sqrt(3*(1-nu**2))

        # Capillary pressure as void shrinks
        def P_cap(r_void):
            return 2*gamma / r_void   # assumes contact angle ≈0 inside porous shell

        # Find the void radius where P_cap first exceeds P_crit
        try:
            r_buckle = fsolve(lambda r: P_cap(r) - P_crit(r), a_crust*0.7)[0]
        except:
            r_buckle = a_crust*0.1   # fallback

        reduction = r_buckle / a_crust

        print("\n=== FINAL MORPHOLOGY PREDICTION ===")
        print(f"Crust forms at radius       : {a_crust*1e6:.1f} μm")
        print(f"Shell thickness (approx)    : {h_avg*1e6:.2f} μm")
        print(f"Young's modulus E           : {E/1e9:.2f} GPa")
        print(f"Buckling starts at void radius ≈ {r_buckle*1e6:.1f} μm")

        if xi0 > 0.4:
            morph = "Dense solid particle (no hollow stage)"
        elif reduction > 0.95:
            morph = "Nearly perfect hollow sphere (very stiff shell)"
        elif reduction > 0.8:
            morph = "Lightly dimpled hollow sphere"
        elif reduction > 0.5:
            morph = "Clearly buckled / crumpled sphere"
        elif reduction > 0.2:
            morph = "Strongly collapsed → raisin-like or multi-dimpled"
        else:
            morph = "Extreme collapse → donut/toroidal or flat particle possible"

        print(f"→ Predicted final morphology: {morph}")
        return morph, a_crust*1e6, reduction

    # ==============================================================
    #     End of new buckling section
    # ==============================================================

    # Time integration (unchanged)
    t = np.linspace(0, t_crust + 50, 2000)   # extended a bit
    a = np.zeros_like(t)

    for i, ti in enumerate(t):
        if ti <= t_crust:
            a[i] = np.sqrt(a0**2 - K1 * ti)
        else:
            remaining = a_crust**3 - 3*K2*(ti - t_crust)
            if remaining > 0:
                a[i] = remaining**(1/3)
            else:
                a[i] = a[-2] if i>0 else a_crust   # freeze at final size
                break

    # Run the morphology prediction and print
    final_morphology, a_crust_um, buckle_ratio = predict_final_morphology()

    if plot:
        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(t, a*1e6, 'b-', linewidth=2.5, label='Particle radius')
        plt.axvline(t_crust, color='r', linestyle='--', label=f'Crust formation ({a_crust_um:.1f} μm)')
        plt.axhline(a_crust_um, color='orange', linestyle=':', linewidth=2, label='Hollow shell (no buckle)')
        plt.xlabel('Time (s)')
        plt.ylabel('Radius (μm)')
        plt.title(f'10% solids droplet → {final_morphology.split(" → ")[-1]}\nInitial radius = {a0_um} μm | E = {E/1e9:.2f} GPa')
        plt.grid(alpha=0.3)
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(t, (a/a0)**3, 'g-', linewidth=2.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Volume / initial volume')
        plt.title('Volume evolution')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    return t, a*1e6, a_crust*1e6, final_morphology

# ================================
# Easy examples – just change E!
# ================================

# Stiff salt crust → light buckling
droplet_evaporation_model(a0_um=100, E=15e9, rho_solid=2170)   # NaCl-like

# Typical case → clear buckling
droplet_evaporation_model(a0_um=100, E=2e9)                    # many salts/oxides

# Soft protein/polymer → raisin or donut
droplet_evaporation_model(a0_um=100, E=0.2e9, rho_solid=1300)  # milk-like