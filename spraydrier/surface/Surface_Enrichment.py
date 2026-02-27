import sys
import math
import pandas as pd
import numpy as np
import argparse
import re

"""
Time-dependent radial diffusion simulator for surface enrichment.

This script reads experimental/simulation parameters from an Excel workbook
and runs a two-solute radial diffusion + adsorption simulation for each
selected workbook column. The output is an Excel workbook
`surface_enrichment_timeseries.xlsx` containing time-series sheets.

Expected Excel row labels (case-insensitive substring matching):
- D32 (calculated with Moni) or similar -> particle diameter in micrometers (µm)
- Surface recession velocity or Evaporation Velocity -> v (m/s)
- D_moni, D_igg, D_ds, D_bhv -> diffusion coefficients (m^2/s)
- Drug Substance conc. (mg/mL) / ds_conc -> DS concentration (mg/mL)
- Moni conc. (mg/mL) / moni_conc -> Moni concentration (mg/mL)
- k_ads_DS (m/s), k_des_DS (1/s) -> adsorption/desorption params for DS
- k_ads_Moni (m/s), k_des_Moni (1/s) -> adsorption/desorption params for Moni
- Gamma_max (mg/m^2) -> maximum adsorbed mass per area (mg/m^2)
- Estimated Feed Surface Tension (N/m) or Surface tension -> sigma (N/m)
- Marangoni Number or Marangoni -> (dimensionless)

Notes on adsorption implementation:
- Adsorption follows a Langmuir-like kinetic form implemented as
    dGamma/dt = k_ads * c_bulk * (1 - Gamma/Gamma_max) - k_des * Gamma
    where k_ads has units m/s, c_bulk is bulk concentration in mg/m^3,
    k_des has units 1/s and Gamma is adsorbed mass per area (mg/m^2).

Outputs per time step include:
- pct_igg_surface, pct_moni_surface (percent composition at the surface)
- gamma_igg_mg_per_m2, gamma_moni_mg_per_m2 (adsorbed mass per area)
- Pe_igg, Pe_moni (Péclet numbers = v*R/D)
- v_mig_igg_m_per_s, v_mig_moni_m_per_s (migration velocity approximations)
- J_igg_mg_per_m2_s, J_moni_mg_per_m2_s (approx. migration flux)
"""


def find_row(df, substr):
    for idx in df.index:
        if isinstance(idx, str) and substr.lower() in idx.lower():
            return idx
    return None


def sanitize_sheet_name(name: str) -> str:
    # Excel sheet name limits: <=31 chars and cannot contain [: \/?*[]]
    s = re.sub(r"[:\\\\/?*\[\]]", "_", str(name))
    return s[:31]


def _read_params_from_column(df, col):
    # safe extraction with coercion and diagnostics
    out = {'last_col': col}
    def safe_get_row(substr_list, default=None):
        for s in substr_list:
            r = find_row(df, s)
            if r is not None:
                val = df.loc[r, col]
                return val, r
        return default, None

    # D32 -> R0 (D32 in µm)
    d32_val, d32_row = safe_get_row(['D32 (calculated', 'D32 (calculated with', 'D32 (calculated with Moni)'])
    R0 = None
    if d32_row is not None:
        try:
            d32_num = pd.to_numeric(d32_val, errors='coerce')
            if not np.isnan(d32_num):
                R0 = float(d32_num) / 2.0 * 1e-6
        except Exception:
            R0 = None
    # surface recession velocity
    v_val, v_row = safe_get_row(['Surface recession velocity', 'Evaporation Velocity'])
    v_evap = pd.to_numeric(v_val, errors='coerce')
    # diffusion coeffs
    D_moni_val, _ = safe_get_row(['D_moni', 'D_mo', 'D_moni ('])
    D_igg_val, _ = safe_get_row(['D_bhv-1300', 'D_igg', 'D_ds', 'D_bhv'])
    D_moni = pd.to_numeric(D_moni_val, errors='coerce')
    D_igg = pd.to_numeric(D_igg_val, errors='coerce')
    # surface tension and marangoni (if present)
    sigma_val, sigma_row = safe_get_row(['Estimated Feed Surface Tension (N/m)', 'Estimated Feed Surface Tension', 'Surface tension', 'Surface tension (N/m)'])
    mar_val, mar_row = safe_get_row(['Marangoni Number', 'Marangoni'])
    sigma = pd.to_numeric(sigma_val, errors='coerce')
    marangoni = pd.to_numeric(mar_val, errors='coerce')
    # concentrations (mg/mL)
    igg_val, igg_row = safe_get_row(['Drug Substance conc', 'Drug Substance conc. (mg/mL)', 'ds_conc'])
    moni_val, moni_row = safe_get_row(['Moni conc', 'Moni conc. (mg/mL)', 'moni_conc'])
    igg_conc = pd.to_numeric(igg_val, errors='coerce')
    moni_conc = pd.to_numeric(moni_val, errors='coerce')

    out.update({
        'R0': R0,
        'v_evap': float(v_evap) if not pd.isna(v_evap) else 0.0,
        'D_moni': float(D_moni) if not pd.isna(D_moni) else np.nan,
        'D_igg': float(D_igg) if not pd.isna(D_igg) else np.nan,
        'igg_conc': float(igg_conc) if not pd.isna(igg_conc) else None,
        'moni_conc': float(moni_conc) if not pd.isna(moni_conc) else None,
        'sigma': float(sigma) if not pd.isna(sigma) else None,
        'marangoni': float(marangoni) if not pd.isna(marangoni) else None,
    })

    # adsorption params (optional)
    k_ads_igg_val, _ = safe_get_row(['k_ads_DS (m/s)', 'k_ads_igg', 'k_ads_ds'])
    k_des_igg_val, _ = safe_get_row(['k_des_DS (1/s)', 'k_des_igg', 'k_des_ds'])
    k_ads_moni_val, _ = safe_get_row(['k_ads_Moni (m/s)', 'k_ads_moni'])
    k_des_moni_val, _ = safe_get_row(['k_des_Moni (1/s)', 'k_des_moni'])
    Gamma_max_val, _ = safe_get_row(['Gamma_max (mg/m^2)', 'Gamma_max', 'Gamma max'])

    k_ads_igg = pd.to_numeric(k_ads_igg_val, errors='coerce')
    k_des_igg = pd.to_numeric(k_des_igg_val, errors='coerce')
    k_ads_moni = pd.to_numeric(k_ads_moni_val, errors='coerce')
    k_des_moni = pd.to_numeric(k_des_moni_val, errors='coerce')
    Gamma_max = pd.to_numeric(Gamma_max_val, errors='coerce')

    out.update({
        'k_ads_igg': float(k_ads_igg) if not pd.isna(k_ads_igg) else None,
        'k_des_igg': float(k_des_igg) if not pd.isna(k_des_igg) else None,
        'k_ads_moni': float(k_ads_moni) if not pd.isna(k_ads_moni) else None,
        'k_des_moni': float(k_des_moni) if not pd.isna(k_des_moni) else None,
        'Gamma_max': float(Gamma_max) if not pd.isna(Gamma_max) else None,
    })

    # diagnostic print
    print('\nDiagnostics for column:', col)
    rows_to_check = [d32_row, v_row, igg_row, moni_row]
    for r in [d32_row, v_row, igg_row, moni_row, sigma_row, mar_row]:
        if r is None:
            continue
        val = df.loc[r, col]
        print(f"  {r}: {val}")

    return out


def load_from_workbook(fn='data.xlsx', batch_ids=None):
    # read workbook with index in first column
    df = pd.read_excel(fn, index_col=0)
    cols = list(df.columns)
    def norm_col(s):
        return re.sub(r"\s+", "", str(s)).lower()

    if batch_ids:
        # normalize and match case-insensitively; allow multiple comma-separated ids
        selected = []
        for bid in batch_ids:
            bid = str(bid).strip()
            # exact normalized match first
            matches = [c for c in cols if isinstance(c, str) and norm_col(c) == norm_col(bid)]
            if not matches:
                # try substring normalized match
                matches = [c for c in cols if isinstance(c, str) and norm_col(bid) in norm_col(c)]
            if matches:
                selected.extend(matches)
            else:
                print(f"Warning: batch id '{bid}' not found in workbook columns")
        if not selected:
            print('No requested batch ids found; falling back to last column')
            selected = [cols[-1]]
    else:
        # interactive selection: list available columns and ask user to choose
        print('\nAvailable columns in workbook:')
        for i, c in enumerate(cols):
            print(f"  {i+1:3d}: {c}")
        ans = input('\nEnter comma-separated column numbers to process (or "all" to process all, Enter to use last column): ').strip()
        if ans == '' or ans.lower() in ('last',):
            selected = [cols[-1]]
        elif ans.lower() in ('all', '*'):
            selected = cols[:]
        else:
            picks = []
            for part in ans.split(','):
                part = part.strip()
                try:
                    idx = int(part) - 1
                    if 0 <= idx < len(cols):
                        picks.append(cols[idx])
                except ValueError:
                    # allow entering a literal column header
                    matches = [c for c in cols if norm_col(c) == norm_col(part) or norm_col(part) in norm_col(c)]
                    if matches:
                        picks.extend(matches)
            selected = picks if picks else [cols[-1]]

    # build param dict per selected column
    params_map = {}
    for col in selected:
        params_map[col] = _read_params_from_column(df, col)
    return params_map


def simulate_two_species(R0, v_evap, D1, D2, c1_0, c2_0, R_final_frac=0.12, N=120, t_max=None,
                         k_ads_1=0.0, k_des_1=0.0, k_ads_2=0.0, k_des_2=0.0, Gamma_max=1.0):
    """Simulate two solutes (1=DS, 2=Moni) in normalized coordinate xi in [0,1].
    Returns time-series arrays for:
      times, pct1_surface, pct2_surface, gamma1, gamma2,
      Pe1, Pe2, v_mig1, v_mig2, J1, J2
    All arrays have matching lengths (one entry per time step).
    """
    xi = np.linspace(0.0, 1.0, N)
    # initial concentrations (mg/mL)
    # Defensive defaults: if caller passed NaN/None for D or concentrations, coerce to sensible defaults
    try:
        if c1_0 is None or (isinstance(c1_0, float) and np.isnan(c1_0)):
            print('Warning: c1_0 (DS concentration) missing; defaulting to 47.5 mg/mL')
            c1_0 = 47.5
    except Exception:
        c1_0 = 47.5
    try:
        if c2_0 is None or (isinstance(c2_0, float) and np.isnan(c2_0)):
            # if Moni missing default to 0 mg/mL (no excipient) to avoid spurious 100% values
            print('Warning: c2_0 (Moni concentration) missing; defaulting to 0.0 mg/mL')
            c2_0 = 0.0
    except Exception:
        c2_0 = 0.0

    # Diffusion defaults
    if D1 is None or (isinstance(D1, float) and np.isnan(D1)):
        print('Warning: D1 (DS diffusion) missing; defaulting to 2.5e-11 m^2/s')
        D1 = 2.5e-11
    if D2 is None or (isinstance(D2, float) and np.isnan(D2)):
        print('Warning: D2 (Moni diffusion) missing; defaulting to 2.267e-11 m^2/s')
        D2 = 2.267349230220751e-11

    c1 = np.ones(N) * c1_0
    c2 = np.ones(N) * c2_0

    t = 0.0
    R = R0
    R_final = R0 * R_final_frac

    times = []
    pct1_list = []
    pct2_list = []
    gamma1_ts = []
    gamma2_ts = []
    Pe1_ts = []
    Pe2_ts = []
    v_mig1_ts = []
    v_mig2_ts = []
    J1_ts = []
    J2_ts = []

    gamma1 = 0.0
    gamma2 = 0.0

    iters = 0
    max_iters = 200000

    while R > R_final and iters < max_iters:
        iters += 1
        # timestep
        dt = min(1e-3, 0.1 * R / max(1e-12, abs(v_evap)))
        if t + dt > (t_max if t_max is not None else 1e9):
            dt = (t_max - t) if t_max is not None else dt

        dRdt = -abs(v_evap)

        # build radial grid
        dr = R / N
        r_centers = (np.arange(N) + 0.5) * dr

        # diffusion operators
        def build_L(D):
            L = np.zeros((N, N))
            for i in range(N):
                ri = r_centers[i]
                r_im = ri - 0.5 * dr
                r_ip = ri + 0.5 * dr
                if i == 0:
                    L[0, 0] = -6.0 * D / (dr * dr)
                    if N > 1:
                        L[0, 1] = 6.0 * D / (dr * dr)
                elif i == N - 1:
                    coefL = D * r_im**2 / (ri**2 * dr * dr)
                    L[i, i-1] = -coefL
                    L[i, i] = coefL
                else:
                    coef_im = D * r_im**2 / (ri**2 * dr * dr)
                    coef_ip = D * r_ip**2 / (ri**2 * dr * dr)
                    L[i, i-1] = -coef_im
                    L[i, i] = coef_im + coef_ip
                    L[i, i+1] = -coef_ip
            return L

        L1 = build_L(D1)
        L2 = build_L(D2)

        I = np.eye(N)
        M1 = I - dt * L1
        M2 = I - dt * L2
        try:
            c1_new = np.linalg.solve(M1, c1)
            c2_new = np.linalg.solve(M2, c2)
        except np.linalg.LinAlgError:
            print('LinAlgError solving implicit system; aborting')
            break

        c1 = np.maximum(c1_new, 0.0)
        c2 = np.maximum(c2_new, 0.0)

        # shrink domain renormalization
        R_new = R + dRdt * dt
        if R_new <= 0:
            break
        vol_scale = (R_new / R)**3
        c1 *= 1.0 / vol_scale
        c2 *= 1.0 / vol_scale

        t += dt
        R = R_new

        # outer shell mass and concentrations
        shell_vols = ((np.arange(1, N+1) * dr)**3 - (np.arange(0, N) * dr)**3)
        masses1 = c1 * shell_vols
        masses2 = c2 * shell_vols
        dissolved_mass1_surface = masses1[-1]
        dissolved_mass2_surface = masses2[-1]
        vol_shell = shell_vols[-1]
        conc1_outer_mg_per_m3 = dissolved_mass1_surface / vol_shell * 1e6
        conc2_outer_mg_per_m3 = dissolved_mass2_surface / vol_shell * 1e6

        # adsorption kinetics -- Langmuir-like kinetic form
        # convert outer-shell concentration from mg/m^3 to use in J_ads = k_ads * c_bulk
        # Langmuir kinetics: dGamma/dt = k_ads * c_bulk * (1 - Gamma/Gamma_max) - k_des * Gamma
        if Gamma_max <= 0:
            # fallback to simple linear kinetics if Gamma_max invalid
            J_ads_1 = (k_ads_1 or 0.0) * conc1_outer_mg_per_m3
            J_ads_2 = (k_ads_2 or 0.0) * conc2_outer_mg_per_m3
            dgamma1_dt = J_ads_1 - (k_des_1 or 0.0) * gamma1
            dgamma2_dt = J_ads_2 - (k_des_2 or 0.0) * gamma2
        else:
            # shared-site Langmuir-like kinetics: adsorption is proportional to available sites
            available = max(0.0, 1.0 - (gamma1 + gamma2) / Gamma_max)
            dgamma1_dt = (k_ads_1 or 0.0) * conc1_outer_mg_per_m3 * available - (k_des_1 or 0.0) * gamma1
            dgamma2_dt = (k_ads_2 or 0.0) * conc2_outer_mg_per_m3 * available - (k_des_2 or 0.0) * gamma2

        # ensure we don't exceed capacity due to discrete step overshoot
        total_gamma = gamma1 + gamma2
        if total_gamma + (dgamma1_dt + dgamma2_dt) * dt > Gamma_max and (dgamma1_dt + dgamma2_dt) > 0:
            remaining = max(0.0, Gamma_max - total_gamma)
            incoming = max(1e-30, (dgamma1_dt + dgamma2_dt) * dt)
            scale = remaining / incoming
            dgamma1_dt *= scale
            dgamma2_dt *= scale

        gamma1 += dgamma1_dt * dt
        gamma2 += dgamma2_dt * dt
        gamma1 = min(max(gamma1, 0.0), Gamma_max)
        gamma2 = min(max(gamma2, 0.0), Gamma_max)

        # surface composition
        adsorbed_total = gamma1 + gamma2
        if adsorbed_total > 0 and any([k_ads_1, k_ads_2, k_des_1, k_des_2]):
            pct1_surface = 100.0 * gamma1 / adsorbed_total
            pct2_surface = 100.0 * gamma2 / adsorbed_total
        else:
            denom = (dissolved_mass1_surface + dissolved_mass2_surface)
            pct1_surface = 100.0 * dissolved_mass1_surface / denom if denom > 0 else 0.0
            pct2_surface = 100.0 - pct1_surface

        times.append(t)
        pct1_list.append(pct1_surface)
        pct2_list.append(pct2_surface)
        gamma1_ts.append(gamma1)
        gamma2_ts.append(gamma2)

        # Pe, migration velocity and flux (computed per time step)
        Pe1 = (abs(v_evap) * R) / max(1e-30, D1)
        Pe2 = (abs(v_evap) * R) / max(1e-30, D2)
        v_mig1 = v_evap * (1.0 - 1.0 / Pe1) if Pe1 != 0 else 0.0
        v_mig2 = v_evap * (1.0 - 1.0 / Pe2) if Pe2 != 0 else 0.0
        J1 = v_evap * conc1_outer_mg_per_m3 * (Pe1 / (Pe1 + 1.0))
        J2 = v_evap * conc2_outer_mg_per_m3 * (Pe2 / (Pe2 + 1.0))

        Pe1_ts.append(Pe1)
        Pe2_ts.append(Pe2)
        v_mig1_ts.append(v_mig1)
        v_mig2_ts.append(v_mig2)
        J1_ts.append(J1)
        J2_ts.append(J2)

    return (np.array(times), np.array(pct1_list), np.array(pct2_list), np.array(gamma1_ts), np.array(gamma2_ts),
            np.array(Pe1_ts), np.array(Pe2_ts), np.array(v_mig1_ts), np.array(v_mig2_ts), np.array(J1_ts), np.array(J2_ts))


def main():
    parser = argparse.ArgumentParser(description='Surface enrichment timeseries generator')
    parser.add_argument('--workbook', type=str, default='data.xlsx',
                        help='Path to the workbook to read parameters from (default: data.xlsx)')
    parser.add_argument('--batch-ids', type=str, help='Comma-separated batch ids or column headers to process (case-insensitive). If omitted, will prompt interactively or use last column.')
    args = parser.parse_args()

    batch_ids = None
    if args.batch_ids:
        batch_ids = [s.strip() for s in args.batch_ids.split(',') if s.strip()]

    # load parameters for requested columns (or last column as default)
    wb_path = args.workbook
    try:
        params_map = load_from_workbook(wb_path, batch_ids=batch_ids)
        print(f"Loaded parameters for columns: {list(params_map.keys())}")
    except Exception as e:
        print('Could not load workbook:', e)
        params_map = {}

    # For each selected column, run simulation and store results in separate sheets
    out_file = 'surface_enrichment_timeseries.xlsx'
    any_written = False
    with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
        for col, params in params_map.items():
            # apply fallbacks for missing params (treat NaN as missing)
            def coerce(val, default):
                try:
                    if val is None:
                        return default
                    # pandas uses numpy.nan which is float; pd.isna covers both
                    if pd.isna(val):
                        return default
                    return val
                except Exception:
                    return default

            R0 = float(coerce(params.get('R0'), 4.29e-6))
            v_evap = float(coerce(params.get('v_evap'), 5.7e-5))
            D_moni = float(coerce(params.get('D_moni'), 8.5e-11))
            D_igg = float(coerce(params.get('D_igg'), 2.5e-11))
            igg_conc = float(coerce(params.get('igg_conc'), 47.5))
            moni_conc = float(coerce(params.get('moni_conc'), 2.5))

            # if both D values were originally missing, warn and indicate fallbacks used
            if (params.get('D_moni') is None or pd.isna(params.get('D_moni'))) and (params.get('D_igg') is None or pd.isna(params.get('D_igg'))):
                print(f"Warning: both D_moni and D_igg missing for column '{col}'; using defaults D_moni={D_moni:.3e}, D_igg={D_igg:.3e}")

            sigma = params.get('sigma')
            mar = params.get('marangoni')
            print(f"Running simulation for column '{col}': R0={R0:.3e}, v_evap={v_evap:.3e}, D_igg={D_igg:.3e}, D_moni={D_moni:.3e}, sigma={sigma}, marangoni={mar}")
            # pass adsorption params (use defaults if None)
            k_ads_igg = float(coerce(params.get('k_ads_igg'), 0.0))
            k_des_igg = float(coerce(params.get('k_des_igg'), 0.0))
            k_ads_moni = float(coerce(params.get('k_ads_moni'), 0.0))
            k_des_moni = float(coerce(params.get('k_des_moni'), 0.0))
            Gamma_max = float(coerce(params.get('Gamma_max'), 1.0))

            (times, pct1, pct2, gamma1_ts, gamma2_ts,
             Pe1_ts, Pe2_ts, v_mig1_ts, v_mig2_ts, J1_ts, J2_ts) = simulate_two_species(
                R0, v_evap, D_igg, D_moni, igg_conc, moni_conc,
                k_ads_1=k_ads_igg, k_des_1=k_des_igg,
                k_ads_2=k_ads_moni, k_des_2=k_des_moni,
                Gamma_max=Gamma_max)

            out_df = pd.DataFrame({
                'time_s': times,
                'pct_igg_surface': pct1,
                'pct_moni_surface': pct2,
                'gamma_igg_mg_per_m2': gamma1_ts,
                'gamma_moni_mg_per_m2': gamma2_ts,
                'Pe_igg': Pe1_ts,
                'Pe_moni': Pe2_ts,
                'v_mig_igg_m_per_s': v_mig1_ts,
                'v_mig_moni_m_per_s': v_mig2_ts,
                'J_igg_mg_per_m2_s': J1_ts,
                'J_moni_mg_per_m2_s': J2_ts,
            })
            # annotate sheet with sigma and marangoni used
            try:
                out_df['sigma_N_per_m'] = float(sigma) if sigma is not None else None
            except Exception:
                out_df['sigma_N_per_m'] = None
            try:
                out_df['Marangoni_Number'] = float(mar) if mar is not None else None
            except Exception:
                out_df['Marangoni_Number'] = None
            sheet_name = sanitize_sheet_name(col)
            out_df.to_excel(writer, sheet_name=sheet_name, index=False)
            any_written = True

    if any_written:
        print('Wrote time series workbook to', out_file)
    else:
        print('No columns processed; no output written.')


if __name__ == '__main__':
    main()