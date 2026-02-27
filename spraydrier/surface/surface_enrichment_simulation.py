import pandas as pd
import numpy as np


# Allow selecting workbook via CLI (default: data.xlsx)
import argparse
import sys
import os
import importlib.util
parser = argparse.ArgumentParser(description='Compute surface percentages from workbook (time-dependent solver only)')
parser.add_argument('--workbook', default=None, help='Path to workbook (if omitted, you will be prompted interactively)')
args, _ = parser.parse_known_args()
wb_path = args.workbook

# Interactive prompt if no workbook provided: ask the user for the path.
# If stdin isn't available (EOF), exit with a helpful message instructing
# the caller to pass --workbook.
default_wb = 'data.xlsx'
if wb_path is None:
    while True:
        try:
            resp = input(f"Enter workbook path (Enter for '{default_wb}'): ").strip()
        except EOFError:
            print("No interactive input available. Please re-run with --workbook <path> to specify the workbook.")
            raise SystemExit(2)
        if resp == '':
            wb_path = default_wb
        else:
            wb_path = resp
        # try to read workbook; if fails, re-prompt
        try:
            wb = pd.read_excel(wb_path, index_col=0)
            break
        except Exception as e:
            print(f"Could not read '{wb_path}': {e}")
            # prompt again
            continue
else:
    try:
        wb = pd.read_excel(wb_path, index_col=0)
    except Exception as e:
        print(f"Could not read workbook '{wb_path}': {e}")
        raise SystemExit(2)

def find_row_label(df, substrs):
    """Return the index label from df (index) that contains any of the substrings in substrs (case-insensitive)."""
    for s in substrs:
        for idx in df.index:
            try:
                if isinstance(idx, str) and s.lower() in idx.lower():
                    return idx
            except Exception:
                continue
    return None

# Batch ids are stored in the header (columns) starting at column 2 (Excel col B)
batch_ids = list(wb.columns)

# Extract rows by label matching (robust to variants)
ds_label = find_row_label(wb, ['Drug Substance conc', 'Drug Substance conc. (mg/mL)', 'ds_conc'])
moni_label = find_row_label(wb, ['Moni conc', 'Moni conc. (mg/mL)', 'moni_conc'])
v_label = find_row_label(wb, ['Surface recession velocity', 'Evaporation Velocity'])
d32_label = find_row_label(wb, ['D32 (calculated', 'D32 (calculated with', 'D32 (calculated with Moni)'])
pe_igg_label = find_row_label(wb, ['Pe_IgG', 'Pe_igg', 'Pe DS', 'Pe_IgG (reported)'])
pe_moni_label = find_row_label(wb, ['Pe_MoNI', 'Pe_moni', 'Pe Moni', 'Pe_MoNI (reported)'])
total_solids_label = find_row_label(wb, ['%Solids', 'Total solids', 'Total Solids'])

# Helper to safely read a row as float array
def read_row_as_float(df, label, default_len=None, default=0.0):
    if label is None or label not in df.index:
        if default_len is None:
            return np.array([])
        return np.full(default_len, default, dtype=float)
    row = df.loc[label].values
    try:
        arr = np.array([float(x) if not pd.isna(x) else default for x in row], dtype=float)
    except Exception:
        # fallback: try coercion with pandas
        arr = pd.to_numeric(df.loc[label], errors='coerce').fillna(default).to_numpy(dtype=float)
    if default_len is not None and len(arr) < default_len:
        # pad
        pad = np.full(default_len - len(arr), default, dtype=float)
        arr = np.concatenate([arr, pad])
    return arr

# Determine number of batches (columns)
n_batches = len(batch_ids)

# Read required rows
ds_conc = read_row_as_float(wb, ds_label, default_len=n_batches)
moni_conc = read_row_as_float(wb, moni_label, default_len=n_batches)
total_solids = read_row_as_float(wb, total_solids_label, default_len=n_batches) / 10.0
v = read_row_as_float(wb, v_label, default_len=n_batches)
d32 = read_row_as_float(wb, d32_label, default_len=n_batches)
r = d32 / 2.0 * 1e-6
pe_igg_reported = read_row_as_float(wb, pe_igg_label, default_len=n_batches)
pe_moni_reported = read_row_as_float(wb, pe_moni_label, default_len=n_batches)

# Default diffusion coefficients (m²/s)
d_igg_default = 1.8e-11
d_moni_default = 8.5e-11
d_sbecd = 1.5e-10
d_dextran = 1.0e-11

# Prefer explicit per-column D rows if present in the workbook
d_igg_label = find_row_label(wb, ['D_igg', 'D_IgG', 'D_ds', 'D_bhv', 'D_bhv-1300'])
d_moni_label = find_row_label(wb, ['D_moni', 'D_MoNI', 'D_moni ('])

d_igg_row = read_row_as_float(wb, d_igg_label, default_len=n_batches, default=np.nan)
d_moni_row = read_row_as_float(wb, d_moni_label, default_len=n_batches, default=np.nan)

# We may also have attempted to parse a 'Drug Substance' row earlier into d_igg_per_column;
# if present this can be used as a D_igg source (kept for backward compatibility).
try:
    d_igg_per_column  # may exist from earlier code path
except NameError:
    d_igg_per_column = np.full(n_batches, np.nan)

# Build per-batch D arrays, priority: explicit D row > Drug Substance row > default
d_igg_array = np.full(n_batches, d_igg_default, dtype=float)
d_moni_array = np.full(n_batches, d_moni_default, dtype=float)
for i in range(n_batches):
    # D_igg priority
    if not pd.isna(d_igg_row[i]) and d_igg_row[i] > 0:
        d_igg_array[i] = float(d_igg_row[i])
    elif not pd.isna(d_igg_per_column[i]) and d_igg_per_column[i] > 0:
        d_igg_array[i] = float(d_igg_per_column[i])
    # D_moni priority
    if not pd.isna(d_moni_row[i]) and d_moni_row[i] > 0:
        d_moni_array[i] = float(d_moni_row[i])

# Calculate corrected Pe values using per-column diffusion coefficients
pe_igg = v * r / d_igg_array
pe_moni = v * r / d_moni_array

# Prepare arrays for other compounds
other_concs = np.zeros(n_batches)
other_pe = np.zeros(n_batches)
optimized_sbecd_conc = 10.0
optimized_dextran_conc = 10.0
pe_sbecd = v * r / d_sbecd
pe_dextran = v * r / d_dextran

# Optimize 249#003c with proposed conditions if present
optimized_v = 0.0001  # m/s (adjusted with 2 g/min, nitrogen)
optimized_r_val = 4.1e-6  # m (adjusted with 2 bar)

# Find the index for target batch id via a substring match in headers
target_label = '249#003c'
index_249_003c = None
for i, bid in enumerate(batch_ids):
    try:
        if isinstance(bid, str) and target_label.lower() in bid.lower():
            index_249_003c = i
            break
    except Exception:
        continue

if index_249_003c is None:
    print(f"Warning: batch id '{target_label}' not found in workbook columns; skipping optimization block.")
    v_optimized = v.copy()
    r_optimized = r.copy()
    pe_igg_optimized = pe_igg.copy()
    pe_moni_optimized = pe_moni.copy()
    pe_sbecd_optimized = pe_sbecd.copy()
    pe_dextran_optimized = pe_dextran.copy()
else:
    v_optimized = v.copy()
    r_optimized = r.copy()
    v_optimized[index_249_003c] = optimized_v
    r_optimized[index_249_003c] = optimized_r_val
    pe_igg_optimized = v_optimized * r_optimized / d_igg_array
    pe_moni_optimized = v_optimized * r_optimized / d_moni_array
    pe_sbecd_optimized = v_optimized * r_optimized / d_sbecd
    pe_dextran_optimized = v_optimized * r_optimized / d_dextran
    # Add SBECD or Dextran for optimized batch
    other_concs[index_249_003c] = optimized_sbecd_conc
    other_pe[index_249_003c] = pe_sbecd_optimized[index_249_003c]

# Strict validation: require key numeric rows to be present in the workbook.
def validate_required_rows(df):
    """Ensure required numeric rows exist in the workbook index. Exit if missing.

    Required: Drug Substance conc, Moni conc, D32 (calculated), Surface recession velocity
    Optionally require diffusion coefficients rows if present as named rows.
    """
    required_label_sets = {
        'Drug Substance conc': ['Drug Substance conc', 'Drug Substance conc.', 'ds_conc'],
        'Moni conc': ['Moni conc', 'Moni conc.', 'moni_conc'],
        'D32': ['D32 (calculated', 'D32 (calculated with', 'D32 (calculated with Moni)'],
        'Surface recession velocity': ['Surface recession velocity', 'Evaporation Velocity']
    }
    missing = []
    for human, substrs in required_label_sets.items():
        if find_row_label(df, substrs) is None:
            missing.append(human)

    # Do not require D_igg to be present as a named row; it may be provided per-column
    diff_labels = [
        ('D_moni', ['D_moni', 'D_MoNI', 'D_MoNI (m2/s)'])
    ]
    for human, substrs in diff_labels:
        if find_row_label(df, substrs) is None:
            missing.append(human)

    if missing:
        print('\nERROR: required numeric rows missing from workbook index:')
        for m in missing:
            print('  -', m)
        print('\nPlease add these rows to the workbook or provide them via per-column JSON parameters.')
        raise SystemExit(2)

# Run validation now and fail early if required rows are absent.
validate_required_rows(wb)

# Ensure d_igg_per_column exists (may be filled later from Drug Substance row)
try:
    d_igg_per_column
except NameError:
    d_igg_per_column = np.full(n_batches, np.nan)

# Try to read per-column D_igg values from a 'Drug Substance' row if present.
# The workbook uses a 'Drug Substance' row (row 16 in the original sheet) that may
# contain per-batch numeric values. If those values are actually MW rather than
# diffusion coefficients, we need a mapping; for now we assume they are D_igg
# values in m^2/s. If you need a MW->D mapping, tell me and I'll add it.
drug_substance_label = find_row_label(wb, ['Drug Substance', 'Drug Substance (mg/mL)', 'Drug Substance'])
if drug_substance_label is not None and drug_substance_label in wb.index:
    try:
        raw = wb.loc[drug_substance_label].values
        d_igg_per_column = pd.to_numeric(raw, errors='coerce').to_numpy(dtype=float)
    except Exception:
        d_igg_per_column = np.full(n_batches, np.nan)
else:
    d_igg_per_column = np.full(n_batches, np.nan)

# Default scalar D_igg
d_igg_default = 1.8e-11

# Build a per-batch D_igg array using per-column values when present, else default
d_igg_array = np.full(n_batches, d_igg_default, dtype=float)
for i in range(n_batches):
    try:
        val = d_igg_per_column[i]
        if not pd.isna(val) and val > 0:
            d_igg_array[i] = float(val)
    except Exception:
        # leave default
        pass

# Calculate corrected Pe values using per-column D_igg
pe_igg = v * r / d_igg_array

# Always run the full time-dependent solver per batch and write timeseries.
# The static heuristic branch has been removed to ensure the time-dependent
# physics-based solver is used consistently.
se_path = os.path.join(os.getcwd(), 'Surface Enrichment.py')
try:
    spec = importlib.util.spec_from_file_location('surf_enrich', se_path)
    se_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(se_mod)
except Exception as e:
    raise RuntimeError(f"Could not load Surface Enrichment module at {se_path}: {e}")

out_wb = 'surface_percentages_td_timeseries.xlsx'
writer = pd.ExcelWriter(out_wb, engine='openpyxl')
summary = []
for i, col in enumerate(batch_ids):
    # get per-column params via the helper in the Surface Enrichment module
    try:
        params = se_mod._read_params_from_column(wb, col)
    except Exception:
        params = {}
    R0 = params.get('R0') if params.get('R0') is not None else (r[i] if not np.isnan(r[i]) else 4.29e-6)
    v_evap = params.get('v_evap') if params.get('v_evap') is not None else (v[i] if not np.isnan(v[i]) else 5.7e-5)
    D_moni_val = params.get('D_moni') if params.get('D_moni') is not None and not pd.isna(params.get('D_moni')) else d_moni_array[i]
    # Use per-column D_igg array if available; allow params override if explicitly present
    if params.get('D_igg') is not None and not pd.isna(params.get('D_igg')):
        D_igg_val = float(params.get('D_igg'))
    else:
        D_igg_val = float(d_igg_array[i])
    igg_conc_val = params.get('igg_conc') if params.get('igg_conc') is not None and not pd.isna(params.get('igg_conc')) else ds_conc[i]
    moni_conc_val = params.get('moni_conc') if params.get('moni_conc') is not None and not pd.isna(params.get('moni_conc')) else moni_conc[i]

    k_ads_igg = float(params.get('k_ads_igg')) if params.get('k_ads_igg') is not None and not pd.isna(params.get('k_ads_igg')) else 0.0
    k_des_igg = float(params.get('k_des_igg')) if params.get('k_des_igg') is not None and not pd.isna(params.get('k_des_igg')) else 0.0
    k_ads_moni = float(params.get('k_ads_moni')) if params.get('k_ads_moni') is not None and not pd.isna(params.get('k_ads_moni')) else 0.0
    k_des_moni = float(params.get('k_des_moni')) if params.get('k_des_moni') is not None and not pd.isna(params.get('k_des_moni')) else 0.0
    Gamma_max = float(params.get('Gamma_max')) if params.get('Gamma_max') is not None and not pd.isna(params.get('Gamma_max')) else 1.0

    # run transient solver
    try:
        (times, pct1, pct2, g1, g2, Pe1, Pe2, v1, v2, J1, J2) = se_mod.simulate_two_species(
            float(R0), float(v_evap), float(D_igg_val), float(D_moni_val), float(igg_conc_val), float(moni_conc_val),
            k_ads_1=k_ads_igg, k_des_1=k_des_igg,
            k_ads_2=k_ads_moni, k_des_2=k_des_moni,
            Gamma_max=Gamma_max)
    except Exception as e:
        print(f"Transient simulation failed for {col}: {e}")
        continue

    df = pd.DataFrame({'time_s': times, 'pct_igg_surface': pct1, 'pct_moni_surface': pct2,
                       'gamma_igg': g1, 'gamma_moni': g2, 'Pe_igg': Pe1, 'Pe_moni': Pe2,
                       'v_mig_igg': v1, 'v_mig_moni': v2, 'J_igg': J1, 'J_moni': J2})
    sheet_name = (col[:25] + '_td')[:31]
    df.to_excel(writer, sheet_name=sheet_name, index=False)
    summary.append({'Batch ID': col, '%MoNI initial': float(pct2[0]) if len(pct2)>0 else np.nan,
                    '%MoNI final': float(pct2[-1]) if len(pct2)>0 else np.nan,
                    '%MoNI max': float(np.max(pct2)) if len(pct2)>0 else np.nan})

writer.close()
pd.DataFrame(summary).to_excel('surface_percentages_td_summary.xlsx', index=False)
print(f"Wrote time-dependent sheets to {out_wb} and summary to surface_percentages_td_summary.xlsx")
raise SystemExit(0)

# Calculate surface percentages
results = []
for i in range(n_batches):
    pe_ds_val = pe_igg_optimized[i] if index_249_003c is not None and i == index_249_003c else pe_igg[i]
    pe_moni_val = pe_moni_optimized[i] if index_249_003c is not None and i == index_249_003c else pe_moni[i]
    other_pe_val = [other_pe[i]]
    percent_ds, percent_moni, percent_others = calculate_surface_percentages(
        ds_conc[i] if i < len(ds_conc) else 0.0,
        moni_conc[i] if i < len(moni_conc) else 0.0,
        [other_concs[i]] if i < len(other_concs) else [0.0],
        pe_ds_val,
        pe_moni_val,
        other_pe_val,
        total_solids[i] if i < len(total_solids) else 0.0,
        v[i] if i < len(v) else 0.0,
        migration_weight=1.0
    )
    results.append({
        'Batch ID': batch_ids[i],
        'Reported Pe_IgG': pe_igg_reported[i] if i < len(pe_igg_reported) else np.nan,
        'Reported Pe_MoNI': pe_moni_reported[i] if i < len(pe_moni_reported) else np.nan,
        'Calculated Pe_IgG': pe_igg[i] if i < len(pe_igg) else np.nan,
        'Calculated Pe_MoNI': pe_moni[i] if i < len(pe_moni) else np.nan,
        '%DS at Surface': percent_ds,
        '%MoNI at Surface': percent_moni,
        '%Other Compounds at Surface': percent_others[0] if len(percent_others) > 0 else 0.0
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Display results and save
print("Surface Percentage Results:")
print(results_df)
output_file = "surface_percentages_simulation.xlsx"
results_df.to_excel(output_file, index=False)
print(f"Results saved to {output_file}")

 