import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

parser = argparse.ArgumentParser(description='Generate PNG plots from time-dependent surface percentage workbook')
parser.add_argument('--workbook', default=None, help='TD timeseries workbook (if omitted, you will be prompted interactively)')
parser.add_argument('--batches', nargs='*', help='Batch IDs to plot (substring match). If omitted, plot all sheets.')
parser.add_argument('--outdir', default='td_plots', help='Output directory for PNG files')
args = parser.parse_args()

wb_path = args.workbook
if wb_path is None:
    default = 'surface_percentages_td_timeseries.xlsx'
    try:
        resp = input(f"Enter TD workbook path (Enter for '{default}'): ").strip()
    except EOFError:
        print('No interactive input available; please re-run with --workbook <path>')
        raise SystemExit(2)
    wb_path = resp if resp != '' else default
outdir = args.outdir
os.makedirs(outdir, exist_ok=True)

# Load workbook sheets
try:
    xls = pd.ExcelFile(wb_path)
except Exception as e:
    print(f"Could not open '{wb_path}': {e}")
    raise SystemExit(2)

sheets = xls.sheet_names

# Helper to determine if a sheet should be plotted
def sheet_matches(sheet_name, batch_filters):
    if not batch_filters:
        return True
    for f in batch_filters:
        if f.lower() in sheet_name.lower():
            return True
    return False

for sheet in sheets:
    if not sheet_matches(sheet, args.batches):
        continue
    df = pd.read_excel(xls, sheet_name=sheet)
    # Expect columns: time_s, pct_igg_surface, pct_moni_surface
    if 'time_s' not in df.columns:
        print(f"Skipping sheet {sheet}: no time_s column")
        continue
    t = df['time_s']
    igg_col = None
    moni_col = None
    for c in df.columns:
        if 'pct_igg' in c.lower():
            igg_col = c
        if 'pct_moni' in c.lower() or 'pct_moni' in c.lower():
            moni_col = c
    plt.figure(figsize=(8,4))
    if igg_col is not None:
        plt.plot(t, df[igg_col], label='%DS (IgG)')
    if moni_col is not None:
        plt.plot(t, df[moni_col], label='%MoNI')
    plt.xlabel('Time (s)')
    plt.ylabel('Surface %')
    plt.title(sheet)
    plt.legend()
    fname = os.path.join(outdir, f"{sheet[:50].replace(' ','_')}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=200)
    plt.close()
    print(f"Wrote {fname}")

print('Done')
