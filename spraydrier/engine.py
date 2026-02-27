# spraydrier/engine.py

from __future__ import annotations

import sys
import os
from pathlib import Path
import inspect
import time
import traceback
import argparse
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import numpy as np
import json

# ─── Robust project root detection + sys.path fix ──────────────────────────────────────────
THIS_FILE = os.path.abspath(__file__)
SPRAYDRIER_DIR = os.path.dirname(THIS_FILE)
PROJECT_ROOT = os.path.dirname(SPRAYDRIER_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    print(f"[DEBUG] Added project root to sys.path: {PROJECT_ROOT}")

print("[DEBUG] sys.path[0] (should be project root):", sys.path[0])

# ────────────────────────────────────────────────
# Core simulation import
# ────────────────────────────────────────────────
try:
    from spraydrier.core.simulation import run_full_spray_drying_simulation
    print("[DEBUG] Successfully imported run_full_spray_drying_simulation")
except ImportError as e:
    print(f"[ERROR] Failed to import simulation: {e}")
    raise

# ────────────────────────────────────────────────
# Output from your updated output.py (relative import from parent)
# ────────────────────────────────────────────────
from output import save_output, preferred_order, label_map

# ────────────────────────────────────────────────
# Configuration & utilities
# ────────────────────────────────────────────────
from config import CalibrationConfig, ModelPaths, load_calibration

# ────────────────────────────────────────────────
# ML models loading
# ────────────────────────────────────────────────
try:
    from ml.ml_models import LoadedModels, load_models
    print("[engine] ML models loader found")
except Exception as e:
    LoadedModels = Any
    load_models = lambda *args, **kwargs: None
    print(f"[engine] Warning: ML models not loaded - {e}")

# ────────────────────────────────────────────────
# Surface composition analyzer
# ────────────────────────────────────────────────
try:
    from surface.integrated_surface_composition_analyzer import (
        IntegratedSurfaceCompositionAnalyzer,
    )
    print("[engine] Surface analyzer found")
except Exception as e:
    IntegratedSurfaceCompositionAnalyzer = lambda *args, **kwargs: None
    print(f"[engine] Warning: Surface analyzer init failed - {e}")

# Progress callback type
ProgressCallback = Callable[[Dict[str, Any]], None]


@dataclass
class TrialInput:
    trial_id: str
    params: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrialResult:
    trial_id: str
    ok: bool = False
    outputs: Dict[str, Any] = field(default_factory=dict)
    histories: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None
    runtime_s: float = 0.0


class SprayDryerEngine:
    def __init__(
        self,
        model_paths: ModelPaths,
        calibration_path: Path | str | None = "calibration.json",
        enable_surface: bool = True,
        enable_ml_load: bool = True,
    ):
        self.model_paths = model_paths
        self.calibration: CalibrationConfig = load_calibration(calibration_path)

        self.models: Optional[LoadedModels] = None
        if enable_ml_load and load_models is not None:
            try:
                self.models = load_models(model_paths)
                print("[engine] ML models loaded successfully")
            except Exception as e:
                self.models = None
                print(f"[engine] Warning: ML models failed to load - {e}")

        self.surface_analyzer = None
        if enable_surface and IntegratedSurfaceCompositionAnalyzer is not None:
            try:
                self.surface_analyzer = IntegratedSurfaceCompositionAnalyzer()
                print("[engine] Surface composition analyzer initialized")
            except Exception as e:
                self.surface_analyzer = None
                print(f"[engine] Warning: Surface analyzer init failed - {e}")

        # Cache simulation signature
        self._sim_sig = inspect.signature(run_full_spray_drying_simulation)
        self._sim_params = list(self._sim_sig.parameters.keys())
        self._sim_takes_single_inputs_dict = (
            len(self._sim_params) == 1 and self._sim_params[0] == "inputs"
        )

        # Synonyms for input normalization
        self._synonyms: Dict[str, str] = {
            "inlet_temp": "T1_C",
            "outlet_temp": "T_outlet_C",
            "feed_rate": "feed_g_min",
            "gas_flow": "m1_m3ph",
            "droplet_d50": "D32_m",
            "solids_frac": "solids_frac",
        }

        self._reserved_top_level = {"formulation", "metadata"}

        # Expanded defaults (reduced spam)
        self._required_defaults: Dict[str, Any] = {
            "dryer": "b290",
            "V_chamber_m3": 0.00212,
            "Span": 1.5,
            "D10_actual": 0.5,
            "D50_actual": 3.0,
            "D90_actual": 6.0,
            "viscosity": "n",
            "surface_tension": "n",
            "cyclone_type": "standard",
            "gas1": "air",
            "gas2": "air",
            "atom_pressure": 1.0,
            "nozzle_tip_d_mm": 0.5,
            "nozzle_cap_d_mm": 1.0,
            "nozzle_level": "middle",
            "ds": "default_drug",
            "ds_conc": 5.0,
            "ds_mw": 150000.0,
            "D_solute": 1e-9,
            "moni_conc": 0.0,
            "pH": 7.0,
            "buffer": "none",
            "buffer_conc": 0.0,
            "stabilizer_A": "none",
            "stab_A_conc": 0.0,
            "additive_B": "none",
            "additive_B_conc": 0.0,
            "additive_C": "none",
            "additive_C_conc": 0.0,
            "feed": "n",
            "feed_mL_min": 5.0,
            "rho_l": 1.0,
            "moisture_input": "n",
            "moisture_content": 0.05,
            "T_ambient": 25.0,
            "calibration_factor": None,
            "observed_lab_moisture": None,
            "solids_frac": 0.1,
            "T1_C": 150.0,
            "RH1": 5.0,
            "m1_m3ph": 30.0,
        }

        # Synced with your simulation.py tuple
        self._sim_param_names = [
            'batch_id', 'dryer', 'V_chamber_m3', 'cyclone_type', 'cyclone_factor', 'gas1', 'T1_C', 'RH1', 'm1_m3ph',
            'gas2', 'T2_C', 'RH2', 'atom_pressure', 'nozzle_tip_d_mm', 'nozzle_cap_d_mm', 'nozzle_level', 'T_outlet_C',
            'ds', 'ds_conc', 'ds_mw', 'viscosity', 'surface_tension', 'D_solute', 'solids_frac', 'moni_conc', 'pH',
            'buffer', 'buffer_conc', 'stabilizer_A', 'stab_A_conc', 'additive_B', 'additive_B_conc', 'additive_C',
            'additive_C_conc', 'feed_g_min', 'rho_l', 'moisture_content', 'D10_actual', 'D50_actual', 'D90_actual',
            'Span', 'ratio', 'viscosity_moni', 'surface_tension_moni', 'R_d', 'R_ag', 'gamma', 'h_vap', 'P_exit',
            'atom_pressure_pa', 'feed_rate_kg_min', 'Q_loss', 'Cd', 'T_ambient_K', 'drying_gas_props', 'atom_gas_props',
            'mu_g_atom', 'Psat_ambient', 'p_v_ambient', 'p_d_ambient', 'X_w', 'p_v', 'p_d', 'rho_final', 'C_p_dry_drying',
            'C_p_humid_drying', 'Psat_initial_ag', 'p_v_atm_in', 'p_d_atm_in', 'X_w_atom', 'C_p_humid_atom', 'rho_atom_in',
            'A_throat', 'pressure_ratio', 'crit_ratio', 'choked_condition', 'atom_gas_mass_flow', 'T_throat', 'c_throat',
            'term', 'M_exit', 'T_exit', 'c_exit', 'u_ag', 'm_dry_kg_s', 'T_outlet_ag_adiabatic_K', 'T_mixed_K',
            'T_outlet_ag_K', 'p_v_atm_exit', 'p_d_atm_exit', 'rho_ag_exit', 'mu_g_exit', 'phi', 'initial_moisture_content',
            'evap_water_kgph', 'RH_out', 'Pv_out', 'Psat_out', 'condensed_kg_s', 'actual_water_evap', 'calibration_factor',
            'observed_lab_frac', 'moisture_predicted', 'measured_RH', 'max_iter', 'rh_tol', 'moist_tol', 'curr_RH',
            'curr_moist', 'evap_water_kgph', 'condensed_bulk_kg_s', 'it', 'evap_kgph', 'new_RH', 'Pv_out_local',
            'Psat_out_local', 'condensed_kg_s', 'new_moist', 'RH_out', 'Pv_out', 'Psat_out', 'moisture_predicted',
            'actual_water_evap', 'We', 'Oh', 'D32_without_moni', 'D32_with_moni', 'radius_history_um', 't_eval',
            'pe_metrics', 'energy_balance', 'efficiency', 'required_inlet_temp', 'sigma_effective', 'Nu', 'Sh', 'h',
            'k_m', 'condensed_bulk_kg_s', 'condensed_surface_kg_s', 'morphology_indicators', 'Tg_array', 'temperature_array',
            'shell_formation_time', 'enhanced_tg_results'
        ]

    def _normalize_inputs_dict(
        self,
        raw: Dict[str, Any],
        warnings_out: List[str],
        trial_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        ignored: List[str] = []

        for k, v in raw.items():
            if k in self._reserved_top_level:
                continue
            k2 = self._synonyms.get(k, k)
            if pd.isna(v) or v == "":
                continue
            out[k2] = v

        for k in raw:
            if k in self._reserved_top_level:
                ignored.append(k)
        if ignored:
            warnings_out.append(f"Ignored (handled separately): {ignored}")

        for k, v in self._required_defaults.items():
            if k not in out or pd.isna(out[k]) or out[k] == "":
                out[k] = v
                warnings_out.append(f"Used default for {k}: {v}")

        # Fix batch_id
        batch_id_val = out.get("batch_id")
        if pd.isna(batch_id_val) or batch_id_val is None or str(batch_id_val).strip() == "":
            out["batch_id"] = trial_id or f"batch_{time.strftime('%Y%m%d_%H%M%S')}"
            warnings_out.append("batch_id was missing/NaN → assigned default")

        return out

    def _call_simulation(self, inputs_dict: Dict[str, Any]) -> Any:
        if self._sim_takes_single_inputs_dict:
            return run_full_spray_drying_simulation(inputs_dict)
        return run_full_spray_drying_simulation(**inputs_dict)

    def run_trial(self, trial: TrialInput) -> TrialResult:
        t0 = time.time()
        res = TrialResult(trial_id=trial.trial_id, ok=False)

        try:
            raw_params = dict(trial.params)
            sim_inputs = self._normalize_inputs_dict(
                raw_params, res.warnings, trial_id=trial.trial_id
            )

            sim_out = self._call_simulation(sim_inputs)
            print(f"[DEBUG] sim_out type: {type(sim_out)}")

            if isinstance(sim_out, tuple):
                expected_len = len(self._sim_param_names)
                actual_len = len(sim_out)
                if actual_len != expected_len:
                    print(f"[WARNING] Tuple length mismatch: got {actual_len}, expected {expected_len}")
                sim_out = dict(zip(self._sim_param_names[:actual_len], sim_out))

            outputs: Dict[str, Any] = {"simulation": sim_out}

            # Apply calibration factors (if loaded)
            if hasattr(self.calibration, 'moisture_factor'):
                sim_out['moisture_predicted'] = sim_out.get('moisture_predicted', 0.05) * self.calibration.moisture_factor
            if hasattr(self.calibration, 'rh_factor'):
                sim_out['RH_out'] = sim_out.get('RH_out', 0.0) * self.calibration.rh_factor
            if hasattr(self.calibration, 'outlet_temp_factor'):
                sim_out['T_outlet_C'] = sim_out.get('T_outlet_C', 80.0) * self.calibration.outlet_temp_factor
            if hasattr(self.calibration, 'd50_factor'):
                sim_out['D50_calc'] = sim_out.get('D50_calc', 3.0) * self.calibration.d50_factor

            # Surface analysis (fixed: no 'sim_out' kwarg)
            if self.surface_analyzer is not None:
                try:
                    surface_params = dict(raw_params)
                    surface_params.update(sim_inputs)
                    surface_out = self.surface_analyzer.run_simulation_for_trial(surface_params)
                    print(f"[DEBUG] Surface out type: {type(surface_out)}")
                    if isinstance(surface_out, dict):
                        outputs.update(surface_out)
                    else:
                        outputs["surface"] = surface_out
                except Exception as e:
                    res.warnings.append(f"Surface analysis failed: {e}")

            # ML morphology prediction (safe check)
            if self.models is not None:
                try:
                    if hasattr(self.models.morphology, 'predict'):
                        features = {k: v for k, v in sim_out.items() if k in ['Pe', 'Ma', 'T_outlet_C', 'RH_out', 'solids_frac']}
                        morphology = self.models.morphology.predict([list(features.values())])[0]
                        outputs['morphology_predicted'] = morphology
                        outputs['morphology_mechanism'] = 'Based on Pe > 10 and Darcy pressure'
                    else:
                        raise AttributeError("Loaded morphology model has no .predict method")
                except Exception as e:
                    res.warnings.append(f"ML prediction failed: {e}")

            res.outputs = outputs
            res.ok = True

        except KeyError as e:
            missing = str(e).strip("'")
            res.error = f"Missing required input: {missing}"
            res.outputs = {"partial_inputs": sim_inputs, "missing_key": missing}
            res.warnings.append(f"KeyError caught: {missing}")
            res.ok = False
        except Exception as e:
            res.error = str(e)
            res.outputs = {"error_details": str(e), "traceback": traceback.format_exc()}
            res.warnings.append(f"Exception in trial: {e}")
            res.ok = False

        res.runtime_s = time.time() - t0
        return res

    def run_batch(
        self,
        trials: List[TrialInput],
        progress: Optional[ProgressCallback] = None,
    ) -> List[TrialResult]:
        results: List[TrialResult] = []
        total = len(trials)

        for i, t in enumerate(trials, start=1):
            if progress:
                progress({"phase": "running", "trial_id": t.trial_id, "index": i, "total": total})

            print(f"[Progress] Running trial {i}/{total}: {t.trial_id}")
            r = self.run_trial(t)
            results.append(r)

            if progress:
                progress({
                    "phase": "done",
                    "trial_id": t.trial_id,
                    "ok": r.ok,
                    "index": i,
                    "total": total,
                })

        return results

    def run_from_excel(self, excel_path: str, sheet_name: str = "Sheet1", progress: Optional[ProgressCallback] = None) -> List[TrialResult]:
        try:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)

            # Detect format
            if 'trial_id' in df.columns.str.lower():
                print("[engine] Detected new results format (trials as rows)")
                df = df.set_index('trial_id').T
            else:
                # Old format: params as rows, batches as columns
                print("[engine] Detected old format (parameters as rows)")
                params = df.iloc[1:, 0].tolist()
                batch_ids_raw = df.iloc[0, 1:].tolist()
                data = df.iloc[1:, 1:].T
                data.columns = params
                batch_ids = [str(val) if pd.notna(val) else f"batch_{i+1}" for i, val in enumerate(batch_ids_raw)]
                data['batch_id'] = batch_ids
                df = data

            print(f"[engine] Loaded {len(df.columns)} trials from {excel_path}")
        except Exception as e:
            print(f"[engine] Failed to load {excel_path}: {e}")
            return []

        trials = []
        for idx, row in df.iterrows():
            params = row.to_dict()
            batch_id_raw = params.get('batch_id')
            trial_id = str(batch_id_raw).strip() if pd.notna(batch_id_raw) and batch_id_raw else f"batch_{idx + 1}"
            trial = TrialInput(trial_id=trial_id, params=params)
            trials.append(trial)

        results = self.run_batch(trials, progress)
        self._export_results_to_excel(results, excel_path)

        return results

    def _export_results_to_excel(self, results: List[TrialResult], original_path: str):
        output_path = original_path.replace('.xlsx', '_results.xlsx')

        # Use your preferred_order for row alignment
        for r in results:
            if not r.ok or not isinstance(r.outputs.get('simulation'), dict):
                continue

            sim_out = r.outputs['simulation']
            flat_result = {
                'batch_id': r.trial_id,
                **sim_out,
                **r.outputs.get('surface', {}),
                'Predicted Morphology': r.outputs.get('morphology_predicted', 'N/A'),
                'Morphology Confidence': r.outputs.get('morphology_confidence', 0.0),
            }

            save_output(
                flat_result,
                inputs={},  # or pass original sim_inputs if you want
                filename=output_path,
                input_param_order=preferred_order
            )

        print(f"[engine] Exported aligned results to {output_path}")
        print(f"[engine] Processed {len(results)} trials")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spray Dryer Engine - Training & Prediction")
    parser.add_argument("--mode", choices=["train", "predict"], default=None,
                        help="Mode: 'train' for calibration on historical data, 'predict' for prototypes")
    parser.add_argument("--input", type=str, default=None,
                        help="Path to input Excel file")
    args = parser.parse_args()

    # Interactive prompt if no arguments
    if args.mode is None or args.input is None:
        print("\n" + "="*60)
        print("Spray Dryer Engine Mode Selection")
        print("1 - Training / Calibration (full historical dataset)")
        print("2 - Prediction (prototype formulations)")
        print("="*60)
        choice = input("Enter 1 or 2: ").strip()

        if choice == "1":
            args.mode = "train"
        elif choice == "2":
            args.mode = "predict"
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)

        if args.input is None:
            if args.mode == "train":
                default_file = r"C:\Users\dunca\Desktop\python\snapshot_mab_input (1).xlsx"
            else:
                default_file = r"C:\Users\dunca\Desktop\python\data\BHV1400_input_template.xlsx"
            args.input = input(f"Enter input Excel file path (default: {default_file}): ").strip() or default_file

    print(f"\n[MAIN] Running in {args.mode.upper()} mode with file: {args.input}")

    # Use correct path to ml/ folder
    ml_base = Path("spraydrier/ml")
    model_paths = ModelPaths(
        d50=ml_base / "d50_model.pkl",
        moisture=ml_base / "moisture_model.pkl",
        morphology=ml_base / "morphology_model.pkl",
        outlet_temp=ml_base / "outlet_temp_model.pkl",
        rh_outlet=ml_base / "rh_outlet_model.pkl"
    )

    missing = []
    model_attrs = ['d50', 'moisture', 'morphology', 'outlet_temp', 'rh_outlet']
    for attr in model_attrs:
        p = getattr(model_paths, attr, None)
        if p and not p.exists():
            missing.append(str(p))
    if missing:
        print(f"[WARNING] Missing ML models: {', '.join(missing)} → ML {'skipped' if args.mode == 'predict' else 'not used in train mode'}")

    # In training mode we don't need ML loaded
    enable_ml = (args.mode == "predict" and not bool(missing))
    engine = SprayDryerEngine(
        model_paths=model_paths,
        enable_surface=True,
        enable_ml_load=enable_ml
    )

    path = Path(args.input)
    if path.exists():
        results = engine.run_from_excel(str(path))
        print(f"[MAIN] Completed {len(results)} trials")
    else:
        print(f"[MAIN] File not found → {path}")
        sys.exit(1)

    print("\nWorkflow finished. Check *_results.xlsx files in the data folder.")

    if args.mode == "train":
        print("\n" + "="*60)
        print("TRAINING MODE COMPLETE")
        print("Next step: Train/update ML models using the generated results:")
        print(f"   python spraydrier/ml/learn_calibration.py")
        print("   (make sure it reads the *_results.xlsx file just created)")
        print("="*60)

    if args.mode == "predict" and engine.models is None:
        print("\nNote: ML predictions were skipped (models missing or not loaded).")
        print("     Run training mode first on snapshot data to generate models.")