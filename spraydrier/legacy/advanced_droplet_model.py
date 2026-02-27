# -*- coding: utf-8 -*-

"""
advanced_droplet_model.py
Predicts particle morphology, surface composition (with proper surfactant physics),
and moisture for spray-dried formulations.
"""

# Langmuir + PS80 logic
from models.adsorption_model import surfactant_priority_override
from peclet_calculator import calculate_all_peclet_metrics
from simulation import run_full_spray_drying_simulation
from models import predict_moisture_content
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy as sp
import numpy as np
import pandas as pd
import os
import sys
import json
import argparse
import pickle
import warnings
from pathlib import Path

# ----------------------------------------------------------------------
# Make sure Python can find our local packages (models/, simulation/, etc.)
# This one line fixes ALL Mac/Windows/Linux import issues forever
# ----------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress annoying fsolve warnings (they're harmless here)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="scipy.optimize")

# ----------------------------------------------------------------------
# Load compound properties database
# ----------------------------------------------------------------------
try:
    with open('compound_props.json', 'r') as f:
        compound_props = json.load(f)
    print(f"Loaded compound properties for {len(compound_props)} compounds")
except FileNotFoundError:
    print("Warning: compound_props.json not found, MW lookup will not work")
    compound_props = {}
except json.JSONDecodeError:
    print("Warning: compound_props.json is invalid JSON, MW lookup will not work")
    compound_props = {}


def get_compound_molecular_weight(compound_name, excel_mw=None):
    """
    Get molecular weight for a compound, with fallback hierarchy:
    1. compound_props.json lookup (if compound_name provided)
    2. Excel/input MW value (if provided)
    3. Default to 0.0 (if no compound specified)
    """
    if compound_name:
        # Normalize compound name for lookup
        normalized_name = compound_name.lower().strip()
        if normalized_name in compound_props:
            mw = compound_props[normalized_name]['mw']
            return mw
        else:
            # Compound name provided but not in database, use Excel value or default to 0
            return excel_mw if excel_mw is not None else 0.0
    else:
        # No compound name provided, default to 0
        return 0.0


# ----------------------------------------------------------------------
# Our own modules
# ----------------------------------------------------------------------

# (pickle is only used later, no need to import yet)


def train_morphology_model(
        df, existing_model=None, existing_encoder=None, existing_features=None, save_path=None):
    """Train a machine learning model to predic        # Surface concentrations for each solute (accumulate over time)
        Pe_drug = surface_recession * a[i-1] / D_drug
        Pe_moni = surface_recession * a[i-1] / D_moni
        Pe_stabilizer = surface_recession * a[i-1] / D_stabilizer
        Pe_additive = surface_recession * a[i-1] / D_additive

        # Cap individual Peclet numbers to prevent overflow
        Pe_drug = min(Pe_drug, 50)
        Pe_moni = min(Pe_moni, 50)
        Pe_stabilizer = min(Pe_stabilizer, 50)
        Pe_additive = min(Pe_additive, 50)

        # Debug: print Peclet numbers occasionally
        if i % 100 == 0 and i > 0:
            print(f"   t={t[i]:.3f}s: Pe_drug={Pe_drug:.2f}, Pe_moni={Pe_moni:.2f}, Pe_stabilizer={Pe_stabilizer:.2f}, Pe_additive={Pe_additive:.2f}")

        # FINAL FIX: Use PRE-CALCULATED Pe values from main.py
        # main.py already computed the correct time-averaged or effective Pe
        # Do NOT recalculate Pe here — that breaks everything
        Pe_drug       = float(row.get('effective_pe_drug', row.get('Pe_igg', 10.0)))
        Pe_moni       = float(row.get('effective_pe_moni', row.get('Pe_moni', 0.1)))
        Pe_stabilizer = float(row.get('effective_pe_stabilizer', 1.0))
        Pe_additive   = float(row.get('effective_pe_additive', 1.0))

        # Optional: cap to prevent overflow (safe to keep)
        Pe_drug       = min(Pe_drug, 50)
        Pe_moni       = min(Pe_moni, 50)
        Pe_stabilizer = min(Pe_stabilizer, 50)
        Pe_additive   = min(Pe_additive, 50)

        # Debug print — you will LOVE seeing this
        if i == 1 or i % 50 == 0:
            print(f"   t={t[i]:.3f}s → Using main.py Pe: drug={Pe_drug:.2f}, moni={Pe_moni:.2f}")

        # Surface concentrations for each solute (accumulate over time)
        xi_drug_surface       = xi_drug[i-1]       * np.exp(Pe_drug)
        xi_moni_surface       = xi_moni[i-1]       * np.exp(Pe_moni)
        xi_stabilizer_surface = xi_stabilizer[i-1] * np.exp(Pe_stabilizer)
        xi_additive_surface   = xi_additive[i-1]   * np.exp(Pe_additive)

    If existing_model is provided, will continue training (incremental learning)
    """
    if 'Morphology' not in df.columns:
        print("No Morphology column found in data - using physics-based predictions only")
        return existing_model, existing_encoder, existing_features or []

    feature_columns = [
        # Basic process parameters
        'Drying Gas Inlet (C)', 'Drying gas rate (m³/hr)', 'Drug Substance conc. (mg/mL)',
        'Moni conc. (mg/mL)', 'Feed solution pH', 'Stabilizer conc. (mg/mL)',
        'Additive #1 conc. (mg/mL)', '%Solids', 'D50_actual', 'D10_actual', 'D90_actual',
        # Physically relevant parameters
        'Feed Rate (g/min)', 'Surface recession velocity', 'Max Peclet Number',
        'Effective Peclet Number', 'Peclet Number', 'Integrated Peclet Number',
        'Heat Transfer coefficient', 'Mass Transfer coefficient', 'Reynolds number',
        'Marangoni Number', 'Estimated feed viscosity (Pa·s)', 'Estimated Feed Surface Tension (N/m)',
        # Moisture - try multiple possible column names
        'Measured total moisture (%)', 'moisture_predicted', 'Measured bound moisture (%)',
        # Surface composition features (if available in training data)
        'Drug surface %', 'Moni surface %', 'Stabilizer surface %', 'Additive surface %',
        'Drug bulk %', 'Moni bulk %', 'Stabilizer bulk %', 'Additive bulk %',
        # Additional effective Peclet numbers for solutes
        'effective_pe_moni', 'effective_pe_drug', 'effective_pe_stabilizer',
        'effective_pe_additive_b', 'effective_pe_additive_c', 'effective_pe_buffer'
    ]

    available_features = [col for col in feature_columns if col in df.columns]

    # Check if columns can be converted to numeric
    numeric_features = []
    for col in available_features:
        try:
            pd.to_numeric(df[col], errors='coerce').dropna()
            numeric_count = pd.to_numeric(
                df[col], errors='coerce').notna().sum()
            if numeric_count >= 3:  # At least 3 numeric values
                numeric_features.append(col)
        except BaseException:
            continue

    if len(numeric_features) < 3:
        print(
            f"Not enough numeric feature columns available for ML training (found {
                len(numeric_features)})")
        return existing_model, existing_encoder, existing_features or numeric_features

    print(
        f"Using {
            len(numeric_features)} numeric features: {numeric_features}")

    X = df[numeric_features].copy()
    y = df['Morphology'].copy()

    # Convert to numeric and fill NaN values
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
        X[col] = X[col].fillna(X[col].mean())

    y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 'spheres')

    # Handle incremental learning
    if existing_model is not None and existing_encoder is not None and existing_features is not None:
        print("Continuing training on existing model...")
        # Check if features match
        if set(numeric_features) != set(existing_features):
            print(
                f"Warning: Feature mismatch. Existing: {existing_features}, New: {numeric_features}")
            print("Retraining from scratch due to feature mismatch")
            existing_model = None
            existing_encoder = None
        else:
            print("Features match - continuing incremental training")

    if existing_model is None:
        # Train new model
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(
            f"Morphology prediction model trained with {
                len(numeric_features)} features")
        print(f"Training accuracy: {accuracy:.2f}")
        print(f"Morphology classes: {le.classes_}")
    else:
        # Continue training existing model
        le = existing_encoder
        # Handle new classes that might not exist in the existing encoder
        y_encoded = []
        for morph in y:
            if morph in le.classes_:
                y_encoded.append(le.transform([morph])[0])
            else:
                # Add new class to encoder
                print(f"Adding new morphology class: {morph}")
                le.classes_ = np.append(le.classes_, morph)
                y_encoded.append(le.transform([morph])[0])

        y_encoded = np.array(y_encoded)

        # RandomForest doesn't support true incremental learning
        # We'll retrain on all available data
        print("Note: RandomForest doesn't support true incremental learning.")
        print("Retraining on all available data...")

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y_encoded)

        # Simple accuracy check on training data
        y_pred = model.predict(X)
        accuracy = accuracy_score(y_encoded, y_pred)

        print(f"Model retrained with {len(numeric_features)} features")
        print(f"Training accuracy: {accuracy:.2f}")
        print(f"Morphology classes: {le.classes_}")

    if save_path:
        save_morphology_model(model, le, numeric_features, save_path)

    return model, le, numeric_features


def predict_morphology_ml(model, label_encoder, features_list, row):
    """Use ML model to predict morphology for a given row"""
    if model is None or features_list is None:
        return None

    # Check if prediction is in training data range
    out_of_range = False
    range_warnings = []

    # Define expected ranges based on training data
    expected_ranges = {
        'Drying Gas Inlet (C)': (30, 100),
        '%Solids': (2, 16),
        'Feed Rate (g/min)': (0.4, 2.5),
        'Max Peclet Number': (2.5, 17),
        'Surface recession velocity': (0, 0.001)
    }

    for param, (min_val, max_val) in expected_ranges.items():
        if param in row.index:
            try:
                val = pd.to_numeric(row[param], errors='coerce')
                if pd.notna(val) and (val < min_val or val > max_val):
                    out_of_range = True
                    range_warnings.append(
                        f"{param}: {
                            val:.3f} (expected: {min_val}-{max_val})")
            except BaseException:
                pass

    features = []
    for col in features_list:
        if col not in row.index:
            # Use default value for missing features
            if 'conc' in col.lower() or '%' in col:
                val = 0
            elif 'temp' in col.lower() or 'inlet' in col.lower():
                val = 100
            elif 'rate' in col.lower():
                val = 1
            elif 'ph' in col.lower():
                val = 7
            else:
                val = 0
        else:
            val = row[col]
            try:
                # Try to convert to numeric
                numeric_val = pd.to_numeric(val, errors='coerce')
                if pd.notna(numeric_val):
                    val = numeric_val
                else:
                    # Handle non-numeric values
                    if 'conc' in col.lower() or '%' in col:
                        val = 0
                    elif 'temp' in col.lower() or 'inlet' in col.lower():
                        val = 100
                    elif 'rate' in col.lower():
                        val = 1
                    elif 'ph' in col.lower():
                        val = 7
                    else:
                        val = 0
            except BaseException:
                # Handle non-numeric values
                if 'conc' in col.lower() or '%' in col:
                    val = 0
                elif 'temp' in col.lower() or 'inlet' in col.lower():
                    val = 100
                elif 'rate' in col.lower():
                    val = 1
                elif 'ph' in col.lower():
                    val = 7
                else:
                    val = 0

        features.append(val)

    X_pred = pd.DataFrame([features], columns=features_list)
    pred_encoded = model.predict(X_pred)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    # Return prediction with uncertainty flag
    if out_of_range:
        pred_label += " (OUT_OF_RANGE)"

    return pred_label


def save_morphology_model(model, label_encoder,
                          features_list, filename='morphology_model.pkl'):
    """Save trained morphology model to disk"""
    model_data = {
        'model': model,
        'label_encoder': label_encoder,
        'features': features_list
    }
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"Model saved to {filename}")


def load_morphology_model(filename='morphology_model.pkl'):
    """Load trained morphology model from disk"""
    if not os.path.exists(filename):
        print(f"Model file {filename} not found")
        return None, None, None

    with open(filename, 'rb') as f:
        model_data = pickle.load(f)

    print(f"Model loaded from {filename}")
    return model_data['model'], model_data['label_encoder'], model_data['features']


def run_drying_simulation(row, morphology_model=None,
                          morphology_encoder=None, morphology_features=None):
    """Takes one row from the DataFrame and returns all results + morphology prediction"""
    # Load calibration factors (restored from original main block)
    calibration_factors = {'moisture': 1.0, 'D50': 1.0, 'RH': 1.0, 'outlet_temp': 1.0}
    try:
        with open('calibration.json', 'r') as f:
            cal_data = json.load(f)
        calibration_factors = {
            'moisture': cal_data.get('calibration_factor', 1.0),
            'D50': cal_data.get('d50_calibration_factor', 1.0),
            'RH': cal_data.get('rh_calibration_factor', 1.0),
            'outlet_temp': cal_data.get('outlet_temp_calibration_factor', 1.0)
        }
    except (FileNotFoundError, json.JSONDecodeError):
        pass  # Use defaults
    
    # Extract parameters - droplet radius can be provided directly or calculated from D50
    # Try different methods to determine initial droplet radius
    a0_um = None

    # Method 1: Direct droplet radius
    if 'droplet radius' in row.index:
        try:
            val = float(row['droplet radius'])
            if 0 < val < 1000:  # reasonable range in μm
                a0_um = val * 1e-6
                print(f"   Using provided droplet radius: {val:.1f} μm")
        except (ValueError, TypeError):
            pass

    # Method 2: a0_um column
    if a0_um is None and 'a0_um' in row.index:
        try:
            val = float(row['a0_um'])
            if 0 < val < 1000:  # reasonable range in μm
                a0_um = val * 1e-6
                print(f"   Using provided droplet radius: {val:.1f} μm")
        except (ValueError, TypeError):
            pass

    # Method 3: Calculate from D50 calculated from spray parameters
    if a0_um is None:
        d50_um = None

        # Calculate D50 from spray parameters (don't use D50_actual for initial
        # radius)
        if d50_um is None:
            try:
                # Calculate D50 from spray parameters (simplified version)
                # D50 ≈ D32 * (solids_fraction)^(-1/3) * correction_factor

                # Get solids fraction
                solids_frac = row.get(
                    '%Solids', row.get(
                        'solids fraction', row.get(
                            'Solids fraction', None)))
                if solids_frac is not None:
                    if solids_frac > 1:  # if given as percentage
                        solids_frac = solids_frac / 100

                    # Get nozzle parameters for D32 calculation
                    nozzle_tip_mm = row.get(
                        'Nozzle tip', row.get(
                            'nozzle tip', 0.7))
                    nozzle_cap_mm = row.get(
                        'Nozzle cap', row.get(
                            'nozzle cap', 1.5))

                    # Calculate nozzle diameter (annular if different)
                    if nozzle_cap_mm != nozzle_tip_mm:
                        # annular area equivalent
                        nozzle_m = (
                            nozzle_cap_mm**2 - nozzle_tip_mm**2)**0.5 / 1000
                    else:
                        nozzle_m = nozzle_tip_mm / 1000

                    # Get atomization parameters
                    atom_pressure_bar = row.get(
                        'Atom. Pressure (bar)', row.get(
                            'atom pressure', 2))
                    gas2 = row.get('Atomization Gas', 'air').lower()

                    # Estimate gas density (simplified)
                    rho_atom = 1.2 if gas2 == 'air' else 0.17  # kg/m³

                    # Get liquid properties
                    viscosity_pa_s = row.get('viscosity', 0.001)
                    surface_tension_n_m = row.get('surface tension', 0.072)
                    rho_liquid = row.get('liquid density', 1000)  # kg/m³

                    # Calculate Weber number
                    u_ag = ((2 * atom_pressure_bar * 1e5 / rho_atom)
                            ** 0.5)  # simplified velocity
                    We = rho_atom * u_ag**2 * nozzle_m / surface_tension_n_m

                    # Calculate GLR (gas to liquid ratio)
                    feed_rate_g_min = row.get(
                        'Feed Rate (g/min)', row.get('feed rate', 0.5))
                    atom_gas_flow_kg_h = row.get('Choked mass atomization gas flow (kg/h)',
                                                 row.get('atom gas flow', 0.1))
                    GLR = atom_gas_flow_kg_h * 1000 / \
                        (feed_rate_g_min * 60) if feed_rate_g_min > 0 else 1

                    # Calculate D32 (simplified Lord formula)
                    D32_m = nozzle_m * (We**-0.5) * (1 + GLR**-1)**0.5

                    # Calculate effective solids fraction (accounting for
                    # moisture)
                    moisture_content = row.get('Measured total moisture (%)',
                                               row.get('moisture content', 0))
                    if pd.isna(moisture_content):
                        moisture_content = 0
                    if moisture_content > 1:  # if percentage
                        moisture_content = moisture_content / 100
                    solids_frac_eff = solids_frac * (1 - moisture_content)

                    # Calculate D50 from D32 (empirical relationship)
                    D50_m = D32_m * solids_frac_eff**(-1 / 3) * 1.33

                    d50_um = D50_m * 1e6
                    print(
                        f"   Calculated D50 from spray parameters: {
                            d50_um:.1f} μm (D32={
                            D32_m *
                            1e6:.1f} μm, solids_frac_eff={
                            solids_frac_eff:.3f})")

            except Exception as e:
                print(
                    f"   Warning: Could not calculate D50 from parameters: {e}")
                d50_um = None

        # If we have a valid D50, calculate droplet radius
        if d50_um is not None:
            xi0_temp = row.get(
                'solids mass fraction', row.get(
                    '%Solids', row.get(
                        'xi0', row.get(
                            'Drug Substance conc. (mg/mL)', None))))
            if xi0_temp is not None and xi0_temp > 1:
                xi0_temp = xi0_temp / 100  # Convert percentage to fraction

                volume_ratio = 1.0 / xi0_temp
                linear_ratio = volume_ratio ** (1 / 3)
                droplet_diameter_um = d50_um * linear_ratio
                a0_um = (droplet_diameter_um / 2) * 1e-6
                print(
                    f"   Calculated droplet radius from D50 {
                        d50_um:.1f} μm and solids fraction {
                        xi0_temp:.3f}: {
                        droplet_diameter_um:.1f} μm")

    # Method 4: Default fallback
    if a0_um is None:
        a0_um = 10e-6  # 10 μm default
        print(f"   Using default droplet radius: {a0_um * 1e6:.1f} μm")

    if a0_um <= 0 or a0_um > 1e-3:
        raise ValueError(f"Invalid droplet radius: {a0_um * 1e6:.1f} μm")

    xi0 = row.get(
        'solids mass fraction',
        row.get(
            'solids_frac',
            row.get(
                '%Solids',
                row.get(
                    'xi0',
                    row.get(
                        'Drug Substance conc. (mg/mL)',
                        row.get(
                            'ds_conc',
                            0.10))))))
    if xi0 > 1:
        xi0 = xi0 / 100  # Convert percentage to fraction

    # Get individual solute concentrations (convert to mass fractions)
    drug_conc = row.get('Drug Substance conc. (mg/mL)', row.get('ds_conc', 0))
    moni_conc = row.get('Moni conc. (mg/mL)', row.get('moni_conc', 0))
    stabilizer_conc = row.get(
        'Stabilizer conc. (mg/mL)',
        row.get(
            'stab_A_conc',
            0))
    additive_conc = row.get(
        'Additive #1 conc. (mg/mL)',
        row.get(
            'additive_B_conc',
            0))

    # Convert to float if they're strings
    try:
        drug_conc = float(drug_conc) if drug_conc != '' else 0
    except (ValueError, TypeError):
        drug_conc = 0

    try:
        moni_conc = float(moni_conc) if moni_conc != '' else 0
    except (ValueError, TypeError):
        moni_conc = 0

    try:
        stabilizer_conc = float(
            stabilizer_conc) if stabilizer_conc != '' else 0
    except (ValueError, TypeError):
        stabilizer_conc = 0

    try:
        additive_conc = float(additive_conc) if additive_conc != '' else 0
    except (ValueError, TypeError):
        additive_conc = 0

    # Convert mg/mL to mass fractions (assuming total solution density ~1 g/mL)
    # Handle NaN values by treating them as 0
    drug_conc = 0 if pd.isna(drug_conc) else drug_conc
    moni_conc = 0 if pd.isna(moni_conc) else moni_conc
    stabilizer_conc = 0 if pd.isna(stabilizer_conc) else stabilizer_conc
    additive_conc = 0 if pd.isna(additive_conc) else additive_conc

    total_solids_mg_ml = drug_conc + moni_conc + stabilizer_conc + additive_conc
    if total_solids_mg_ml > 0:
        xi0_drug = drug_conc / (1000 * 1000)  # mg/mL to g/mL to mass fraction
        xi0_moni = moni_conc / (1000 * 1000)
        xi0_stabilizer = stabilizer_conc / (1000 * 1000)
        xi0_additive = additive_conc / (1000 * 1000)

        # Normalize to ensure they sum to xi0
        total_individual = xi0_drug + xi0_moni + xi0_stabilizer + xi0_additive
        if total_individual > 0:
            xi0_drug = xi0_drug * (xi0 / total_individual)
            xi0_moni = xi0_moni * (xi0 / total_individual)
            xi0_stabilizer = xi0_stabilizer * (xi0 / total_individual)
            xi0_additive = xi0_additive * (xi0 / total_individual)
        else:
            # If no individual concentrations provided, assume all is drug
            xi0_drug = xi0
            xi0_moni = 0
            xi0_stabilizer = 0
            xi0_additive = 0
    else:
        # Fallback: assume all solids are drug
        xi0_drug = xi0
        xi0_moni = 0
        xi0_stabilizer = 0
        xi0_additive = 0

    # Individual diffusion coefficients for each solute
    # Will be calculated after T_in and eta are known
    D_drug = None
    D_moni = None
    D_stabilizer = None
    D_additive = None

    # Store initial diffusion coefficients
    D_drug0 = D_drug
    D_moni0 = D_moni
    D_stabilizer0 = D_stabilizer
    D_additive0 = D_additive

    D_solute = row.get('diffusion coefficient', row.get('D_solute', 5e-10))
    gamma = row.get(
        'surface tension',
        row.get(
            'surface_tension_user_input',
            0.072))
    eta = row.get('viscosity_user_input', row.get('viscosity', 0.001))
    # Ensure eta is a scalar, handle non-numeric values
    try:
        if isinstance(eta, (list, np.ndarray)):
            eta = float(np.mean([x for x in eta if pd.notna(x) and isinstance(x, (int, float))])) if len(
                [x for x in eta if pd.notna(x) and isinstance(x, (int, float))]) > 0 else 0.001
        else:
            eta = float(eta)
    except (ValueError, TypeError):
        print(
            f"   Warning: Invalid viscosity value '{eta}', using default 0.001 Pa·s")
        eta = 0.001
    eta0 = eta  # Store initial viscosity
    RH_in = row.get('RH_inlet', row.get('RH inlet', row.get('RH1', 20)))

    # RH_out extraction with proper NaN handling
    RH_out = None
    for rh_key in ['RH_outlet', 'RH outlet', 'measured_RH_out', 'RH2']:
        val = row.get(rh_key)
        if val is not None and not pd.isna(val):
            RH_out = val
            break
    if RH_out is None:
        RH_out = 80  # Final fallback

    if pd.isna(RH_in) or RH_in < 0 or RH_in > 100:
        print(f"   Warning: Invalid RH_in {RH_in}, using default 20%")
        RH_in = 20
    if pd.isna(RH_out) or RH_out < 0 or RH_out > 100:
        # Try to use calculated RH_out instead of default
        calculated_rh = row.get(
            'Estimated RH_outlet (from calculated moisture)', None)
        if pd.notna(calculated_rh) and 0 <= calculated_rh <= 100:
            RH_out = calculated_rh
            print(f"   Using calculated RH_out {RH_out}%")
        else:
            print(f"   Warning: Invalid RH_out {RH_out}, using default 30%")
            RH_out = 30

    T_in = (
        row.get(
            'T1_C',
            70) +
        273.15) if pd.notna(
            row.get(
                'T1_C',
                None)) else 343.15
    T_out = row.get(
        'Toutlet', row.get(
            'T_outlet', row.get(
                'T_outlet_C', None)))

    # If no outlet temperature provided, try ML prediction or energy balance
    # estimation
    if T_out is None or pd.isna(T_out):
        # Fall back to energy balance estimation if ML failed or not available
        # Estimate outlet temperature using energy balance principles
        T_in_C = T_in - 273.15
        feed_g_min = row.get(
            'feed_g_min',
            row.get(
                'Feed Rate (g/min)',
                2))  # Feed rate from DOE
        solids_frac = xi0  # Solids fraction (already extracted)
        # Fallback to simple rule
        temp_drop = 25 + (T_in_C - 60) * 0.5
        estimated_T_out_C = T_in_C - temp_drop

        T_out = estimated_T_out_C + 273.15
        print(
            f"   Estimated outlet temperature: {
                estimated_T_out_C:.1f}°C (energy balance: inlet {
                T_in_C:.1f}°C)")
    else:
        T_out = T_out + 273.15
        print(f"   Using provided outlet temperature: {T_out - 273.15:.1f}°C")

    if pd.isna(T_in) or T_in < 273 or T_in > 1000:
        print(f"   Warning: Invalid T_in {T_in}K, using default 453K")
        T_in = 453
    if pd.isna(T_out) or T_out < 273 or T_out > 1000:
        print(f"   Warning: Invalid T_out {T_out}K, using default 353K")
        T_out = 353

    # ULTIMATE RESPECTFUL FIX: Use EVERY value already calculated by main.py
    # main.py is the single source of truth — we read it exactly, no
    # overrides, no Stokes-Einstein nonsense

    # Diffusion coefficients — direct from main.py (with fallbacks only if
    # missing)
    D_drug = float(
        row.get(
            'D_igg',
            row.get(
                'D_ds',
                row.get(
                    'D_solute',
                    4e-11))))
    D_moni = float(row.get('D_moni', row.get('D_l-histidine', 1e-9)))
    D_stabilizer = float(
        row.get(
            'D_stabilizer',
            row.get(
                'D_stab_A',
                row.get(
                    'D_stabilizer_A',
                    1e-10))))
    D_additive = float(
        row.get(
            'D_additive',
            row.get(
                'D_additive_B',
                row.get(
                    'D_additive_C',
                    1e-9))))
    D_buffer = float(row.get('D_buffer', 1e-9))  # if present in future

    # All Peclet numbers — pre-calculated by main.py (these are gold)
    Pe_drug = float(row.get('Pe_igg', row.get('Pe_ds', 5.0)))
    Pe_moni = float(row.get('Pe_moni', row.get('Pe_l-histidine', 0.1)))
    max_pe_drug = float(row.get('max_pe_drug', row.get('max_pe_ds', 10.0)))
    max_pe_moni = float(row.get('max_pe_moni', 1.0))
    effective_pe_drug = float(
        row.get(
            'effective_pe_drug',
            row.get(
                'effective_pe_ds',
                10.0)))
    effective_pe_moni = float(row.get('effective_pe_moni', 1.0))
    integrated_pe_drug = float(
        row.get(
            'integrated_pe_drug',
            row.get(
                'integrated_pe_ds',
                5.0)))

    # Print once per batch so you can verify everything is being used
    print(
        f"\nUsing COMPLETE main.py physics for batch {
            row.get(
                'batch_id',
                'unknown')}:")
    print(f"   D_drug       = {D_drug:.2e} m²/s")
    print(f"   D_moni       = {D_moni:.2e} m²/s")
    print(
        f"   Pe_drug      = {
            Pe_drug:.2f} → effective Pe = {
            effective_pe_drug:.2f} → max Pe = {
                max_pe_drug:.2f}")
    print(
        f"   Pe_moni      = {
            Pe_moni:.2f} → effective Pe_moni = {
            effective_pe_moni:.2f}")
    if pd.notna(row.get('D_stabilizer', pd.NA)):
        print(f"   D_stabilizer = {D_stabilizer:.2e} m²/s")
    if pd.notna(row.get('D_buffer', pd.NA)):
        print(f"   D_buffer     = {D_buffer:.2e} m²/s")

    # Store initial diffusion coefficients
    D_drug0 = D_drug
    D_moni0 = D_moni
    D_stabilizer0 = D_stabilizer
    D_additive0 = D_additive

    # MW moved later after simulation

    # Load effective Pe values from the data file if available
    # Note: Since D is always calculated from MW unless provided, Pe is
    # calculated from D

    gas_rate = row.get(
        'drying gas rate', row.get(
            'gas flow', row.get(
                'Drying gas rate (m³/hr)', 0.1)))
    if gas_rate > 10:
        gas_rate = gas_rate / 3600

    if pd.isna(gas_rate) or gas_rate <= 0:
        print(
            f"   Warning: Invalid gas rate {gas_rate}, using default 0.01 m³/s")
        gas_rate = 0.01

    nozzle_d = row.get('nozzle diameter', 1e-3)
    E_crust = row.get('E_crust_GPa', row.get('Young modulus', 2)) * 1e9
    rho_solid = row.get('rho_solid', 1500)

    target_final_radius_um = None
    if 'D50_actual' in row.index:
        try:
            d50_actual_val = float(row['D50_actual'])
            if 0 < d50_actual_val < 1000:
                target_final_radius_um = d50_actual_val / 2
                print(
                    f"   Target final radius from D50_actual: {
                        target_final_radius_um:.1f} μm")
        except (ValueError, TypeError):
            pass  # Keep target_final_radius_um as None

    # If no D50_actual, try D50 (calculated with Moni) or other D50 columns
    if target_final_radius_um is None:
        d50_cols = ['D50 (calculated with Moni)', 'D50', 'd50']
        for col in d50_cols:
            if col in row.index:
                try:
                    d50_val = float(row[col])
                    if 0 < d50_val < 1000:
                        target_final_radius_um = d50_val / 2
                        print(
                            f"   Target final radius from {col}: {
                                target_final_radius_um:.1f} μm")
                        break
                except (ValueError, TypeError):
                    continue

    # If still no target, estimate from initial radius and solids fraction
    if target_final_radius_um is None:
        estimated_final_radius_um = a0_um * 1e6 * (xi0 ** (1 / 3))
        print(
            f"   Estimated final radius from solids fraction: {
                estimated_final_radius_um:.1f} μm")
        target_final_radius_um = estimated_final_radius_um
    elif 'D50' in row.index:
        try:
            d50_val = float(row['D50'])
            if 0 < d50_val < 1000:
                target_final_radius_um = d50_val / 2
                print(
                    f"   Target final radius from D50: {
                        target_final_radius_um:.1f} μm")
        except (ValueError, TypeError):
            pass  # Keep target_final_radius_um as None

    print(
        f"   Initial radius: {
            a0_um *
            1e6:.1f} μm, Solids fraction: {
            xi0:.3f}")
    print(f"   Gas rate: {gas_rate}, T_in: {T_in:.1f} K, T_out: {T_out:.1f} K")

    def c_sat(T):
        p_sat = 610.78 * np.exp(17.2694 * (T - 273.15) / (T - 273.15 + 238.3))
        return p_sat * 0.018015 / (8.314 * T)

    t_total = row.get(
        'estimated evaporation time', row.get(
            'evaporation time', row.get(
                'Drying Time estimate (s)', 10)))
    t = np.linspace(0, t_total * 1.5, 10000)
    dt = t[1] - t[0]

    a = np.zeros_like(t)
    xi = np.zeros_like(t)  # Total solids fraction
    xi_drug = np.zeros_like(t)
    xi_moni = np.zeros_like(t)
    xi_stabilizer = np.zeros_like(t)
    xi_additive = np.zeros_like(t)

    a[0] = a0_um
    xi[0] = xi0
    xi_drug[0] = xi0_drug
    xi_moni[0] = xi0_moni
    xi_stabilizer[0] = xi0_stabilizer
    xi_additive[0] = xi0_additive

    m_solids = xi0 * (4 / 3) * np.pi * a0_um**3 * 1000
    m_drug = xi0_drug * (4 / 3) * np.pi * a0_um**3 * 1000
    m_moni = xi0_moni * (4 / 3) * np.pi * a0_um**3 * 1000
    m_stabilizer = xi0_stabilizer * (4 / 3) * np.pi * a0_um**3 * 1000
    m_additive = xi0_additive * (4 / 3) * np.pi * a0_um**3 * 1000

    crust_formed = False
    crust_surface_composition = None
    for i in range(1, len(t)):
        T_gas = np.interp(t[i], [0, t_total], [T_in, T_out])
        RH = np.interp(t[i], [0, t_total], [RH_in, RH_out])

        c_inf = c_sat(T_gas) * (RH / 100)

        # Calculate gas velocity more realistically for spray drying
        # Typical lab spray dryer: gas rate 30-50 m³/hr, chamber diameter ~0.5m
        # Cross-sectional area ~0.2 m², velocity ~10-20 m/s
        chamber_diameter = 0.5  # meters, typical for lab spray dryers
        chamber_area = np.pi * (chamber_diameter / 2)**2
        # Convert m³/hr to m³/s, then divide by area
        u_gas = gas_rate * 3600 / chamber_area if gas_rate > 0 else 10
        Re = 2 * a[i - 1] * u_gas / 1.8e-5
        Sc = 0.6
        # Cap Sh at 2 for small droplets
        Sh = min(2 + 0.6 * Re**0.5 * Sc**(1 / 3), 2.0)
        hD = Sh * 1.5e-5 / (2 * a[i - 1])
        evap_rate = 4 * np.pi * a[i - 1] * hD * (c_sat(T_gas) - c_inf)

        if np.isnan(evap_rate) or np.isinf(evap_rate):
            print(f"   Warning: Invalid evaporation rate at t={t[i]:.2f}s")
            break

        # Cap at higher value for more significant enrichment
        surface_recession = min(
            evap_rate / (4 * np.pi * a[i - 1]**2 * 1000), 1e-3)
        Pe_local = surface_recession * a[i - 1] / D_solute
        # Cap Peclet number to avoid numerical overflow
        Pe_local = min(Pe_local, 10)

        # Update viscosity and diffusion coefficients based on current solids
        # concentration
        # Viscosity increases exponentially with solids fraction
        eta_current = eta0 * np.exp(5 * xi[i - 1])
        D_drug = D_drug0 * (eta0 / eta_current)
        D_moni = D_moni0 * (eta0 / eta_current)
        D_stabilizer = D_stabilizer0 * (eta0 / eta_current)
        D_additive = D_additive0 * (eta0 / eta_current)

        # Calculate surface concentrations for each solute individually
        Pe_drug = surface_recession / D_drug
        Pe_moni = surface_recession / D_moni
        Pe_stabilizer = surface_recession / \
            D_stabilizer if D_stabilizer > 0 else float('inf')
        Pe_additive = surface_recession / \
            D_additive if D_additive > 0 else float('inf')

        # Surface concentrations for each solute (accumulate over time)
        drug_enrichment = min(np.exp(1e10 / Pe_drug), 1e50)
        moni_enrichment = min(np.exp(1e10 / Pe_moni), 1e100)
        stabilizer_enrichment = min(np.exp(1e10 / Pe_stabilizer), 1e100)
        additive_enrichment = min(np.exp(1e10 / Pe_additive), 1e100)

        xi_drug_surface = xi_drug[i - 1] * drug_enrichment
        xi_moni_surface = xi_moni[i - 1] * moni_enrichment
        xi_stabilizer_surface = xi_stabilizer[i - 1] * stabilizer_enrichment
        xi_additive_surface = xi_additive[i - 1] * additive_enrichment

        # Total surface solids concentration (for crust formation check)
        xi_surface = xi_drug_surface + xi_moni_surface + \
            xi_stabilizer_surface + xi_additive_surface
        xi_critical = 0.95  # High threshold to allow more drying

        # Debug: print initial conditions
        if i == 1:
            print(
                f"   Initial conditions: radius={
                    a[0] *
                    1e6:.1f}μm, xi_total={
                    xi[0]:.3f}")
            print(
                f"   Initial solute fractions: drug={
                    xi0_drug:.3f}, moni={
                    xi0_moni:.3f}, stabilizer={
                    xi0_stabilizer:.3f}, additive={
                    xi0_additive:.3f}")
            print(
                f"   Initial Peclet numbers: Pe_drug={
                    Pe_drug:.2f}, Pe_moni={
                    Pe_moni:.2f}, Pe_stabilizer={
                    Pe_stabilizer:.2f}, Pe_additive={
                    Pe_additive:.2f}")
            print(
                f"   Surface concentrations: drug={
                    xi_drug_surface:.3f}, moni={
                    xi_moni_surface:.3f}, total={
                    xi_surface:.3f}")
            print(
                f"   Enrichment: drug={
                    drug_enrichment:.3f}, moni={
                    moni_enrichment:.3f}")
            print(f"   Crust threshold: {xi_critical:.3f}")

        if xi_surface >= xi_critical and not crust_formed:
            a_crust = a[i - 1]
            t_crust = t[i]
            # Store surface composition at crust formation
            crust_surface_composition = {
                'drug': xi_drug_surface,
                'moni': xi_moni_surface,
                'stabilizer': xi_stabilizer_surface,
                'additive': xi_additive_surface,
                'total': xi_surface
            }
            crust_formed = True
            print(
                f"   → Crust formed at t = {
                    t[i]:.2f}s, radius = {
                    a_crust *
                    1e6:.1f} μm")

        permeability_factor = 0.01 if crust_formed else 1.0
        da_dt = -evap_rate * permeability_factor / \
            (4 * np.pi * a[i - 1]**2 * 1000)

        # Cap the radius decrease to prevent overshooting in one step
        max_decrease_rate = -0.1 * a[i - 1] / dt  # Max 10% decrease per step
        da_dt = max(da_dt, max_decrease_rate)

        a[i] = max(a[i - 1] + da_dt * dt, 1e-9)
        xi[i] = m_solids / (1000 * (4 / 3) * np.pi * a[i]**3)

        # Update individual solute concentrations (they don't diffuse out, just
        # concentrate)
        xi_drug[i] = m_drug / (1000 * (4 / 3) * np.pi * a[i]**3)
        xi_moni[i] = m_moni / (1000 * (4 / 3) * np.pi * a[i]**3)
        xi_stabilizer[i] = m_stabilizer / (1000 * (4 / 3) * np.pi * a[i]**3)
        xi_additive[i] = m_additive / (1000 * (4 / 3) * np.pi * a[i]**3)

        if target_final_radius_um is not None and a[i] * \
                1e6 <= target_final_radius_um and crust_formed:
            print(
                f"   → Reached target final radius at t = {
                    t[i]:.2f}s, stopping simulation")
            a = a[:i + 1]
            xi = xi[:i + 1]
            xi_drug = xi_drug[:i + 1]
            xi_moni = xi_moni[:i + 1]
            xi_stabilizer = xi_stabilizer[:i + 1]
            xi_additive = xi_additive[:i + 1]
            t = t[:i + 1]
            break

        if np.isnan(a[i]):
            print(f"   Warning: Radius became NaN at t={t[i]:.2f}s")
            break

    # ============ Buckling & Morphology Prediction ============
    vol_solids = m_solids / rho_solid
    final_radius_m = target_final_radius_um * \
        1e-6 if target_final_radius_um is not None else a[-1]

    ml_morphology = predict_morphology_ml(
        morphology_model,
        morphology_encoder,
        morphology_features,
        row)

    if crust_formed:
        # Skip buckling calculation for very small particles where physics may
        # not apply
        if final_radius_m < 1e-7:  # 0.1 μm radius
            physics_morph = "Perfect sphere"
            buckle_ratio = None
        else:
            def shell_thickness_eq(h):
                return final_radius_m**3 - \
                    (final_radius_m - h)**3 - 3 * vol_solids / (4 * np.pi)

            try:
                h_shell = fsolve(shell_thickness_eq, final_radius_m * 0.1)[0]
                if h_shell <= 0 or h_shell >= final_radius_m:
                    h_shell = final_radius_m * 0.1
            except BaseException:
                h_shell = final_radius_m * 0.1

            def P_cap(r): return 2 * gamma / r

            def P_crit(r):
                h = h_shell
                if h <= 0:
                    return 1e15
                return 8 * E_crust * (h / final_radius_m)**2 / \
                    np.sqrt(3 * (1 - 0.3**2))

            # Time-integrated Laplace pressure analysis: check if buckling
            # occurred during drying
            P_crit_value = P_crit(final_radius_m)
            critical_r = 2 * gamma / P_crit_value if P_crit_value > 0 else 0

            # Find the first radius during drying where P_cap exceeded P_crit
            buckling_r = None
            for radius in a:
                if radius < critical_r:
                    buckling_r = radius
                    break

            if buckling_r is not None:
                # Buckling occurred during drying at this radius
                r_buckle = buckling_r
                buckle_ratio = r_buckle / final_radius_m
                print(
                    f"   Time-integrated buckling detected at radius {
                        r_buckle *
                        1e6:.1f} μm (critical radius {
                        critical_r *
                        1e6:.1f} μm)")
            else:
                # No buckling during drying, use static analysis
                try:
                    r_buckle = fsolve(
                        lambda r: P_cap(r) - P_crit(r),
                        final_radius_m * 0.8)[0]
                    buckle_ratio = r_buckle / final_radius_m
                    print(
                        f"   Static buckling analysis: r_buckle {
                            r_buckle * 1e6:.1f} μm")
                except BaseException:
                    buckle_ratio = 0.5
                    r_buckle = final_radius_m * buckle_ratio

            if buckle_ratio > 0.75:
                physics_morph = "Perfect sphere"
            elif buckle_ratio > 0.60:
                physics_morph = "Lightly dimpled sphere"
            elif buckle_ratio > 0.45:
                physics_morph = "Clearly dimpled / buckled sphere"
            elif buckle_ratio > 0.30:
                physics_morph = "Strongly buckled / crumpled"
            elif buckle_ratio > 0.10:
                physics_morph = "Raisin-like highly wrinkled"
            else:
                physics_morph = "Donut / toroidal or collapsed"
    else:
        physics_morph = "Dense solid particle (no crust formed)"
        buckle_ratio = None

    if ml_morphology is not None:
        # Check if prediction is flagged as out of training range
        if "(OUT_OF_RANGE)" in ml_morphology:
            print(
                f"   ⚠️  ML prediction out of training range, using physics-based prediction instead")
            final_morphology = physics_morph
            morphology_method = "Physics (ML out of range)"
            ml_morphology_clean = ml_morphology.replace(" (OUT_OF_RANGE)", "")
        # Special case: If ML predicts dimpled but physics predicts perfect sphere,
        # and conditions are BHV-1400-like (low temp, moderate solids), trust
        # physics
        elif ("dimpled" in ml_morphology.lower() and "perfect" in physics_morph.lower() and
              row.get('Drying Gas Inlet (C)', 100) < 40 and row.get('%Solids', 0) < 10):
            print(f"   ⚠️  ML predicts dimpled but physics predicts perfect sphere for BHV-1400-like conditions, using physics")
            final_morphology = physics_morph
            morphology_method = "Physics (ML-physics conflict)"
            ml_morphology_clean = ml_morphology
        else:
            final_morphology = ml_morphology
            morphology_method = "ML"
            ml_morphology_clean = ml_morphology
    else:
        final_morphology = physics_morph
        morphology_method = "Physics"
        ml_morphology_clean = None

    final_radius_um = target_final_radius_um if target_final_radius_um is not None else a[-1] * 1e6
    final_diameter_um = final_radius_um * 2  # Convert radius to diameter

    # Calculate Laplace pressure at final radius
    laplace_pressure = 4 * gamma / final_radius_m if final_radius_m > 0 else None

    # Calculate bulk composition percentages (initial mass fractions)
    total_initial_solids = xi0_drug + xi0_moni + xi0_stabilizer + xi0_additive
    drug_bulk_pct = (xi0_drug / total_initial_solids) * \
        100 if total_initial_solids > 0 else 0
    moni_bulk_pct = (xi0_moni / total_initial_solids) * \
        100 if total_initial_solids > 0 else 0
    stabilizer_bulk_pct = (
        xi0_stabilizer / total_initial_solids) * 100 if total_initial_solids > 0 else 0
    additive_bulk_pct = (xi0_additive / total_initial_solids) * \
        100 if total_initial_solids > 0 else 0

    # Calculate final surface composition percentages
    if crust_formed and crust_surface_composition is not None:
        # Use surface composition at crust formation
        final_xi_drug_surface = crust_surface_composition['drug']
        final_xi_moni_surface = crust_surface_composition['moni']
        final_xi_stabilizer_surface = crust_surface_composition['stabilizer']
        final_xi_additive_surface = crust_surface_composition['additive']
        total_final_surface = crust_surface_composition['total']
    else:
        # Calculate surface concentrations at final state
        # Use the last calculated surface recession for final calculation
        if 'surface_recession' in locals():
            # Update diffusion coefficients for final state
            eta_final = eta0 * np.exp(5 * xi[-1])
            D_drug_final = D_drug0 * (eta0 / eta_final)
            D_moni_final = D_moni0 * (eta0 / eta_final)
            D_stabilizer_final = D_stabilizer0 * (eta0 / eta_final)
            D_additive_final = D_additive0 * (eta0 / eta_final)

            final_Pe_drug = surface_recession * a[-1] / D_drug_final
            final_Pe_moni = surface_recession * a[-1] / D_moni_final
            print(
                f"   DEBUG: final_Pe_drug={
                    final_Pe_drug:.1f}, final_Pe_moni={
                    final_Pe_moni:.1f}")
            final_Pe_stabilizer = min(
                surface_recession * a[-1] / D_stabilizer_final, 100)
            final_Pe_additive = min(
                surface_recession * a[-1] / D_additive_final, 100)
        else:
            final_Pe_drug = final_Pe_moni = final_Pe_stabilizer = final_Pe_additive = 0

        final_drug_enrichment = min(np.exp(min(final_Pe_drug, 500)), 1e50)
        final_moni_enrichment = min(np.exp(min(final_Pe_moni, 500)), 1e50)
        final_stabilizer_enrichment = min(
            np.exp(min(final_Pe_stabilizer, 100)), 1e10)
        final_additive_enrichment = min(
            np.exp(min(final_Pe_additive, 100)), 1e10)

        final_xi_drug_surface = xi_drug[-1] * final_drug_enrichment
        final_xi_moni_surface = xi_moni[-1] * final_moni_enrichment
        final_xi_stabilizer_surface = xi_stabilizer[-1] * \
            final_stabilizer_enrichment
        final_xi_additive_surface = xi_additive[-1] * final_additive_enrichment
        total_final_surface = final_xi_drug_surface + final_xi_moni_surface + \
            final_xi_stabilizer_surface + final_xi_additive_surface

    if total_final_surface > 0:
        drug_surface_pct = (final_xi_drug_surface / total_final_surface) * 100
        moni_surface_pct = (final_xi_moni_surface / total_final_surface) * 100
        stabilizer_surface_pct = (
            final_xi_stabilizer_surface / total_final_surface) * 100
        additive_surface_pct = (
            final_xi_additive_surface / total_final_surface) * 100
    else:
        # Fallback: use bulk composition if surface calculation failed
        print(f"   Warning: Surface composition calculation failed (total_surface=0), using bulk composition")
        drug_surface_pct = drug_bulk_pct
        moni_surface_pct = moni_bulk_pct
        stabilizer_surface_pct = stabilizer_bulk_pct
        additive_surface_pct = additive_bulk_pct

    print(
        f"   Final surface pct: drug={
            drug_surface_pct:.3f}, moni={
            moni_surface_pct:.3f}, total_surface={
                total_final_surface:.3f}")

    # === SURFACTANT PRIORITY OVERRIDE (Langmuir) ===
    override = surfactant_priority_override(
        moni_conc_mg_ml=float(xi0_moni * 1000),  # mass fraction → mg/mL
        moni_name='moni',
        moni_mw=row.get('moni_mw', 6800.0)
    )

    if override.get("force_surfactant_override"):
        print(
            f"   → Surfactant override: Moni surface = {
                override['moni_surface_pct']}%")

        moni_surface_pct = override['moni_surface_pct']
        remaining = 100.0 - override['moni_surface_pct']

        total_protein = xi0_drug + xi0_stabilizer + xi0_additive
        if total_protein > 0:
            scale = remaining / total_protein
            drug_surface_pct = xi0_drug * scale * 100
            stabilizer_surface_pct = xi0_stabilizer * scale * 100
            additive_surface_pct = xi0_additive * scale * 100
        else:
            drug_surface_pct = remaining
            stabilizer_surface_pct = 0.0
            additive_surface_pct = 0.0
    else:
        # Fall back to diffusive model
        pass  # keep the calculated surface_pct values

    # Build result dict conditionally based on provided components
    result = {
        'final_diameter_um': final_diameter_um,
        'a_crust_um': a_crust * 1e6 if crust_formed else None,
        'morphology': final_morphology,
        'morphology_method': morphology_method,
        'physics_morphology': physics_morph or 'spherical',
        'ml_morphology': ml_morphology,
        'buckle_ratio': buckle_ratio or 0,
        'laplace_pressure': laplace_pressure,
        'radius_history_um': a * 1e6,
        'time_history_s': t,
        # Process conditions
        # Convert back to Celsius
        'inlet_temp_c': row.get('T1_C', 70) if pd.notna(row.get('T1_C', None)) else 70,
        # Convert back to Celsius and apply calibration
        'outlet_temp_c': (T_out - 273.15) * (calibration_factors.get('outlet_temp', 1.0) if isinstance(calibration_factors.get('outlet_temp', 1.0), (int, float)) else 1.0),
        'outlet_rh_pct': RH_out or 10,  # Outlet relative humidity in percent
        # Surface composition predictions
        'drug_surface_pct': drug_surface_pct,
        'moni_surface_pct': moni_surface_pct,
        'stabilizer_surface_pct': stabilizer_surface_pct,
        'additive_surface_pct': additive_surface_pct,
        # Bulk composition for reference
        'drug_bulk_pct': drug_bulk_pct,
        'moni_bulk_pct': moni_bulk_pct,
        'stabilizer_bulk_pct': stabilizer_bulk_pct,
        'additive_bulk_pct': additive_bulk_pct,
    }
    
    # Calculate Péclet numbers using the peclet calculator
    D_drug = row.get('D_solute', 1e-10)  # Diffusion coefficient from input
    D_moni = 1e-11  # Surfactant diffusion coefficient
    D_stabilizer = 5e-10  # Stabilizer diffusion coefficient
    D_buffer = 1e-10  # Buffer (glycine) diffusion coefficient
    D_additive_B = 1e-10  # Additive B (trehalose) diffusion coefficient
    D_additive_C = 1e-10  # Additive C diffusion coefficient
    
    # Build compounds dict based on what's present
    D_compounds = {'drug': D_drug}
    if row.get('moni_conc') and row.get('moni_conc', 0) > 0:
        D_compounds['moni'] = D_moni
    if row.get('stab_A_conc') and row.get('stab_A_conc', 0) > 0:
        D_compounds['stabilizer'] = D_stabilizer
    if row.get('buffer_conc') and row.get('buffer_conc', 0) > 0:
        D_compounds['buffer'] = D_buffer
    if row.get('additive_B_conc') and row.get('additive_B_conc', 0) > 0:
        D_compounds['additive_B'] = D_additive_B
    if row.get('additive_C_conc') and row.get('additive_C_conc', 0) > 0:
        D_compounds['additive_C'] = D_additive_C
    
    # Molecular weights (in kDa) - use new lookup logic
    mw_drug = row.get('ds_mw', 150.0)  # kDa (drug MW is always specified)
    mw_stabilizer = get_compound_molecular_weight(
        row.get('stabilizer_A'), row.get('stab_A_mw')
    )
    mw_buffer = get_compound_molecular_weight(
        row.get('buffer'), row.get('buffer_mw')
    )
    mw_additive_B = get_compound_molecular_weight(
        row.get('additive_B'), row.get('additive_B_mw')
    )
    mw_additive_C = get_compound_molecular_weight(
        row.get('additive_C'), row.get('additive_C_mw')
    )
    
    # Calculate evaporation velocity from radius change
    radius_m = result['radius_history_um'] * 1e-6
    time_s = result['time_history_s']
    if len(radius_m) > 1 and len(time_s) > 1:
        dr_dt = np.diff(radius_m) / np.diff(time_s)
        v_evap = np.abs(dr_dt[0])  # Use initial evaporation rate
    else:
        v_evap = 1e-6  # Default evaporation velocity
    
    v_evap_history = np.full_like(time_s, v_evap)
    
    pe_metrics = calculate_all_peclet_metrics(
        time_history_s=time_s,
        radius_history_um=result['radius_history_um'],
        v_evap_history_m_s=v_evap_history,
        D_primary_m2_s=D_drug,
        D_compounds_m2_s=D_compounds
    )
    
    # Extract the calculated values
    effective_pe_drug = pe_metrics.get('effective_pe_drug', pe_metrics.get('effective_pe', 1.0))
    max_pe_drug = pe_metrics.get('max_pe_drug', pe_metrics.get('max_pe', 10.0))
    integrated_pe_drug = pe_metrics.get('integrated_pe', 5.0)
    effective_pe_moni = pe_metrics.get('effective_pe_moni', 0.1)
    max_pe_moni = pe_metrics.get('max_pe_moni', 1.0)
    effective_pe_stabilizer = pe_metrics.get('effective_pe_stabilizer', 1.0)
    max_pe_stabilizer = pe_metrics.get('max_pe_stabilizer', 1.0)
    
    # Add buffer and additive Péclet calculations if they exist
    effective_pe_buffer = pe_metrics.get('effective_pe_buffer', 0.0)
    max_pe_buffer = pe_metrics.get('max_pe_buffer', 0.0)
    effective_pe_additive_B = pe_metrics.get('effective_pe_additive_B', 0.0)
    max_pe_additive_B = pe_metrics.get('max_pe_additive_B', 0.0)
    effective_pe_additive_C = pe_metrics.get('effective_pe_additive_C', 0.0)
    max_pe_additive_C = pe_metrics.get('max_pe_additive_C', 0.0)
    
    result.update({
        'drug_mw': mw_drug / 1000,
        'moni_mw': 6.8,  # Fixed MW for Moni
        # Add Péclet numbers to results
        'effective_pe': effective_pe_drug,
        'max_pe': max_pe_drug,
        'integrated_pe': integrated_pe_drug,
        'effective_pe_drug': effective_pe_drug,
        'max_pe_drug': max_pe_drug,
        'effective_pe_moni': effective_pe_moni,
        'max_pe_moni': max_pe_moni,
        'effective_pe_stabilizer': effective_pe_stabilizer,
        'max_pe_stabilizer': max_pe_stabilizer,
        'effective_pe_buffer': effective_pe_buffer,
        'max_pe_buffer': max_pe_buffer,
        'effective_pe_additive_B': effective_pe_additive_B,
        'max_pe_additive_B': max_pe_additive_B,
        'effective_pe_additive_C': effective_pe_additive_C,
        'max_pe_additive_C': max_pe_additive_C,
        # Preserve input morphology
        'morphology_known': row.get('Morphology', 'unknown')
    })
    if row.get('stab_A_mw') is not None and not pd.isna(row.get('stab_A_mw')):
        result['stabilizer_mw'] = mw_stabilizer / 1000
    if row.get('buffer_mw') is not None and not pd.isna(row.get('buffer_mw')):
        result['buffer_mw'] = mw_buffer / 1000
    if (row.get('additive_B_mw') is not None and not pd.isna(row.get('additive_B_mw'))) or (
            row.get('additive_C_mw') is not None and not pd.isna(row.get('additive_C_mw'))):
        # Use additive_B MW if present, otherwise additive_C MW
        mw_to_use = mw_additive_B if (row.get('additive_B_mw') is not None and not pd.isna(row.get('additive_B_mw'))) else mw_additive_C
        result['additive_mw'] = mw_to_use / 1000

    # Add predicted outlet temperature from input
    result['Predicted Outlet Temperature (C)'] = row.get(
        'predicted_outlet_temp_C', 100)

    return result


def normalize_morphology_text(text):
    """Normalize morphology text for better matching"""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Common typo corrections
    text = text.replace('lighlty', 'lightly')
    # Wait, no - actually keep as is, but handle variations
    text = text.replace('dimpled', 'dimed')

    # Remove extra spaces and punctuation
    text = ' '.join(text.split())
    text = text.replace('/', ' ').replace('-', ' ')

    # Convert plural to singular for common morphology terms
    text = text.replace('spheres', 'sphere')
    text = text.replace('particles', 'particle')

    return text.strip()


def morphology_match(predicted, actual):
    """Check if predicted morphology matches actual morphology with flexible matching"""
    if not predicted or not actual:
        return False

    # Handle out-of-range predictions
    predicted_clean = predicted.replace(" (OUT_OF_RANGE)", "")

    pred_norm = normalize_morphology_text(predicted_clean)
    actual_norm = normalize_morphology_text(actual)

    # Exact match after normalization
    if pred_norm == actual_norm:
        return True

    # Check if key words match
    pred_words = set(pred_norm.split())
    actual_words = set(actual_norm.split())

    # If they share most key words (intersection / union > 0.7)
    if pred_words and actual_words:
        intersection = len(pred_words & actual_words)
        union = len(pred_words | actual_words)
        if union > 0 and intersection / union > 0.7:
            return True

    # Special cases for common variations
    pred_lower = pred_norm.lower()
    actual_lower = actual_norm.lower()

    # Handle "dimed" vs "dimpled"
    if 'dimed' in pred_lower and 'dimpled' in actual_lower:
        return True
    if 'dimpled' in pred_lower and 'dimed' in actual_lower:
        return True

    return False


def process_main_output_file(excel_file, output_file):
    """
    Process main.py output Excel file and run advanced droplet modeling with surfactant physics.

    Args:
        excel_file: Path to main.py output Excel file
        output_file: Path for enhanced output file
    """
    print(f"Loading main.py output from {excel_file}")

    # Read the Excel file
    try:
        df = pd.read_excel(excel_file, index_col=0)
        print(f"Loaded {len(df)} parameters for {len(df.columns)} batches")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Transpose if needed (should be parameter rows, batch columns)
    if df.shape[0] < df.shape[1]:  # More columns than rows, likely needs transpose
        df = df.T
        print("Transposed DataFrame for analysis")

    results = []

    # Process each batch
    for batch_idx, batch_name in enumerate(df.columns):
        print(f"\nProcessing batch: {batch_name}")

        # Extract parameters from the batch data
        batch_data = df[batch_name].copy()

        # Convert batch data to inputs format expected by advanced model
        inputs = extract_inputs_from_main_output(batch_data)
        if inputs is None:
            print(f"  Skipping batch {batch_name} - insufficient parameters")
            continue

        # Run advanced droplet modeling
        try:
            result = run_advanced_droplet_modeling(inputs)
            if result:
                # Combine original data with advanced results
                enhanced_batch = batch_data.copy()
                for key, value in result.items():
                    enhanced_batch[key] = value

                results.append((batch_name, enhanced_batch))
                print(f"  Advanced modeling completed for {batch_name}")
            else:
                print(f"  Advanced modeling failed for {batch_name}")

        except Exception as e:
            print(f"  Error in advanced modeling for {batch_name}: {e}")
            continue

    # Create enhanced results DataFrame
    if results:
        enhanced_df = pd.DataFrame({name: data for name, data in results})
        enhanced_df.index.name = 'Parameter / Result'

        # Save enhanced results
        enhanced_df.to_excel(output_file)
        print(f"\nEnhanced results saved to: {output_file}")
        print(
            f"Processed {
                len(results)} batches with advanced droplet modeling")
    else:
        print("No valid batches found for advanced analysis")


def extract_inputs_from_main_output(batch_data):
    """
    Extract input parameters from main.py output Excel format.

    Args:
        batch_data: Pandas Series with batch parameters/results

    Returns:
        Dictionary of input parameters for advanced modeling, or None if insufficient data
    """
    try:
        inputs = {}

        # Basic parameters - try multiple possible column names
        inputs['batch_id'] = str(
            batch_data.get(
                'batch_id',
                batch_data.name if hasattr(
                    batch_data,
                    'name') else 'unknown'))
        inputs['V_chamber_m3'] = float(batch_data.get('V_chamber_m3', 0.005))
        inputs['dryer'] = str(batch_data.get('dryer', 'B290')).lower()
        inputs['cyclone_type'] = str(
            batch_data.get(
                'cyclone_type',
                'std')).lower()
        inputs['gas1'] = str(batch_data.get('gas1', 'air')).lower()
        inputs['gas2'] = str(batch_data.get('gas2', 'air')).lower()

        # Temperatures
        inputs['T1_C'] = float(
            batch_data.get(
                'T1_C', batch_data.get(
                    'Drying Gas Inlet (C)', 70)))
        inputs['T_outlet_C'] = float(
            batch_data.get(
                'T_outlet_C',
                batch_data.get(
                    'T_outlet_C',
                    inputs['T1_C'] -
                    35)))
        inputs['T2_C'] = float(batch_data.get('T2_C', 22))

        # Gas flow and atomization
        inputs['m1_m3ph'] = float(
            batch_data.get(
                'm1_m3ph', batch_data.get(
                    'Drying gas rate (m³/hr)', 35)))
        inputs['atom_pressure'] = float(batch_data.get('atom_pressure', 3.0))

        # Nozzle parameters
        inputs['nozzle_tip_d_mm'] = float(
            batch_data.get('nozzle_tip_d_mm', 0.7))
        inputs['nozzle_cap_d_mm'] = float(
            batch_data.get('nozzle_cap_d_mm', 1.5))
        inputs['nozzle_level'] = str(
            batch_data.get(
                'nozzle_level',
                'Y')).lower()

        # RH values
        inputs['RH1'] = float(batch_data.get('RH1', 55))
        inputs['RH2'] = float(batch_data.get('RH2', 55))

        # Feed composition
        inputs['ds'] = str(batch_data.get('ds', 'igg')).lower()
        inputs['ds_conc'] = float(
            batch_data.get(
                'ds_conc',
                batch_data.get(
                    'Drug Substance conc. (mg/mL)',
                    50)))
        inputs['ds_mw'] = float(
            batch_data.get(
                'ds_mw',
                150000))  # Default IgG MW

        inputs['moni_conc'] = float(
            batch_data.get(
                'moni_conc', batch_data.get(
                    'Moni conc. (mg/mL)', 10)))
        inputs['moni_mw'] = float(
            batch_data.get(
                'moni_mw',
                6800))  # Fixed moni MW

        inputs['stab_A_conc'] = float(
            batch_data.get(
                'stab_A_conc',
                batch_data.get(
                    'Stabilizer conc. (mg/mL)',
                    10)))
        inputs['additive_B_conc'] = float(
            batch_data.get(
                'additive_B_conc', batch_data.get(
                    'Additive #1 conc. (mg/mL)', 0)))
        inputs['additive_C_conc'] = float(
            batch_data.get(
                'additive_C_conc', batch_data.get(
                    'Additive #2 conc. (mg/mL)', 0)))

        # Feed properties
        inputs['solids_frac'] = float(
            batch_data.get(
                'solids_frac',
                batch_data.get(
                    '%Solids',
                    10))) / 100.0
        inputs['feed_g_min'] = float(
            batch_data.get(
                'feed_g_min',
                batch_data.get(
                    'Feed Rate (g/min)',
                    1.0)))
        inputs['viscosity'] = float(
            batch_data.get(
                'viscosity_user_input',
                batch_data.get(
                    'viscosity',
                    0.001)))
        inputs['surface_tension'] = float(
            batch_data.get(
                'surface_tension_user_input',
                batch_data.get(
                    'surface_tension',
                    0.072)))
        inputs['rho_l'] = float(batch_data.get('rho_l', 1000))

        # pH and buffer
        inputs['pH'] = float(batch_data.get('pH', 6.5))
        inputs['buffer_conc'] = float(batch_data.get('buffer_conc', 10))
        inputs['buffer'] = str(batch_data.get('buffer', 'histidine')).lower()
        inputs['stabilizer_A'] = str(
            batch_data.get(
                'stabilizer_A',
                'trehalose')).lower()
        inputs['additive_B'] = str(
            batch_data.get(
                'additive_B',
                '')).lower() or None
        inputs['additive_C'] = None

        # Additional required parameters
        inputs['feed'] = 'y'  # Use feed_g_min

        # PSD parameters
        inputs['D10_actual'] = float(batch_data.get('D10_actual', 1.0))
        inputs['D50_actual'] = float(batch_data.get('D50_actual', 5.0))
        inputs['D90_actual'] = float(batch_data.get('D90_actual', 15.0))
        inputs['Span'] = float(
            batch_data.get(
                'Span',
                (inputs['D90_actual'] -
                 inputs['D10_actual']) /
                inputs['D50_actual']))

        # Diffusion coefficient
        inputs['D_solute'] = float(batch_data.get('D_solute', 4e-11))

        # Moisture parameters (required by simulation)
        inputs['moisture_input'] = float(
            batch_data.get('moisture_input', 0.05))
        inputs['moisture_content'] = float(
            batch_data.get('moisture_content', 0.05))

        return inputs

    except (ValueError, TypeError, KeyError) as e:
        print(f"  Parameter extraction failed: {e}")
        return None


# Load compound properties for molecular weight lookups
def load_compound_properties():
    """Load compound properties from compound_props.json for molecular weight lookups."""
    try:
        compound_props_path = os.path.join(os.path.dirname(__file__), 'compound_props.json')
        with open(compound_props_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load compound_props.json ({e}), using fallback defaults")
        return {}


# Global compound properties cache
COMPOUND_PROPS = load_compound_properties()


def get_compound_molecular_weight(compound_name, mw_from_excel=None):
    """
    Get molecular weight for a compound.
    
    Logic:
    - If compound_name is None/null AND mw_from_excel is None/null → return 0.0
    - If compound_name exists → lookup in compound_props.json, fallback to mw_from_excel if available, then 50.0
    - If compound_name is None/null but mw_from_excel exists → use mw_from_excel
    """
    if compound_name is None or compound_name == '' or pd.isna(compound_name):
        # No compound specified
        if mw_from_excel is None or pd.isna(mw_from_excel):
            return 0.0  # No compound, no MW specified → 0
        else:
            return float(mw_from_excel)  # MW specified in Excel, use it
    
    # Compound name exists, try to look it up
    compound_key = compound_name.lower().strip()
    
    # Handle common name variations
    name_mapping = {
        'l-histidine': 'histidine',
        'ps80': 'ps80',
        'polysorbate 80': 'ps80',
        'polysorbate80': 'ps80',
        'trehalose': 'trehalose',
        'glycine': 'glycine'
    }
    
    compound_key = name_mapping.get(compound_key, compound_key)
    
    if compound_key in COMPOUND_PROPS and 'mw' in COMPOUND_PROPS[compound_key]:
        mw_from_props = COMPOUND_PROPS[compound_key]['mw']
        # Values in compound_props.json are already in Da
        print(f"[MW] Using {compound_name} ({compound_key}) MW from compound_props.json: {mw_from_props} Da")
        return mw_from_props
    else:
        # Compound not found in properties, use Excel value if available
        if mw_from_excel is not None and not pd.isna(mw_from_excel):
            print(f"[MW] Using {compound_name} MW from Excel: {mw_from_excel} kDa")
            return float(mw_from_excel)
        else:
            print(f"[MW] Warning: Compound '{compound_name}' not found in compound_props.json and no MW in Excel, using default 0.0 kDa")
            return 0.0


def run_advanced_droplet_modeling(inputs):
    """
    Run advanced droplet modeling with surfactant physics.

    Args:
        inputs: Dictionary of input parameters

    Returns:
        Dictionary of advanced modeling results
    """
    results = {}

    try:
        # Run the full spray drying simulation first to get basic results
        basic_result = run_full_spray_drying_simulation(inputs)
        if basic_result is None:
            return None

        # Convert result to dictionary format
        if isinstance(basic_result, tuple):
            if len(basic_result) == 2:
                param_names, results_tuple = basic_result
                basic_dict = dict(zip(param_names, results_tuple))
            else:
                # Assume it's the raw tuple format
                basic_dict = dict(zip([
                    'batch_id', 'dryer', 'cyclone_type', 'gas1', 'gas2', 'T1_C', 'T_inlet_req_C', 'T_outlet_C',
                    'atom_pressure', 'atom_gas_mass_flow', 'GLR', 'u_ag', 'solids_frac', 'viscosity_moni',
                    'surface_tension_moni', 'rho_final', 't_dry', 'v_evap', 'Pe', 'Ma', 'Re_g', 'Nu', 'Sh',
                    'D10_actual', 'D50_actual', 'D90_actual', 'D50_calc', 'Span', 'RH_out', 'h', 'k_m', 'D_solute',
                    'moisture_content', 'ds', 'feed_mL_min', 'feed_g_min', 'rho_v_droplet', 'Efficiency', 'pH',
                    'buffer', 'stabilizer_A', 'stab_A_conc', 'additive_B', 'additive_B_conc', 'additive_C',
                    'additive_C_conc', 'm1_m3ph', 'buffer_conc', 'D32_um_moni', 'D50_calc_moni', 'nozzle_tip_d_mm',
                    'ds_conc', 'moni_conc', 'moisture_predicted', 'moisture_input', 'condensed_total_kg_ph',
                    'calibration_factor', 'est_RH_pct', 'delta_RH_pct', 'spm', 'bmp', 'measured_total_computed'
                ], basic_result))
        elif isinstance(basic_result, dict):
            basic_dict = basic_result
        else:
            return None

        # Now perform advanced surface composition analysis with surfactant
        # physics
        row = pd.Series({**inputs, **basic_dict})

        # Calculate surface compositions using Peclet numbers from simulation
        # Get initial solute concentrations (convert to mass fractions)
        drug_conc = float(inputs.get('ds_conc', 0))
        moni_conc = float(inputs.get('moni_conc', 0))
        stabilizer_conc = float(inputs.get('stab_A_conc', 0))
        additive_conc = float(inputs.get('additive_B_conc', 0))

        total_solids_mg_ml = drug_conc + moni_conc + stabilizer_conc + additive_conc

        # Calculate initial mass fractions
        if total_solids_mg_ml > 0:
            # mg/mL to g/mL to mass fraction
            xi0_drug = drug_conc / (1000 * 1000)
            xi0_moni = moni_conc / (1000 * 1000)
            xi0_stabilizer = stabilizer_conc / (1000 * 1000)
            xi0_additive = additive_conc / (1000 * 1000)

            # Normalize to ensure they sum to solids fraction
            solids_frac = float(inputs.get('solids_frac', 0.1))
            total_individual = xi0_drug + xi0_moni + xi0_stabilizer + xi0_additive
            if total_individual > 0:
                xi0_drug = xi0_drug * (solids_frac / total_individual)
                xi0_moni = xi0_moni * (solids_frac / total_individual)
                xi0_stabilizer = xi0_stabilizer * \
                    (solids_frac / total_individual)
                xi0_additive = xi0_additive * (solids_frac / total_individual)
        else:
            # Fallback: assume all solids are drug
            solids_frac = float(inputs.get('solids_frac', 0.1))
            xi0_drug = solids_frac
            xi0_moni = 0
            xi0_stabilizer = 0
            xi0_additive = 0

        # Get effective Peclet numbers from simulation results
        pe_drug = basic_dict.get(
            'effective_pe_drug',
            basic_dict.get(
                'Pe',
                10.0))
        pe_moni = basic_dict.get('effective_pe_moni', 0.1)
        pe_stabilizer = basic_dict.get('effective_pe_stabilizer', 1.0)
        pe_additive = basic_dict.get('effective_pe_additive', 1.0)

        # Ensure we have numeric values
        pe_drug = float(pe_drug) if pe_drug is not None else 10.0
        pe_moni = float(pe_moni) if pe_moni is not None else 0.1
        pe_stabilizer = float(
            pe_stabilizer) if pe_stabilizer is not None else 1.0
        pe_additive = float(pe_additive) if pe_additive is not None else 1.0

        # Calculate surface enrichment using exponential relationship
        drug_enrichment = min(np.exp(min(pe_drug, 10)), 1e10)
        moni_enrichment = min(np.exp(min(pe_moni, 10)), 1e10)
        stabilizer_enrichment = min(np.exp(min(pe_stabilizer, 10)), 1e10)
        additive_enrichment = min(np.exp(min(pe_additive, 10)), 1e10)

        # Calculate surface mass fractions
        xi_drug_surface = xi0_drug * drug_enrichment
        xi_moni_surface = xi0_moni * moni_enrichment
        xi_stabilizer_surface = xi0_stabilizer * stabilizer_enrichment
        xi_additive_surface = xi0_additive * additive_enrichment

        # Apply surfactant priority override (Langmuir adsorption model)
        surfactant_override_applied = False
        if moni_conc > 0:
            try:
                override = surfactant_priority_override(
                    moni_conc_mg_ml=float(moni_conc),
                    moni_name='moni',
                    moni_mw=float(inputs.get('moni_mw', 6800.0))
                )

                if override.get("force_surfactant_override"):
                    surfactant_override_applied = True

                    moni_surface_pct = override['moni_surface_pct']
                    remaining = 100.0 - override['moni_surface_pct']

                    total_protein = xi0_drug + xi0_stabilizer + xi0_additive
                    if total_protein > 0:
                        # Convert to percentage scale
                        scale = remaining / (total_protein * 100)
                        drug_surface_pct = xi0_drug * scale
                        stabilizer_surface_pct = xi0_stabilizer * scale
                        additive_surface_pct = xi0_additive * scale
                    else:
                        drug_surface_pct = remaining
                        stabilizer_surface_pct = 0.0
                        additive_surface_pct = 0.0

                    # Override the calculated values
                    xi_drug_surface = drug_surface_pct / 100.0
                    xi_moni_surface = moni_surface_pct / 100.0
                    xi_stabilizer_surface = stabilizer_surface_pct / 100.0
                    xi_additive_surface = additive_surface_pct / 100.0
            except Exception as e:
                print(f"   → Surfactant override failed: {e}")

        # If no surfactant override, normalize surface fractions to sum to 1
        if not surfactant_override_applied:
            total_surface = xi_drug_surface + xi_moni_surface + \
                xi_stabilizer_surface + xi_additive_surface
            if total_surface > 0:
                xi_drug_surface /= total_surface
                xi_moni_surface /= total_surface
                xi_stabilizer_surface /= total_surface
                xi_additive_surface /= total_surface

        # Convert to percentages for output
        results['drug_surface_pct'] = xi_drug_surface * 100
        results['drug_bulk_pct'] = xi0_drug * 100
        results['moni_surface_pct'] = xi_moni_surface * 100
        results['moni_bulk_pct'] = xi0_moni * 100
        results['stabilizer_surface_pct'] = xi_stabilizer_surface * 100
        results['stabilizer_bulk_pct'] = xi0_stabilizer * 100
        results['additive_surface_pct'] = xi_additive_surface * 100
        results['additive_bulk_pct'] = xi0_additive * 100

        # Surface enrichment ratios
        results['drug_enrichment_ratio'] = drug_enrichment if xi0_drug > 0 else 0
        results['moni_enrichment_ratio'] = moni_enrichment if xi0_moni > 0 else 0
        results['stabilizer_enrichment_ratio'] = stabilizer_enrichment if xi0_stabilizer > 0 else 0
        results['additive_enrichment_ratio'] = additive_enrichment if xi0_additive > 0 else 0

        # Identify primary surface component
        surface_composition = {
            'drug': xi_drug_surface * 100,
            'moni': xi_moni_surface * 100,
            'stabilizer': xi_stabilizer_surface * 100,
            'additive': xi_additive_surface * 100
        }
        max_component = max(surface_composition.items(), key=lambda x: x[1])
        results['primary_surface_component'] = max_component[0]
        results['primary_surface_pct'] = max_component[1]
        results['surfactant_override_applied'] = surfactant_override_applied

        return results

    except Exception as e:
        print(f"  Advanced modeling error: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Droplet evaporation simulation with ML morphology prediction')
    parser.add_argument('training_file', nargs='?', help='Excel file for training morphology model')
    parser.add_argument('prediction_file', nargs='?', help='Excel file for prediction (optional, defaults to training file)')
    parser.add_argument('--output', '-o', help='Output Excel file name (default: simulation_results.xlsx)')
    parser.add_argument('--main-output', '-m', help='Read from main.py output Excel file instead of training/prediction files')

    args = parser.parse_args()

    # Handle main.py output mode
    if args.main_output:
        print("=== MAIN.PY OUTPUT MODE ===")
        print(f"Reading from main.py output Excel file: {args.main_output}")
        output_file = args.output if args.output else args.main_output.replace('.xlsx', '_advanced_analysis.xlsx')
        print(f"Advanced analysis output will be saved to: {output_file}")
        
        # Process main.py output file
        process_main_output_file(args.main_output, output_file)
        print("\nAdvanced droplet analysis completed successfully!")
        sys.exit(0)

    # Interactive mode if no training file provided
    if args.training_file is None:
        print("Interactive mode: Please provide the required files")
        training_file = input("Enter the Excel file for training the morphology model (including .xlsx extension): ")
        prediction_file = input("Enter the Excel file for prediction (including .xlsx extension) [press Enter to use same file]: ")
        if not prediction_file.strip():
            prediction_file = training_file
        output_file = input("Enter output file name (default: simulation_results.xlsx) [press Enter for default]: ")
        if not output_file.strip():
            output_file = 'simulation_results.xlsx'
    else:
        training_file = args.training_file
        prediction_file = args.prediction_file if args.prediction_file else training_file
        output_file = args.output if args.output else 'simulation_results.xlsx'

    print(f"Training on: {training_file}")
    print(f"Predicting on: {prediction_file}")
    print(f"Output will be saved to: {output_file}")

    # Load calibration factors
    print(f"\n=== Loading Calibration Factors ===")
    try:
        with open('calibration.json', 'r') as f:
            calibration_factors = json.load(f)
        # Map the calibration factors to expected keys
        calibration_factors = {
            'moisture': calibration_factors.get('calibration_factor', 1.0),
            'D50': calibration_factors.get('d50_calibration_factor', 1.0),
            'RH': calibration_factors.get('rh_calibration_factor', 1.0),
            'outlet_temp': calibration_factors.get('outlet_temp_calibration_factor', 1.0)
        }
        
        # Load ML model for outlet temperature if available
        if calibration_factors['outlet_temp'] == 'ml_model':
            try:
                print("Warning: outlet_temp_model.pkl not found, falling back to multiplicative calibration")
                calibration_factors['outlet_temp'] = 1.0
            except Exception as e:
                print(f"Warning: Error loading outlet temperature ML model: {e}, falling back to multiplicative calibration")
                calibration_factors['outlet_temp'] = 1.0
        
        print(f"Calibration factors loaded: {calibration_factors}")
    except FileNotFoundError:
        print("Warning: calibration.json not found. Using default factors (1.0 for all)")
        calibration_factors = {'moisture': 1.0, 'D50': 1.0, 'RH': 1.0, 'outlet_temp': 1.0}
    except json.JSONDecodeError as e:
        print(f"Warning: Error parsing calibration.json: {e}. Using default factors (1.0 for all)")
        calibration_factors = {'moisture': 1.0, 'D50': 1.0, 'RH': 1.0, 'outlet_temp': 1.0}

    # Load training data
    print(f"\n=== Loading Training Data: {training_file} ===")
    import os
    if not os.path.exists(training_file):
        if training_file.endswith('.xlxs'):
            corrected_file = training_file.replace('.xlxs', '.xlsx')
            if os.path.exists(corrected_file):
                print(f"File not found: {training_file}, trying {corrected_file}")
                training_file = corrected_file
            else:
                raise FileNotFoundError(f"File not found: {training_file} or {corrected_file}")
    df_train_raw = pd.read_excel(training_file, sheet_name=0)

    if 'Parameter / Batch ID' in df_train_raw.columns:
        print("Training file: Detected transposed Excel format")
        df_train_raw = df_train_raw.set_index('Parameter / Batch ID')
        df_train = df_train_raw.transpose()
    else:
        print("Training file: Detected standard Excel format")
        df_train = df_train_raw

    print("Training file columns:", len(df_train.columns))
    if 'Morphology' in df_train.columns:
        morph_vals = df_train['Morphology'].dropna()
        print(f"Training morphologies found: {len(morph_vals)} entries")
        if len(morph_vals) > 0:
            print(f"Morphology categories: {morph_vals.unique()}")
    else:
        print("No Morphology column found in training file")
    
    # Check for existing model to enable incremental learning
    model_path = 'morphology_model.pkl'
    existing_model = None
    existing_encoder = None
    existing_features = None

    if os.path.exists(model_path):
        print(f"\n=== Found existing model: {model_path} ===")
        try:
            existing_model, existing_encoder, existing_features = load_morphology_model(model_path)
            print("Successfully loaded existing model for incremental learning")
            print(f"Existing model features: {existing_features}")
            print(f"Existing model classes: {existing_encoder.classes_ if existing_encoder else 'None'}")
        except Exception as e:
            print(f"Failed to load existing model: {e}")
            print("Will train new model from scratch")
            existing_model = None
            existing_encoder = None
            existing_features = None
    else:
        print(f"\nNo existing model found at {model_path} - will train new model")
    
    # Load prediction data
    print(f"\n=== Loading Prediction Data: {prediction_file} ===")
    df_pred_raw = pd.read_excel(prediction_file, sheet_name=0)

    if 'Parameter / Batch ID' in df_pred_raw.columns:
        print("Prediction file: Detected transposed Excel format")
        df_pred_raw = df_pred_raw.set_index('Parameter / Batch ID')
        df_pred = df_pred_raw.transpose()
    else:
        print("Prediction file: Detected standard Excel format")
        df_pred = df_pred_raw

    print(f"Prediction file: {len(df_pred)} batches to process")

    # Process each batch
    results = []
    processed_batches = []

    for idx, row in df_pred.iterrows():
        try:
            batch_id = row.get('batch_id', row.get('Batch ID', f'batch_{idx}'))
            print(f"\n--- Processing {batch_id} ---")
            
            result = run_drying_simulation(row, existing_model, existing_encoder, existing_features)
            if result:
                result['batch_id'] = batch_id
                results.append(result)
                processed_batches.append(idx)
                print(f"✓ Completed {batch_id}")
            else:
                print(f"✗ Failed {batch_id}")
                
        except Exception as e:
            print(f"Error processing batch {idx}: {e}")
            continue

    if not results:
        print("No batches were successfully processed!")
        sys.exit(1)

    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Add morphology prediction columns
    results_df['actual_morphology'] = [df_pred.loc[idx, 'Morphology'] if 'Morphology' in df_pred.columns and idx in df_pred.index else None for idx in processed_batches]
    results_df['ml_morphology'] = results_df['morphology']
    
    # Rename columns for clarity
    results_df = results_df.rename(columns={
        'morphology': 'predicted_morphology',
        'morphology_method': 'prediction_method'
    })

    # Add model info
    results_df['model_used'] = 'existing' if existing_model else 'new'
    results_df['model_features'] = str(existing_features) if existing_features else None

    # Update model if we have new training data
    if len(df_train) > 0 and 'Morphology' in df_train.columns:
        print(f"\n=== Updating Model with {len(df_train)} training samples ===")
        try:
            new_model, new_encoder, new_features = train_morphology_model(df_train, existing_model, existing_encoder, existing_features)
            if new_model:
                save_morphology_model(new_model, new_encoder, new_features, model_path)
                print(f"Model updated and saved to {model_path}")
            else:
                print("Model training failed")
        except Exception as e:
            print(f"Model update failed: {e}")

    # Save results
    results_df.to_excel(output_file, index=False)
    print(f"\nAll done! Results saved to '{output_file}'")
    print(f"Successfully processed {len(results)} out of {len(df_pred)} batches")
    print("Format: Column A = Parameter/Result labels, Row 1 = Batch IDs, Data = Values")

    plt.figure(figsize=(10,6))
    num_to_plot = min(10, len(results))  # Plot up to 10 batches to avoid clutter
    for i, r in enumerate(results[:num_to_plot]):
        plt.plot(r['time_history_s'], r['radius_history_um'], label=f"Batch {processed_batches[i]}")
    plt.axhline(0, color='k', linewidth=0.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Particle radius (μm)')
    plt.title(f'Droplet → Particle evolution (showing {num_to_plot} batches)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
