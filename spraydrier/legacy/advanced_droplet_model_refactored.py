# -*- coding: utf-8 -*-

"""
advanced_droplet_model_refactored.py
Refactored version with improved structure, dead code removed, and better validation.

Predicts particle morphology, surface composition, and moisture for spray-dried formulations.
"""

import os
import sys
import json
import argparse
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project path setup
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

warnings.filterwarnings("ignore", category=UserWarning, module="scipy.optimize")

# Scientific computing imports
import pandas as pd
import numpy as np
import scipy as sp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Custom modules (commented out for standalone operation)
# from models import predict_moisture_content
# from simulation import run_full_spray_drying_simulation
# from models.adsorption_model import surfactant_priority_override


class InputValidator:
    """Validates input parameters for physical reasonableness."""
    
    @staticmethod
    def validate_process_conditions(inlet_temp: float, feed_rate: float, 
                                   gas_rate: float) -> Dict[str, Any]:
        """Validate basic process conditions."""
        issues = []
        
        if not (30 <= inlet_temp <= 300):
            issues.append(f"Inlet temperature {inlet_temp}Â°C outside typical range (30-300Â°C)")
        
        if not (0.1 <= feed_rate <= 10):
            issues.append(f"Feed rate {feed_rate} g/min outside typical range (0.1-10 g/min)")
            
        if not (10 <= gas_rate <= 200):
            issues.append(f"Gas rate {gas_rate} mÂ³/hr outside typical range (10-200 mÂ³/hr)")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': [issue for issue in issues if 'outside typical' in issue]
        }
    
    @staticmethod
    def validate_concentrations(drug_conc: float, moni_conc: float, 
                              stabilizer_conc: float, additive_conc: float) -> Dict[str, Any]:
        """Validate component concentrations."""
        issues = []
        
        if drug_conc < 0:
            issues.append("Drug concentration cannot be negative")
        if moni_conc < 0:
            issues.append("Monoclonal antibody concentration cannot be negative")
        if stabilizer_conc < 0:
            issues.append("Stabilizer concentration cannot be negative")
        if additive_conc < 0:
            issues.append("Additive concentration cannot be negative")
            
        total_conc = drug_conc + moni_conc + stabilizer_conc + additive_conc
        if total_conc > 500:  # mg/mL
            issues.append(f"Total concentration {total_conc:.1f} mg/mL unusually high")
            
        return {
            'valid': len(issues) == 0,
            'issues': issues
        }


class PecletCalculator:
    """Handles Peclet number calculations with proper validation."""
    
    @staticmethod
    def calculate_peclet_numbers(row: pd.Series) -> Dict[str, float]:
        """Calculate Peclet numbers from pre-computed effective values or defaults."""
        
        # Use pre-calculated effective Peclet numbers from main.py
        # This is the correct approach per the code comments
        pe_drug = PecletCalculator._safe_float_conversion(
            row.get('effective_pe_drug', row.get('Pe_igg', 0.5))  # Low surface activity for drug
        )
        pe_moni = PecletCalculator._safe_float_conversion(
            row.get('effective_pe_moni', row.get('Pe_moni', 15.0))  # High surface activity for amphiphilic surfactant
        )
        pe_stabilizer = PecletCalculator._safe_float_conversion(
            row.get('effective_pe_stabilizer', 1.5)  # Increased from 1.0
        )
        pe_additive = PecletCalculator._safe_float_conversion(
            row.get('effective_pe_additive', 1.5)  # Increased from 1.0
        )
        
        # Apply safety caps to prevent numerical overflow
        # Cap at 20 instead of 50 for better numerical stability
        pe_drug = min(pe_drug, 20.0)
        pe_moni = min(pe_moni, 20.0)
        pe_stabilizer = min(pe_stabilizer, 20.0)
        pe_additive = min(pe_additive, 20.0)
        
        return {
            'pe_drug': pe_drug,
            'pe_moni': pe_moni,
            'pe_stabilizer': pe_stabilizer,
            'pe_additive': pe_additive
        }
    
    @staticmethod
    def _safe_float_conversion(value: Any) -> float:
        """Safely convert value to float with error handling."""
        try:
            return float(value) if value is not None else 0.0
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {value} to float, using 0.0")
            return 0.0


class SurfaceCompositionCalculator:
    """Calculates surface composition using physics-based enrichment."""
    
    @staticmethod
    def calculate_surface_enrichment(bulk_fractions: Dict[str, float], 
                                   peclet_numbers: Dict[str, float]) -> Dict[str, float]:
        """Calculate surface enrichment using exponential relationship."""
        
        # Calculate enrichment factors with safety limits
        max_enrichment = 100.0  # Cap enrichment to prevent extreme dominance
        
        enrichment_factors = {}
        for component in ['drug', 'moni', 'stabilizer', 'additive']:
            pe = peclet_numbers.get(f'pe_{component}', 0.0)
            # Cap Pe before exponentiation for numerical stability
            pe_capped = min(pe, 5.0)  # exp(5) ≈ 148, reasonable limit
            enrichment_factors[component] = min(np.exp(pe_capped), max_enrichment)
        
        # Calculate surface mass fractions
        surface_fractions = {}
        for component in ['drug', 'moni', 'stabilizer', 'additive']:
            bulk_frac = bulk_fractions.get(f'xi_{component}', 0.0)
            enrichment = enrichment_factors[component]
            surface_fractions[f'xi_{component}_surface'] = bulk_frac * enrichment
        
        # Normalize to ensure sum = 1
        total_surface = sum(surface_fractions.values())
        if total_surface > 0:
            for key in surface_fractions:
                surface_fractions[key] /= total_surface
        
        return surface_fractions


class MorphologyPredictor:
    """Handles morphology prediction using machine learning."""
    
    def __init__(self):
        self.model = None
        self.encoder = None
        self.feature_columns = []
        
    def load_model(self, model_path: str = 'morphology_model.pkl') -> bool:
        """Load pre-trained morphology model from file."""
        try:
            import pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.encoder = model_data.get('label_encoder') or model_data.get('encoder')
                self.feature_columns = model_data.get('features', model_data.get('feature_columns', []))
                
                if self.model and self.encoder:
                    logger.info(f"Loaded pre-trained morphology model with {len(self.feature_columns)} features")
                    return True
                else:
                    logger.error("Model or encoder missing from saved data")
                    return False
            else:
                logger.error("Invalid model file format")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load morphology model: {e}")
            return False
        
    def get_feature_columns(self) -> List[str]:
        """Get the list of feature columns for ML model."""
        return [
            # Basic process parameters
            'Drying Gas Inlet (C)', 'Drying gas rate (mÂ³/hr)', 'Drug Substance conc. (mg/mL)',
            'Moni conc. (mg/mL)', 'Feed solution pH', 'Stabilizer conc. (mg/mL)',
            'Additive #1 conc. (mg/mL)', '%Solids', 'D50_actual', 'D10_actual', 'D90_actual',
            
            # Physically relevant parameters
            'Feed Rate (g/min)', 'Surface recession velocity', 'Max Peclet Number',
            'Effective Peclet Number', 'Peclet Number', 'Integrated Peclet Number',
            'Heat Transfer coefficient', 'Mass Transfer coefficient', 'Reynolds number',
            'Marangoni Number', 'Estimated feed viscosity (PaÂ·s)', 'Estimated Feed Surface Tension (N/m)',
            
            # Moisture parameters
            'Measured total moisture (%)', 'Powder total moisture content (calculated)', 
            'Measured bound moisture (%)',
            
            # Surface composition features
            'Drug surface %', 'Moni surface %', 'Stabilizer surface %', 'Additive surface %',
            'Drug bulk %', 'Moni bulk %', 'Stabilizer bulk %', 'Additive bulk %',
            
            # Effective Peclet numbers
            'effective_pe_moni', 'effective_pe_drug', 'effective_pe_stabilizer', 
            'effective_pe_additive_b', 'effective_pe_additive_c', 'effective_pe_buffer'
        ]
    
    def train_model(self, df: pd.DataFrame, test_size: float = 0.2, 
                   cv_folds: int = 5) -> Dict[str, Any]:
        """Train morphology prediction model with proper validation."""
        
        if 'Morphology' not in df.columns:
            logger.warning("No Morphology column found in data")
            return {'success': False, 'message': 'No morphology data available'}
        
        # Feature selection
        available_features = [col for col in self.get_feature_columns() if col in df.columns]
        
        # Check for sufficient numeric features
        numeric_features = []
        for col in available_features:
            try:
                numeric_data = pd.to_numeric(df[col], errors='coerce')
                if numeric_data.notna().sum() >= 3:
                    numeric_features.append(col)
            except Exception:
                continue
        
        if len(numeric_features) < 3:
            return {'success': False, 'message': f'Insufficient numeric features: {len(numeric_features)}'}
        
        # Check sample size vs features
        n_samples = len(df)
        if n_samples / len(numeric_features) < 3:
            logger.warning(f"Low sample-to-feature ratio: {n_samples}/{len(numeric_features)} = {n_samples/len(numeric_features):.1f}")
        
        logger.info(f"Training with {len(numeric_features)} features and {n_samples} samples")
        
        # Prepare data
        X = df[numeric_features].copy()
        y = df['Morphology'].copy()
        
        # Handle missing values
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(X[col].mean())
        
        y = y.fillna(y.mode()[0] if len(y.mode()) > 0 else 'spheres')
        
        # Encode labels
        self.encoder = LabelEncoder()
        y_encoded = self.encoder.fit_transform(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5,  # Limit depth to prevent overfitting
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.feature_columns = numeric_features
        
        # Validation
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y_encoded, cv=cv_folds)
        
        # Performance metrics
        y_pred = self.model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = dict(zip(numeric_features, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        results = {
            'success': True,
            'train_accuracy': train_score,
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_features': len(numeric_features),
            'n_samples': n_samples,
            'top_features': top_features,
            'class_distribution': dict(zip(self.encoder.classes_, 
                                         np.bincount(y_encoded)))
        }
        
        logger.info(f"Model trained successfully: Test accuracy = {test_accuracy:.3f}")
        logger.info(f"Cross-validation: {cv_scores.mean():.3f} Â± {cv_scores.std():.3f}")
        
        return results
    
    def predict_morphology(self, features: Dict[str, float]) -> Dict[str, Any]:
        """Predict morphology for given features."""
        
        if self.model is None or self.encoder is None:
            return {'prediction': None, 'confidence': 0.0, 'method': 'none'}
        
        # Prepare feature vector
        feature_vector = []
        for col in self.feature_columns:
            feature_vector.append(features.get(col, 0.0))
        
        feature_array = np.array(feature_vector).reshape(1, -1)
        
        # Make prediction
        prediction = self.model.predict(feature_array)[0]
        probabilities = self.model.predict_proba(feature_array)[0]
        
        morphology = self.encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        return {
            'prediction': morphology,
            'confidence': confidence,
            'method': 'ml',
            'probabilities': dict(zip(self.encoder.classes_, probabilities))
        }


class DropletSimulator:
    """Main class for droplet simulation and prediction."""
    
    def __init__(self):
        self.validator = InputValidator()
        self.peclet_calc = PecletCalculator()
        self.surface_calc = SurfaceCompositionCalculator()
        self.morphology_predictor = MorphologyPredictor()
    
    def simulate_single_batch(self, row: pd.Series) -> Dict[str, Any]:
        """Simulate a single batch with comprehensive error handling."""
        
        try:
            # Input validation
            inlet_temp = float(row.get('Drying Gas Inlet (C)', 70.0))
            feed_rate = float(row.get('Feed Rate (g/min)', 1.0))
            gas_rate = float(row.get('Drying gas rate (mÂ³/hr)', 35.0))
            
            validation_result = self.validator.validate_process_conditions(
                inlet_temp, feed_rate, gas_rate
            )
            
            if not validation_result['valid']:
                logger.warning(f"Validation issues: {validation_result['issues']}")
            
            # Calculate concentrations and bulk fractions
            def safe_float(value, default=0.0):
                try:
                    result = float(value)
                    return result if not np.isnan(result) else default
                except (ValueError, TypeError):
                    return default
            
            drug_conc = safe_float(row.get('ds_conc', row.get('Drug Substance conc. (mg/mL)', 0.0)))
            moni_conc = safe_float(row.get('moni_conc', row.get('Moni conc. (mg/mL)', 0.0)))
            stabilizer_conc = safe_float(row.get('stab_A_conc', row.get('Stabilizer conc. (mg/mL)', 0.0)))
            additive_conc = safe_float(row.get('additive_B_conc', row.get('Additive #1 conc. (mg/mL)', 0.0)))
            
            conc_validation = self.validator.validate_concentrations(
                drug_conc, moni_conc, stabilizer_conc, additive_conc
            )
            
            if not conc_validation['valid']:
                logger.warning(f"Concentration issues: {conc_validation['issues']}")
            
            # Calculate bulk fractions (assuming unit molecular weights for simplicity)
            total_mass = drug_conc + moni_conc + stabilizer_conc + additive_conc
            if total_mass > 0:
                bulk_fractions = {
                    'xi_drug': drug_conc / total_mass,
                    'xi_moni': moni_conc / total_mass,
                    'xi_stabilizer': stabilizer_conc / total_mass,
                    'xi_additive': additive_conc / total_mass
                }
            else:
                bulk_fractions = {'xi_drug': 0, 'xi_moni': 0, 'xi_stabilizer': 0, 'xi_additive': 0}
            
            # Calculate Peclet numbers
            peclet_numbers = self.peclet_calc.calculate_peclet_numbers(row)
            
            # Calculate surface composition
            surface_composition = self.surface_calc.calculate_surface_enrichment(
                bulk_fractions, peclet_numbers
            )
            
            # Predict morphology
            # Create feature dict for ML model
            feature_dict = dict(row)
            feature_dict.update(surface_composition)
            feature_dict.update(peclet_numbers)
            
            morphology_result = self.morphology_predictor.predict_morphology(feature_dict)
            
            # Compile results
            result = {
                'inlet_temp_c': inlet_temp,
                'feed_rate_g_min': feed_rate,
                'gas_rate_m3_hr': gas_rate,
                'total_solids_mg_ml': total_mass,
                
                # Bulk composition
                'drug_bulk_pct': bulk_fractions['xi_drug'] * 100,
                'moni_bulk_pct': bulk_fractions['xi_moni'] * 100,
                'stabilizer_bulk_pct': bulk_fractions['xi_stabilizer'] * 100,
                'additive_bulk_pct': bulk_fractions['xi_additive'] * 100,
                
                # Surface composition
                'drug_surface_pct': surface_composition.get('xi_drug_surface', 0) * 100,
                'moni_surface_pct': surface_composition.get('xi_moni_surface', 0) * 100,
                'stabilizer_surface_pct': surface_composition.get('xi_stabilizer_surface', 0) * 100,
                'additive_surface_pct': surface_composition.get('xi_additive_surface', 0) * 100,
                
                # Peclet numbers
                'pe_drug': peclet_numbers['pe_drug'],
                'pe_moni': peclet_numbers['pe_moni'],
                'pe_stabilizer': peclet_numbers['pe_stabilizer'],
                'pe_additive': peclet_numbers['pe_additive'],
                
                # Morphology prediction
                'predicted_morphology': morphology_result['prediction'],
                'morphology_confidence': morphology_result['confidence'],
                'morphology_method': morphology_result['method'],
                
                # Validation flags
                'process_validation_passed': validation_result['valid'],
                'concentration_validation_passed': conc_validation['valid'],
                'validation_warnings': validation_result.get('warnings', []) + conc_validation.get('warnings', [])
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            return {
                'error': str(e),
                'success': False
            }
    
    def process_dataframe(self, df: pd.DataFrame, train_morphology: bool = True) -> pd.DataFrame:
        """Process entire dataframe with optional morphology model training."""
        
        logger.info(f"Processing {len(df)} batches...")
        
        # Train morphology model if requested and data available
        if train_morphology and 'Morphology' in df.columns:
            training_result = self.morphology_predictor.train_model(df)
            if training_result['success']:
                logger.info(f"Morphology model trained successfully")
                logger.info(f"Features used: {training_result['n_features']}")
                logger.info(f"Test accuracy: {training_result['test_accuracy']:.3f}")
            else:
                logger.warning(f"Morphology training failed: {training_result.get('message', 'Unknown error')}")
        elif not train_morphology:
            # Load pre-trained model
            load_success = self.morphology_predictor.load_model()
            if load_success:
                logger.info("Pre-trained morphology model loaded successfully")
            else:
                logger.warning("Failed to load pre-trained morphology model")
        
        # Process each batch
        results = []
        for i, (idx, row) in enumerate(df.iterrows()):
            logger.info(f"Processing batch {i+1}/{len(df)} (ID: {idx})")
            result = self.simulate_single_batch(row)
            result['batch_index'] = idx
            results.append(result)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Add actual morphology if available
        if 'Morphology' in df.columns:
            results_df['actual_morphology'] = df['Morphology'].values
            
            # Calculate accuracy if predictions were made
            if 'predicted_morphology' in results_df.columns:
                predictions = results_df['predicted_morphology'].dropna()
                actuals = results_df.loc[predictions.index, 'actual_morphology']
                if len(predictions) > 0:
                    accuracy = (predictions == actuals).mean()
                    logger.info(f"Overall prediction accuracy: {accuracy:.3f}")
        
        return results_df


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Advanced Droplet Model - Refactored Version')
    parser.add_argument('input_file', help='Input Excel file with batch data')
    parser.add_argument('-o', '--output', default='simulation_results.xlsx', 
                       help='Output Excel file')
    parser.add_argument('--no-training', action='store_true',
                       help='Skip morphology model training')
    
    args = parser.parse_args()
    
    try:
        # Load data
        logger.info(f"Loading data from {args.input_file}")
        df = pd.read_excel(args.input_file)
        
        # If data is transposed (Parameter / Batch ID format), transpose it
        if 'Parameter / Batch ID' in df.columns:
            df = df.set_index('Parameter / Batch ID').transpose()
            df.index.name = 'batch_id'
        
        logger.info(f"Loaded {len(df)} batches with {len(df.columns)} parameters")
        
        # Initialize simulator
        simulator = DropletSimulator()
        
        # Process data
        train_morphology = not args.no_training
        results_df = simulator.process_dataframe(df, train_morphology=train_morphology)
        
        # Save results
        results_df.to_excel(args.output, index=False)
        logger.info(f"Results saved to {args.output}")
        
        # Print summary
        print("\n=== SIMULATION SUMMARY ===")
        print(f"Processed batches: {len(results_df)}")
        print(f"Successful simulations: {(~results_df['error'].notna()).sum()}")
        
        if 'predicted_morphology' in results_df.columns:
            morphology_counts = results_df['predicted_morphology'].value_counts()
            print(f"\nPredicted morphologies:")
            for morph, count in morphology_counts.items():
                print(f"  {morph}: {count}")
        
        if 'actual_morphology' in results_df.columns and 'predicted_morphology' in results_df.columns:
            predictions = results_df['predicted_morphology'].dropna()
            actuals = results_df.loc[predictions.index, 'actual_morphology']
            if len(predictions) > 0:
                accuracy = (predictions == actuals).mean()
                print(f"\nPrediction accuracy: {accuracy:.1%}")
        
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise


if __name__ == "__main__":
    main()