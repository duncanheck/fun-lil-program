from spraydrier.engine import SprayDryerEngine, TrialInput
from spraydrier.config import ModelPaths
from pathlib import Path

def test_engine_basic():
    models = ModelPaths(
        d50=Path("models/d50_model.pkl"),
        moisture=Path("models/moisture_model.pkl"),
        morphology=Path("models/morphology_model.pkl"),
        outlet_temp=Path("models/outlet_temp_model.pkl"),
        rh_outlet=Path("models/rh_outlet_model.pkl"),
    )

    engine = SprayDryerEngine(models)

    params = {
        "inlet_temp": 150,
        "outlet_temp": 80,
        "solids_frac": 0.1,
        "feed_rate": 3,
        "gas_flow": 35,
        "droplet_d50": 30e-6,
        "formulation": {
            "BSA": 0.5,
            "trehalose": 0.5
        }
    }

    trial = TrialInput("smoke", params)
    result = engine.run_trial(trial)

    assert result.ok
