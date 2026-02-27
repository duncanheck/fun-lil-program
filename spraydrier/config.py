# spraydrier/config.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

@dataclass
class ModelPaths:
    """File paths to the trained ML models (.pkl)."""
    d50: Path
    moisture: Path
    morphology: Path
    outlet_temp: Path
    rh_outlet: Path


@dataclass
class CalibrationConfig:
    """
    Calibration factors and mode switches loaded from calibration.json.
    All fields have safe defaults so the engine can always run.
    """
    moisture_factor: float = 1.0
    rh_factor: float = 1.0
    outlet_temp_factor: float = 1.0
    d50_factor: float = 1.0          # Added: missing from your current version

    # ML mode switches (can override physics-only predictions)
    use_ml_for_d50: bool = True
    use_ml_for_morphology: bool = True
    use_ml_for_outlet_temp: bool = True
    use_ml_for_rh: bool = True

    # Optional future extensions
    use_ml_for_surface: bool = False
    surface_moni_enrichment_factor: float = 1.0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalibrationConfig':
        """Create from dict (e.g., loaded JSON) with safe defaults."""
        return cls(
            moisture_factor=float(data.get("moisture_factor", 1.0)),
            rh_factor=float(data.get("rh_factor", 1.0)),
            outlet_temp_factor=float(data.get("outlet_temp_factor", 1.0)),
            d50_factor=float(data.get("d50_factor", 1.0)),
            use_ml_for_d50=bool(data.get("use_ml_for_d50", True)),
            use_ml_for_morphology=bool(data.get("use_ml_for_morphology", True)),
            use_ml_for_outlet_temp=bool(data.get("use_ml_for_outlet_temp", True)),
            use_ml_for_rh=bool(data.get("use_ml_for_rh", True)),
            use_ml_for_surface=bool(data.get("use_ml_for_surface", False)),
            surface_moni_enrichment_factor=float(data.get("surface_moni_enrichment_factor", 1.0)),
        )


def load_calibration(calibration_path: Path | str | None = "calibration.json") -> CalibrationConfig:
    """
    Load calibration.json into a CalibrationConfig.
    If missing, malformed, or incomplete → returns safe defaults.
    Logs warnings but never raises exceptions (fail-safe).
    """
    if calibration_path is None:
        logger.debug("No calibration path provided → using defaults")
        return CalibrationConfig()

    p = Path(calibration_path)
    if not p.exists():
        logger.warning(f"Calibration file not found: {p} → using defaults")
        return CalibrationConfig()

    try:
        raw_data = json.loads(p.read_text(encoding="utf-8"))
        config = CalibrationConfig.from_dict(raw_data)
        logger.debug(f"Loaded calibration from {p}: {config}")
        return config
    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON in {p}: {e} → using defaults")
    except Exception as e:
        logger.warning(f"Error loading calibration from {p}: {e} → using defaults")

    return CalibrationConfig()