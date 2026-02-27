from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pickle


@dataclass
class LoadedModels:
    d50: object
    moisture: object
    morphology: object
    outlet_temp: object
    rh_outlet: object


def _load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_models(paths) -> LoadedModels:
    return LoadedModels(
        d50=_load_pkl(Path(paths.d50)),
        moisture=_load_pkl(Path(paths.moisture)),
        morphology=_load_pkl(Path(paths.morphology)),
        outlet_temp=_load_pkl(Path(paths.outlet_temp)),
        rh_outlet=_load_pkl(Path(paths.rh_outlet)),
    )
