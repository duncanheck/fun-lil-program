# constants.py

import numpy as np
import pandas as pd

# Physical constants
R_UNIVERSAL = 8.31446261815324
M_AIR = 28.97e-3
M_N2 = 28.0134e-3
R_AIR = R_UNIVERSAL / M_AIR
R_N2 = R_UNIVERSAL / M_N2
R_v = 461.5
C_p_water = 4186
C_p_vapor = 1850

# Temperature grid
_temps_C = np.arange(-110, 91, 5)
_temps_K = [T + 273.15 for T in _temps_C]

# Property functions
def _mu_suth(T, C1, S):
    return C1 * T**1.5 / (T + S)

def _k_pow(T, k_ref):
    return k_ref * (T / 293.15)**0.76

def _D_pow(T, D_ref):
    return D_ref * (T / 293.15)**1.75

# Air DataFrame
_air = pd.DataFrame({
    "Temperature_C": _temps_C,
    "Temperature_K": [round(T, 2) for T in _temps_K],
    "k_g": [_k_pow(T, 0.02624) for T in _temps_K],
    "mu_g": [_mu_suth(T, 1.458e-6, 110.4) for T in _temps_K],
    "Pr_g": [0.707 for T in _temps_K],
    "D_v": [_D_pow(T, 2.1e-5) for T in _temps_K],
})

# Nitrogen DataFrame
_n2 = pd.DataFrame({
    "Temperature_C": _temps_C,
    "Temperature_K": [round(T, 2) for T in _temps_K],
    "k_g": [_k_pow(T, 0.02583) for T in _temps_K],
    "mu_g": [_mu_suth(T, 1.66e-6, 111.0) for T in _temps_K],
    "Pr_g": [0.72 for T in _temps_K],
    "D_v": [_D_pow(T, 2.0e-5) for T in _temps_K],
})