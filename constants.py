"""
    Key physical constants.

    This file contains key physical constants which are not available as part of scipy.constants or included in 
    the ARC module.

    Attributes:
        A_fs (float): The hydrogen fine structure constant.
        A_hfs (float): The hydrogen hyperfine structure constant.
"""

from scipy.constants import h

# Parthey - 'Precision spectroscopy on atomic hydrogen' thesis
A_hfs = {1: 1420405751.7667*h, 2: 177556834.3*h}


# NIST Hydrogen Fine Structure energy levels in inverse cm
# https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable5.htm
nist_energy_levels = {
    '1S0.5': 0.0000,
    '2S0.5': 82258.9544,
    '2P0.5': 82258.9191,
    '2P1.5': 82259.2850,
    '3S0.5': 97492.2217,
    '3P0.5': 97492.2112,
    '3P1.5': 97492.3196,
    '3D1.5': 97492.3195,
    '3D2.5': 97492.3556,
    '4S0.5': 102823.8530,
    '4P0.5': 102823.8486,
    '4P1.5': 102823.8943,
    '4D1.5': 102823.8942,
    '4D2.5': 102823.9095,
    '4F2.5': 102823.9095,
    '4F3.5': 102823.9171,
    '5S0.5': 105291.6309,
    '5P0.5': 105291.6287,
    '5P1.5': 105291.6521,
    '5D1.5': 105291.6520,
    '5D2.5': 105291.6599,
    '5F2.5': 105291.6598,
    '5F3.5': 105291.6637,
    '5G3.5': 105291.6637,
    '5G4.5': 105291.6661
}


# NIST Hydrogen Persistant Lines
# https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable3.htm
nist_decay_rates = {
    '5P1.5 -> 1S0.5': 0.34375e8,
    '5P0.5 -> 1S0.5': 0.34375e8,
    '4P1.5 -> 1S0.5': 0.6819e8,
    '4P0.5 -> 1S0.5': 0.6819e8,
    '3P1.5 -> 1S0.5': 1.6725e8,
    '3P0.5 -> 1S0.5': 1.6725e8,
    '2P1.5 -> 1S0.5': 6.2648e8,
    '2P0.5 -> 1S0.5': 5.2649e8,
    '4D1.5 -> 2P0.5': 0.1719e8,
    '4P1.5 -> 2S0.5': 0.0967e8,
    '4D2.5 -> 2P1.5': 0.2063e8,
    '3D1.5 -> 2P0.5': 0.5388e8,
    '3P1.5 -> 2S0.5': 0.2245e8,
    '3D2.5 -> 2P1.5': 0.6465e8
}
