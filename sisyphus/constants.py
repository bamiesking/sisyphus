"""
    Key physical constants.

    This file contains key physical constants which are not available as part of scipy.constants or included in 
    the ARC module.

    Attributes:
        A_fs (float): The hydrogen fine structure constant.
        A_hfs (float): The hydrogen hyperfine structure constant.
"""

from scipy.constants import h

A_hfs = 1420405751.8*h/4


# NIST Hydrogen Fine Structure energy levels in inverse cm
# https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable5.htm
nist_data = {
    '1s0.5': 0.0000,
    '2s0.5': 82258.9544,
    '2p0.5': 82258.9191,
    '2p1.5': 82259.2850,
    '3s0.5': 97492.2217,
    '3p0.5': 97492.2112,
    '3p1.5': 97492.3196,
    '3d1.5': 97492.3195,
    '3d2.5': 97492.3556,
    '4s0.5': 102823.8530,
    '4p0.5': 102823.8486,
    '4p1.5': 102823.8943,
    '4d1.5': 102823.8942,
    '4d2.5': 102823.9095,
    '4f2.5': 102823.9095,
    '4f3.5': 102823.9171,
    '5s0.5': 105291.6309,
    '5p0.5': 105291.6287,
    '5p1.5': 105291.6521,
    '5d1.5': 105291.6520,
    '5d2.5': 105291.6599,
    '5f2.5': 105291.6598,
    '5f3.5': 105291.6637,
    '5g3.5': 105291.6637,
    '5g4.5': 105291.6661
}