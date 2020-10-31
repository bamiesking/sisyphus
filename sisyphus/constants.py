"""
    Key physical constants.

    This file contains key physical constants which are not available as part of scipy.constants or included in 
    the ARC module.

    Attributes:
        A_fs (float): The hydrogen fine structure constant.
        A_hfs (float): The hydrogen hyperfine structure constant.
"""

from scipy.constants import h, e, pi, epsilon_0, hbar, c

A_fs = e**2/(4*pi*epsilon_0*hbar*c)
A_hfs = 1420405751.8*h/4