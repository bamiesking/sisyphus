"""
    Provides a number of methods to support classes defined in .methods.

    Functions defined here are not crucial to the physics the package is trying to simulate
"""

import numpy as np

def mdot(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
        Performs elementwise matrix multiplication on two arrays of square matrixes (NxN arrays).

        Args:
            A: An array of NxN arrays.
            B: An array of NxN arrays.

        Returns:
            Sum of elementwise matrix multiplication of A and B as an NxN array.

        Raises:
            ValueError: If the shapes of A and B are not equal.

    """

    # Ensure matrices are same shape
    if A.shape != B.shape:
        raise ValueError('Matrix array shape mismatch: can\'t dot {} with {}'.format(A.shape[1:], B.shape[1:]))

    M = np.zeros(A.shape[1:])
    for a,b in zip(A,B):
        M = M + np.dot(a,b)
    return M


def get_orbital_symbol(l: int) -> str:
    """
        Returns the letter corresponding to a particular value of orbital angular momentum 
        quantum number l for l <= 20.

        Args:
            l: The orbital angular momentum quantum number.

        Returns:
            The corresponding spectroscopic notation symbol for the oribital angular momentum quantum number.

        Raises:
            ValueError: If l > 20
    """

    if l > 20:
        raise ValueError('l cannot be greater than 20')

    symbol = { 0: 'S', 
               1: 'P', 
               2: 'D',
               3: 'F',
               4: 'G',
               5: 'H',
               6: 'I',
               7: 'K',
               8: 'L',
               9: 'M',
               10: 'N',
               11: 'O',
               12: 'Q',
               13: 'R',
               14: 'T',
               15: 'U',
               16: 'V',
               17: 'W',
               18: 'X',
               19: 'Y',
               20: 'Z'
             }[l]
    return symbol

def convert_decimal_to_latex_fraction(d):
    sign = ''
    if d < 0:
        sign = '-'
    return r'{{{sign}}}\frac{{{e}}}{{2}}'.format(sign=sign, e=int(2*d))
