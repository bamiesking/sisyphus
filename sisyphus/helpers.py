"""
    Provides a number of methods to support classes defined in .methods.

    Functions defined here are not crucial to the physics the package is trying to simulate
"""

import numpy as np
import functools



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

    if max(l) > 20:
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
             }[max(l)]
    return symbol

def convert_decimal_to_latex_fraction(d):
    sign = ''
    if d < 0:
        sign = '-'
    return r'{{{sign}}}\frac{{{e}}}{{2}}'.format(sign=sign, e=int(2*d))


def triangle_coefficient(a, b, c):
    r"""
        Calculates the triangle coefficient:

            :math:`\Delta (a, b, c)`

        According to https://mathworld.wolfram.com/TriangleCoefficient.html
    """

    # Calculate numerator factors:
    num1 = np.math.factorial(a + b - c)
    num2 = np.math.factorial(b + c - a)
    num3 = np.math.factorial(c + a - b)
    num = num1*num2*num3

    # Calculate denominator
    denom = np.math.factorial(a + b + c + 1)

    return num/denom

def t_sum_3j(j1, j2, j, m1, m2, m):
    # Initialise variables
    total, t, counter = 0, 0, 0

    tmin = np.array([
        j2 - j - m1,
        j1 + m2 - j,
        0
    ]).max()

    tmax = np.array([
        j1 + j2 - j,
        j1 - m1,
        j2 + m2
    ]).min()

    for t in range(tmin, tmax + 1):

        # Construct arguments of factorials to be tested
        factorial_arguments = [
            t,
            j - j2 + t + m1,
            j - j1 + t - m2,
            j1 + j2 - j - t,
            j1 - t - m1,
            j2 - t + m2
        ]

        term = (-1)**t

        # Test whether all factorial arguments are valid 
        if all([i >= 0 for i in factorial_arguments]):
            # Calculate factorials for all terms
            for i in factorial_arguments:
                term *= np.math.factorial(i)

            # Add inverse of product of factorials to the total. Note (-1)**t is equivalent to (-1)**(-t).
            total += 1/term
        t += 1
    return total


def t_sum_6j(j1, j2, j3, J1, J2, J3):
    # Initialise variables
    total, t, counter = 0, 0, 0

    tmin = np.array([
        j1 + j2 + j3,
        j1 + J2 + J3,
        J1 + j2 + J3,
        J1 + J2 + J3
    ]).max()

    tmax = np.array([
        j1 + j2 + J1 + J2,
        j2 + j3 + J2 + J3,
        j3 + j1 + J3 + J1
    ]).min()

    for t in range(tmin, tmax + 1):

        # Construct arguments of factorials to be tested
        factorial_arguments = [
            t - j1 - j2 - j3,
            t - j1 - J2 - J3,
            t - J1 - j2 - J3,
            t - J1 - J2 - j3,
            j1 + j2 + J1 + J2 - t,
            j2 + j3 + J2 + J3 - t,
            j3 + j1 + J3 + J1 - t
        ]

        term = (-1)**t

        # Test whether all factorial arguments are valid 
        if all([i >= 0 for i in factorial_arguments]):
            # Calculate factorials for all terms
            for i in factorial_arguments:
                term *= np.math.factorial(i)

            # Add inverse of product of factorials to the total. Note (-1)**t is equivalent to (-1)**(-t).
            total += np.math.factorial(t+1)/term
        t += 1
    return total


def determine_symmetries(j1, j2, j, m1, m2, m, symbol):
    """
        Determines symmetries of Wigner symbols
    """
    if symbol == '3j':
        equal = [
            format_label(j2, j, j1, m2, m, m1),
            format_label(j, j1, j2, m, m1, m1)
        ]

        prefactor = [
            format_label(j2, j1, j, m2, m1, m),
            format_label(j1, j, j2, m1, m, m2),
            format_label(j, j2, j1, m, m2, m1),
            format_label(j1, j2, j, -1*m1, -1*m2, -1*m)
        ]

        return {1: equal, (-1)**(j1+j2+j): prefactor}
    elif symbol == '6j':

        # Redefine symbols for clarity
        j3, J1, J2, J3 = j, m1, m2, m

        equal = [
            format_label(j2, j1, j3, J2, J1, J3),
            format_label(j3, j1, j2, J3, J1, J2),
            format_label(J1, J2, j3, j1, j2, J3),
            format_label(J1, j2, J3, j1, J2, j3),
            format_label(j1, J2, J3, J1, j2, j3)
        ]

        return {1: equal}


def format_label(j1, j2, j, m1, m2, m):
    return str(j1) + \
           str(j2) + \
           str(j) + \
           str(m1) + \
           str(m2) + \
           str(m)


def precalc(symbol):
    def precalc_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            symbols = {'3j': {}, '6j': {}}
            try:
                with open('symbol_storage.pickle', 'rb') as f:
                    symbols = pickle.load(f)
            except:
                pass

            label = format_label(*args)
            if label in symbols[symbol]:
                return symbols[symbol][label]
            
            value = func(*args, **kwargs)
            symmetries = determine_symmetries(*args, symbol)
            symbols[symbol][label] = value
            for prefactor in symmetries.keys():
                for sym_label in symmetries[prefactor]:
                    symbols[symbol][sym_label] = prefactor*value
            with open('symbol_storage.pickle', 'wb+') as f:
                pickle.dump(symbols, f)

            return value 
        return wrapper
    return precalc_decorator


def triangular_inequalities(tup):
    x, y, z = tup
    return np.abs(x - y) <= z <= x + y


@precalc('3j')
def Wigner3j(j1, j2, j, m1, m2, m):
    r"""
        Calculates the Wigner-3j symbol using the Racah formula:

            :math:`\begin{pmatrix} j_1 & j_2 & j \\ m_1 & m_2 & m \end{pmatrix}`

        According to https://mathworld.wolfram.com/Wigner3j-Symbol.html
    """
    return (-1)**(j1 - j2 - m) * \
           np.sqrt(
                triangle_coefficient(j1, j2, j) * \
                fac(j1 + m1) *
                fac(j1 - m1) *
                fac(j2 + m2) *
                fac(j2 - m2) *
                fac(j + m) *
                fac(j - m)
           ) * \
           t_sum_3j(j1, j2, j, m1, m2, m)


@precalc('6j')
def Wigner6j(j1, j2, j3, J1, J2, J3):
    triads = [
        (j1, j2, j3),
        (j1, J2, J3),
        (J1, j2, J3),
        (J1, J2, j3)
    ]

    # Ensure triads obey triangular inequalities
    if not all([triangular_inequalities(triad) for triad in triads]):
        return 0

    # Ensure triads sum to an integer
    if not all([isinstance(sum([i for i in triad]) % 1, int) for triad in triads]):
        return 0

    return t_sum_6j(j1, j2, j3, J1, J2, J3) * \
           np.sqrt(
                triangle_coefficient(j1, j2, j3) *
                triangle_coefficient(j1, J2, J3) *
                triangle_coefficient(J1, j2, J3) *
                triangle_coefficient(J1, J2, j3)
           )






