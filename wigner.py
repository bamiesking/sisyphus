"""
    Provides a functions to enable calculation of Wigner 3j and 6j symbols
"""


import numpy as np


# Define alias for factorial function
fac = np.math.factorial


def triangle_coefficient(a, b, c):
    r"""
        Calculates the triangle coefficient:

            :math:`\Delta (a, b, c)`

        According to https://mathworld.wolfram.com/TriangleCoefficient.html
    """

    # Calculate numerator factors:
    num1 = fac(a + b - c)
    num2 = fac(b + c - a)
    num3 = fac(c + a - b)
    num = num1*num2*num3

    # Calculate denominator
    denom = fac(a + b + c + 1)

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
                term *= fac(i)

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

    for t in range(int(tmin), int(tmax) + 1):

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
                term *= fac(i)

            # Add inverse of product of factorials to the total. Note (-1)**t is equivalent to (-1)**(-t).
            total += fac(t+1)/term
        t += 1
    return total


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
    if not triangular_inequalities((j1, j2, j)):
        return 0
    elif not (j1+j2+j) % 1 == 0:
        return 0
    elif not m1 + m2 == -1*m:
        return 0
    return (-1)**(j1 - j2 + m) * \
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
    if not all([sum([i for i in triad]) % 1 == 0 for triad in triads]):
        return 0

    return t_sum_6j(j1, j2, j3, J1, J2, J3) * \
           np.sqrt(
                triangle_coefficient(j1, j2, j3) *
                triangle_coefficient(j1, J2, J3) *
                triangle_coefficient(J1, j2, J3) *
                triangle_coefficient(J1, J2, j3)
           )
