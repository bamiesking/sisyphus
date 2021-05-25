from .wigner import Wigner3j, Wigner6j
from sympy import Symbol, symbols, diff, exp, integrate, oo
from sympy.physics.hydrogen import R_nl


# Dunning & Hulet, Ch. 9
def BranchingRatio(l1, s1, j1, i1, f1, m1, l2, s2, j2, i2, f2, m2, k, q):
    factors = [
        2*f1+1,
        2*f2+1,
        2*j1+1,
        2*j2+1,
        2*l1+1,
        Wigner6j(l2, j2, s1, j1, l1, k)**2,
        Wigner6j(j2, f2, i1, f1, j1, k)**2,
        Wigner3j(f2, f1, k, m2, -1*m1, q)**2
    ]
    value = 1
    for factor in factors:
        value *= factor
    return value


@precalc('intR')
def radialIntegral(n1, l1, n2, l2):
    r = Symbol('r')
    integrand = R_nl(n2, l2, r)*R_nl(n1, l1, r)*r**3
    return integrate(integrand, (r, 0, oo))


@precalc('<d>')
def ElectricDipoleTransitionMatrixElement(n1, l1, s1, j1, i1, f1, m1, n2, l2, s2, j2, i2, f2, m2, q):
    ratio = BranchingRatio(l1, s1, j1, i1, f1, m1, l2, s2, j2, i2, f2, m2, 1, q)
    d2 = radialIntegral(n1, l1, n2, l2)
    return e*d2*np.sqrt(ratio)


@precalc('alpha_E1')
def ElectricDipolePolarizability(polarization, applied_frequency, n1, l1, s1, j1, i1, f1, m1, n_max=5):
    terms = []
    s2, i2 = 0.5, 0.5
    for n2 in range(1, n_max+1):
        if n2 != n1:
            for l2 in range(n2):
                for j2 in np.arange(np.abs(l2-s2), l2+s2+1, 1):
                    freq = (2*pi*c)/transition_wavelength(n1, l1, j1, n2, l2, j2)
                    for f2 in np.arange(np.abs(j2-i2), j2+i2+1, 1, dtype=int):
                        for m2 in np.arange(-1*f2, f2, 1, dtype=int):
                            for polarization in [-1, 0, 1]:
                                terms.append((freq*(ElectricDipoleTransitionMatrixElement(n1, l1, s1, j1, i1, f1, m1, n2, l2, s2, j2, i2, f2, m2, polarization)))/(freq**2 - applied_frequency**2))
    val = 2*np.abs(np.array(terms)).sum()/hbar
    return val

@precalc('npbr')
def NPhotonBranchingRatio(l1, s1, j1, i1, f1, m1, l2, s2, j2, i2, f2, m2, n_max=10):
    terms = []
    # s3, i3 = 0.5, 0.5
    for n3 in range(1, n_max+1):
        for l3 in range(n3):
            for s3 in [-0.5, 0.5]:
                for j3 in np.arange(np.abs(l3-s3), l3+s3+1, 1):
                    for i3 in [-0.5, 0.5]:
                        for f3 in np.arange(np.abs(j3-i3), j3+i3+1, 1, dtype=int):
                            for m3 in np.arange(-1*f3, f3+1, 1, dtype=int):
                                for polarization1 in [-1., 0., 1.]:
                                    for polarization2 in [-1., 0., 1.]:
                                        terms.append(BranchingRatio(l1, s1, j1, i1, f1, m1, l3, s3, j3, i3, f3, m3, 1, polarization1) *
                                                     BranchingRatio(l3, s3, j3, i3, f3, m3, l2, s2, j2, i2, f2, m2, 1, polarization2))
    val = np.abs(np.array(terms)).sum()
    return val
