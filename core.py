"""
    Core functions and classes for simulating Sisyphus cooling

    This file contains the key functions and classes that perform calculations which are relevant to the project in a 
    physics context.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.constants import physical_constants, hbar, c, alpha, h
from .constants import A_hfs, nist_energy_levels
from .helpers import mdot, convert_orbital_number_to_letter, convert_decimal_to_latex_fraction
import time
import warnings
import types


def construct_operator(j: int) -> np.ndarray:
    r"""
        Constructs matrix representation of an arbitrary angular momentum operator J given a j value, using properties of angular 
        momentum ladder operators:

        :math:`\hat{j}_+ |j,m\rangle = \sqrt{j(j+1) - m(m+1)} |j,m+1\rangle`
        :math:`\hat{j}_- = \hat{j}_+^\dagger`

        :math:`\hat{j}_x &= \frac{1}{2}\left(\hat{j}_+ + \hat{j}_-\right)`
        :math:`\hat{j}_y &= -\frac{i}{2}\left(\hat{j}_+ - \hat{j}_-\right)`
        :math:`\hat{j}_z &= \frac{1}{2}\left(\hat{j}_+\hat{j}_- - \hat{j}_-\hat{j}_+\right)`

        :math:`\vec{\hat{j}} = \hat{j}_x\vec{i} + \hat{j}_y\vec{j} + \hat{j}_z\vec{k}`

        where :math:`\vec{i},\vec{j},\vec{k}` are unit vectors in the :math:`x,y,z` directions respectively.

        Args: 
            j: The value of the quantum number of the angular momentum operator J to be constructed.

        Returns:
            Array of x, y, and z operators.

    """

    # Enumerate m_j values (z projection quantum number)
    m = [i for i in np.arange(-j,j+1)]

    # Construct j_+ operator element-wise
    j_plus = np.zeros((len(m), len(m)))
    for ix,iy in np.ndindex(j_plus.shape):
        if m[ix] == m[iy]-1:
            j_plus[ix,iy] = np.sqrt(j*(j+1) - m[ix]*(m[ix]+1))

    # Construct j_- operator
    j_minus = j_plus.T

    # Construct j_x, j_y, j_z operators
    j_x = 0.5*(j_plus + j_minus)
    j_y = -0.5j*(j_plus - j_minus)
    j_z = 0.5*(np.dot(j_plus,j_minus) - np.dot(j_minus,j_plus))

    return np.array([j_x, j_y, j_z])

class Field():
    """
        Represents a magnetic field with a given field strength profile as a function of space.

        Args:
            profile: A numpy array of python lambda functions describing the field behaviour in each spacial dimesion

    """

    def __init__(self, profile: np.ndarray):
        self.profile = profile

    def fieldStrength(self, r: np.ndarray) -> np.ndarray:
        """
            Returns the field strength at a point in space.

            Args:
                r: A numpy array representing a position vector.

            Returns:
                A numpy array representing vector field strength.
        """
        return np.array([self.profile[i](r[i]) for i in range(0,3)])


class Atom():
    """
        Represents a hydrogen atom with a given position in an applied magnetic field

        Provides a number of methods for simulating the energy levels of an atom in an applied magnetic field 
        for given quantum numbers.

        Args:
            n (int): The principal quantum number of the atom.
            l (int): The orbital angular momentum quantum number of the atom.
            position (np.ndarray, optional): The position of the atom in 3D space.
            B_field (Field, optional): The applied magnetic field.
            E_field (Field, optional): The applied electric field.
            energy_offset: (float, optional): An amount by which to offset the eigenenergies of the atom.
            energy_scaling: (float, optional): An amount by which to scale the eigenenergies of the atom.

        Attributes:
            n: The principal quantum number.
            l: The orbital angular momentum quantum number.
            s: The electron spin quantum number.
            i: The nuclear spin quantum number.
            j (list): The minmum and maximum possible total electron angular momentum quantum numbers.
            f (functionType): The minimum and maximum possible total atomic angular momentum quantum numbers as a function of j.
            S (np.ndarray): The electron spin angular momentum operator.
            I (np.ndarray): The nuclear spin angular momentum operator.
            L (np.ndarray): The electron orbital angular momentum operator.
            J (np.ndarray): The total electron angular momentum operator.
            F (np.ndarray): The total atomic angular momentum operator.

        Raises:
            ValueError: If l is not between 0 and n-1 inclusive.

    """

    def __init__(self, n, l=0, position=np.array([0, 0, 0]), B_field=Field(np.array([lambda x: 0, lambda y : 0, lambda z : 0])), E_field=Field(np.array([lambda x: 0, lambda y : 0, lambda z : 0])), energy_offset=0, energy_scaling=1):

        # Threshold proximity for two lines to be swapped
        self.epsilon = 1e-28*energy_scaling

        # Store position
        self.position = position

        # Store fields
        self.B_field = B_field
        self.E_field = E_field

        # Store energy adjustments
        self.energy_offset = energy_offset
        self.energy_scaling = energy_scaling

        # Set quantum numbers
        self.n = n

        if not 0 <= l <= n-1:
            raise ValueError('l must be between 0 and n-1 ({})'.format(self.n-1))

        self.l = l
        self.s = 0.5
        self.i = 0.5

        # Create operators
        self.L = np.kron(np.kron(construct_operator(self.l), np.identity(int(2*self.i + 1))), np.identity(int(2*self.i + 1)))
        self.S = np.kron(np.kron(np.identity(int(2*self.l+1)), construct_operator(self.s)), np.identity(int(2*self.i + 1)))
        self.I = np.kron(np.kron(np.identity(int(2*self.l+1)), np.identity(int(2*self.s + 1))), construct_operator(self.i))
        self.J = self.L + self.S
        self.F = self.J + self.I

        self.dim = self.L.shape[1]

    @property
    def Hamiltonian(self):
        """
            np.ndarray: The Hamiltonian of the atom at the current position and field.
        """

        self._Hamiltonian = A_hfs[self.n]*mdot(self.J, self.I) + self.magneticDipoleInteraction + self.electricDipoleInteraction
        return self._Hamiltonian
    

    @property
    def H0(self):
        self._H0 = np.full((self.dim, self.dim), 0+0j, dtype=np.longdouble)

        prev = 0
        for label, i in zip(nist_data.keys(), range(len(nist_data.keys()))):
            if label[0:2] == '{}{}'.format(self.n, get_orbital_symbol([self.l]).lower()):
                j = float(label[2:])
                dim = int(2*j+1)*int(2*self.i+1)
                current = prev + dim
                value = -1*nist_data[label]*(h*c*1e2)
                self._H0[prev:current, prev:current] = value*np.identity(dim)
                prev = current
        return self._H0
    

    @property
    def magneticDipoleInteraction(self):
        self._magneticDipoleInteraction = np.tensordot(physical_constants['Bohr magneton'][0]*(self.L - physical_constants['electron g factor'][0]*self.S) + physical_constants['nuclear magneton'][0]*self.I, self.B_field.fieldStrength(self.position), axes=((0),(0)))
        return self._magneticDipoleInteraction

    @property
    def electricDipoleInteraction(self):
        # Placeholder interaction - to be properly implemented later
        self._electricDipoleInteraction = np.tensordot(np.zeros(self.L.shape), self.E_field.fieldStrength(self.position), axes=((0), (0)))
        return self._electricDipoleInteraction

    @property
    def position(self):
        """
            np.ndarray: The current position of the atom.
        """
        return self._position
    
    @position.setter
    def position(self, position):
        self._position = position


    def eigen(self, position=None, coarse=True):
        """
            Returns the eigenenergies and eigenstates of the atom in its current configuration.

            Args:
                position (np.ndarray): The position of the atom for which the eigenenergies and eigenstates should be evaluated.
        """
        if position is not None:
            self._position = position
        
        raw = np.linalg.eigh(self.Hamiltonian)
        if coarse:
            raw = [raw[0] + np.diag(self.H0), raw[1]]

        # Rescale and offset energies
        adjusted = ([(r+energy_offset)*energy_scaling for r in raw[0]], raw[1])
        return adjusted

    def eigen_range(self, x):
        # Store initial position
        position_init = self.position

        # Set up array to store energies
        eigens = np.zeros((x.size, self.dim), dtype=np.longdouble)

        # Generate energies
        for i,j in zip(x, range(x.size)):
            self.position = np.array([i, i, i])
            eigens[j] = self.eigen(coarse=False)[0]

        # Reset position
        self.position = position_init

        # Fix eigenvalue ordering
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                for k in range(len(x)):
                    if np.abs(eigens[k, i] - eigens[k, j]) < self.epsilon:
                        eigens[k+1:,i], eigens[k+1:,j] = eigens[k+1:,j], eigens[k+1:,i].copy()
                        break
        order = np.argsort(eigens[int(len(x)/20),:])[::-1]

        eigens += np.diag(self.H0)

        return eigens[:, order]

    def plotZeemanEnergyShift(self, n):
        """
            Plots the Zeeman energy shift of the hyperfine levels in the atom.

            Args:
                n (np.ndarray): An array of discrete positions at which to evaluate Zeeman shifts
                ax (matplotlib.pyplot.Axis): The axis object on which to plot the lines
        """

        eigens = self.eigen_range(n)
        for i in range(4):
            print(eigens[:,i].max())

        lines = []
        for i in range(self.dim):
            points = np.array([n, eigens[:,i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lines.append(LineCollection(segments))

        return lines

    def calculateStateMixing(self, n):

        # Store initial position
        position_init = self.position


        # Set up array to store energies
        energies = np.zeros((n.size, self.dim))
        states = np.full((n.size, self.dim, self.dim), 0+0j)

        # Generate energies
        for i,j in zip(n, range(n.size)):
            self.position = np.array([i, i, i])
            eig = self.eigen()
            energies[j] = eig[0]/A_hfs[1]
            states[j] = eig[1]

        # Fix eigenvalue ordering
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                for k in range(len(n)):
                    if np.abs(energies[k, i] - energies[k, j]) < self.epsilon:
                            energies[k+1:,i], energies[k+1:,j] = energies[k+1:,j], energies[k+1:,i].copy()
                            states[k+1:,:,i], states[k+1:,:,j] = states[k+1:,:,j], states[k+1:,:,i].copy()
                            break

        # Reset position
        self.position = position_init

        states = np.absolute(states)
        return states


