"""
    Core functions and classes for simulating Sisyphus cooling

    This file contains the key functions and classes that perform calculations which are relevant to the project in a 
    physics context.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.constants import physical_constants, hbar, c, alpha
from .constants import A_hfs
from .helpers import mdot, get_orbital_symbol, convert_decimal_to_latex_fraction
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

class BField():
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
            B_field (BField, optional): The applied magnetic field.

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

    def __init__(self, n, position=np.array([0, 0, 0]), B_field=BField(np.array([lambda x: 0, lambda y : 0, lambda z : 0]))):

        # Store position
        self.position = position

        # Store field
        self.B_field = B_field

        # Set quantum numbers
        self.n = n
        self.l = [0, self.n]
        self.s = 0.5
        self.i = 0.5
        self.j = lambda l: [np.abs(l-self.s), l + self.s]
        self.f = lambda j : [np.abs(j - self.i), j + self.i]

        # Create operators


        degeneracies = [(int(2*l)+1)*(int(2*self.s)+1)*(int(2*self.i)+1) for l in range(self.n)]
        self.dim = np.array(degeneracies).sum()
        print(self.dim)
        print(degeneracies)
        self.L = np.full((3, self.dim, self.dim), 0+0j)
        self.S = np.full((3, self.dim, self.dim), 0+0j)
        self.I = np.full((3, self.dim, self.dim), 0+0j)
        for l in range(self.n):
            if l == 0:
                prev = 0
            else:
                prev = degeneracies[l-1] - 1
            current = degeneracies[l] + prev
            print(l, prev, current)
            print(self.L[:,prev:current, prev:current].shape, np.kron(np.kron(construct_operator(l),np.identity(int(2*self.s)+1)),np.identity(int(2*self.i)+1)).shape)
            self.L[:,prev:current, prev:current] = np.kron(np.kron(construct_operator(l),np.identity(int(2*self.s)+1)),np.identity(int(2*self.i)+1))
            self.S[:,prev:current, prev:current] = np.kron(np.kron(np.identity(int(2*l)+1),construct_operator(self.s)),np.identity(int(2*self.i)+1))
            self.I[:,prev:current, prev:current] = np.kron(np.kron(np.identity(int(2*l)+1),np.identity(int(2*self.s)+1)),construct_operator(self.i))

        self.J = self.S + self.L 
        self.F = self.J + self.I

        #self.generate_hamiltonian(B_field)
    

    @property
    def Hamiltonian(self):
        """
            np.ndarray: The Hamiltonian of the atom at the current position and field.
        """
        self.En = -1*physical_constants['Rydberg constant times hc in J'][0]/(self.n**2) 

        # if self.l > 0:
        #     self.A_fs = (-1*self.En*(alpha**2)/(self.n))*(1/(self.l*(self.l + 0.5)*(self.l+1)))
        # else:
        #     self.A_fs = 0
        self.A_fs = []

        for l in range(self.n):
            for i in range((int(2*l)+1)*(int(2*self.s)+1)*(int(2*self.i)+1)):
                if l > 0:
                    self.A_fs.append((-1*self.En*(alpha**2)/(self.n))*(1/(l*(l + 0.5)*(l+1))))
                else:
                    self.A_fs.append(0)
        self.A_fs = np.array(self.A_fs)
        print(self.A_fs/A_hfs)
        # print(np.dot(self.A_fs.T,mdot(self.L, self.S)))
        # self.A_fs = np.full(self.A_fs.shape, (-1*self.En*(alpha**2)/(self.n))*(1/((self.n-1)*((self.n-1) + 0.5)*((self.n-1)+1))))

        self.En_corrections = self.En #+ self.RelativisticTerms + self.LambShift
        self.H0 = self.En_corrections*np.identity(self.dim)
        self._MagneticInteraction = lambda r : np.tensordot(physical_constants['Bohr magneton'][0]*(self.L - physical_constants['electron g factor'][0]*self.S) + physical_constants['nuclear magneton'][0]*self.I, self.B_field.fieldStrength(r), axes=((0),(0)))
        self._Hamiltonian = lambda r : self.H0 + np.dot(self.A_fs.T,mdot(self.L, self.S)) #+ A_hfs*mdot(self.I, self.J) + self._MagneticInteraction(r)
        return self._Hamiltonian(self.position)

    @property
    def RelativisticTerms(self):
        """
            float: The Darwin term of the current energy level.
        """
        prefactor = (-1*self.En*alpha**2)/(self.n**2)
        denom = 0.5#self.l + 0.5
        # if self.l == 0:
        #     denom = 1
        self._RelativisticTerms = prefactor*(0.75 - self.n/denom)

        return self._RelativisticTerms
    

    @property
    def LambShift(self):
        """
            float: The Lamb shift of the current energy level.
        """
        self._LambShift = 0
        # if self.l == 0:
        #     self._LambShift = ((alpha**5)*physical_constants['electron mass'][0]*(c**2)/(6*np.pi))*np.log(1/(np.pi*alpha))
        return self._LambShift
    

    @property
    def position(self):
        """
            np.ndarray: The current position of the atom.
        """
        return self._position
    
    @position.setter
    def position(self, position):
        self._position = position


    def eigen(self, position=None):
        """
            Returns the eigenenergies and eigenstates of the atom in its current configuration.

            Args:
                position (np.ndarray): The position of the atom for which the eigenenergies and eigenstates should be evaluated.
        """
        if position is not None:
            self._position = position
        
        return np.linalg.eigh(self.Hamiltonian)

    def plotZeemanEnergyShift(self, n):
        """
            Plots the Zeeman energy shift of the hyperfine levels in the atom.

            Args:
                n (np.ndarray): An array of discrete positions at which to evaluate Zeeman shifts
                ax (matplotlib.pyplot.Axis): The axis object on which to plot the lines
        """

        # Store initial position
        position_init = self.position


        # Set up array to store energies
        eigens = np.zeros((n.size, self.dim))

        # Generate energies
        for i,j in zip(n, range(n.size)):
            self.position = np.array([i, i, i])
            eigens[j] = self.eigen()[0]/A_hfs

        # Reset position
        self.position = position_init


        # Fix eigenvalue ordering
        epsilon = 5e-27/A_hfs # Threshold proximity for two lines to be swapped
        for i in range(self.dim):
            for j in range(i+1, self.dim):
                for k in range(len(n)):
                    if np.abs(eigens[k, i] - eigens[k, j]) < epsilon:
                            eigens[k+1:,i], eigens[k+1:,j] = eigens[k+1:,j], eigens[k+1:,i].copy()
                            break

        lines = []
        # Create a continuous norm to map from data points to colors

        for i in range(self.dim):
            points = np.array([n, eigens[:,i]]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lines.append(LineCollection(segments))

        return lines



