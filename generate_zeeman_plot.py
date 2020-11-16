from sisyphus import BField, Atom, A_hfs, get_orbital_symbol
from scipy.constants import physical_constants
import numpy as np
import matplotlib.pyplot as plt


# Define B field 
profile = np.array([lambda x: 0,
                    lambda y: 0,
                    lambda z: z*A_hfs/physical_constants['Bohr magneton'][0]])
B = BField(profile)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.15)

n = np.arange(0, 1, 0.01)
ax.set_prop_cycle(color=['blue'])
atom1 = Atom(2, 0, B_field=B)
ax = atom1.plotZeemanEnergyShift(n, ax)


ax.set_prop_cycle(color=['red'])
atom2 = Atom(2, 1, B_field=B)
ax = atom2.plotZeemanEnergyShift(n, ax)


# Setup plot
ax.set_title(r'Zeeman shifts for $^1H{{{n}}}^{{{S}}}{{{l}}}$ levels'
             .format(n=atom1.n,
                     l=get_orbital_symbol(atom1.l),
                     S=int(2*atom1.s+1)))

ax.set_xlabel(r'$\frac{\mu_B B}{A_{hfs}}$', fontsize=12)
ax.set_ylabel(r'$\frac{E}{A_{hfs}}$', fontsize=12, rotation=0)
ax.autoscale()

plt.show()
