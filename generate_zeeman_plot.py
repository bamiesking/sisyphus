from sisyphus import Field, Atom, A_hfs, get_orbital_symbol
from scipy.constants import physical_constants
import numpy as np
import matplotlib.pyplot as plt


# Define B field 
profile = np.array([lambda x: 0,
                    lambda y: 0,
                    lambda z: z*A_hfs[1]/physical_constants['Bohr magneton'][0]])
B = Field(profile)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.15)

n = 2
x = np.arange(0, 0.025*1, 0.0001)
for l in range(n-1, n):
    atom = Atom(n, l, B_field=B)
    lines = atom.plotZeemanEnergyShift(x)
    for line, j in zip(lines, range(len(lines))):
        ax.add_collection(line)

ax.set_xlabel(r'$\frac{\mu_B B}{A_{hfs}}$', fontsize=12)
ax.set_ylabel(r'$\frac{E}{A_{hfs}}$', fontsize=12, rotation=0)
ax.autoscale()

plt.show()
