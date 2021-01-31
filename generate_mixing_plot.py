from sisyphus import Field, Atom, A_hfs, get_orbital_symbol
from scipy.constants import physical_constants
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Specify which line we want to determine mixing in relation to:
i = 2 # n=2, l=0, F=1, mF=0 line

# Define B field 
profile = np.array([lambda x: 0,
                    lambda y: 0,
                    lambda z: z*A_hfs[1]/physical_constants['Bohr magneton'][0]])
B = Field(profile)

fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(bottom=0.15)

n = np.arange(0, 1, 0.01)
atom1 = Atom(2, B_field=B)
lines = atom1.plotZeemanEnergyShift(n)


# atom2 = Atom(2, 1, B_field=B)
# lines += atom2.plotZeemanEnergyShift(n)

mixing = atom1.calculateStateMixing(n)
vals = np.full((n.size, mixing.shape[-1]), 0)
norm = plt.Normalize(0, 1)

for j in range(mixing.shape[-1]):
    vals[:,j] = mixing[:, j, i]
    # vals[:,j] = np.einsum('ij,kj->i', mixing[:, i, :], mixing[:, j, :])
    # vals[:,j] = np.tensordot(mixing[0, i, :], mixing[:,j,:], axes=(0, 1))
# vals = np.divide(vals, np.linalg.norm(vals, axis=1)[:,None])


for line,j in zip(lines, range(len(lines))):
    if j == i:
        line.set_color('r')
    else:
        line.set_cmap('viridis')
        line.set_norm(norm)
        line.set_array((vals[:, j]**2))
        ax.add_collection(line)

line = lines[0]
fig.colorbar(line, ax=ax)

# lines = ax.get_lines()
# lines[2].set_color('k')


# Setup plot
# ax.set_title(r'Zeeman shifts for $^1H{{{n}}}^{{{S}}}{{{l}}}$ levels'
#              .format(n=atom1.n,
#                      l=get_orbital_symbol(atom1.l),
#                      S=int(2*atom1.s+1)))

ax.set_xlabel(r'$\frac{\mu_B B}{A_{hfs}}$', fontsize=12)
ax.set_ylabel(r'$\frac{E}{A_{hfs}}$', fontsize=12, rotation=0)
ax.autoscale()

plt.show()
