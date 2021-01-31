from sisyphus import Field, Atom, A_hfs, get_orbital_symbol
from scipy.constants import physical_constants
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# Define B field 
profile = np.array([lambda x: 0,
                    lambda y: 0,
                    lambda z: z*A_hfs[1]/physical_constants['Bohr magneton'][0]])
B = Field(profile)

n = np.arange(0, 1, 0.001)
atom1 = Atom(2, B_field=B)
mixing = atom1.calculateStateMixing(n)
vals = np.full((n.size, mixing.shape[-1]), 0+0j)

fig, axs = plt.subplots(4, 4, sharex=True, sharey=True)
fig.subplots_adjust(hspace=0.5, bottom=0.15, left=0.15)
axs = axs.flatten()

def moving_average(x, w):
    if w == 1:
        return x
    else:
        return np.convolve(x, np.ones(w), 'valid') / w

titles = [
    '1,1',
    '1,0',
    '1,-1',
    '0,0',
    '1,1',
    '1,0',
    '1,-1',
    '0,0',
    '2,2',
    '2,1',
    '2,0',
    '2,-1',
    '2,-2',
    '1,1',
    '1,0',
    '1,-1'
]

average_window = 4

custom_lines = [Line2D([0], [0], color='k', lw=1),
                Line2D([0], [0], color='r', lw=1)]

for i in range(mixing.shape[-1]):
    ax = axs[i]
    for j in range(mixing.shape[-1]):
        vals[:,j] = np.einsum('ij,kj->i', mixing[:, i, :], mixing[:, j, :])
    vals = np.divide(vals, np.linalg.norm(vals, axis=1)[:,None])
    ax.plot(n[:1-average_window], moving_average((vals[:,0:4]**2).sum(axis=1), average_window), label='s', color='k')
    ax.plot(n[:1-average_window], moving_average((vals[:,4:]**2).sum(axis=1), average_window), label='p', color='r')
    ax.set_title(r'$|{{{}}}\rangle$'.format(titles[i]), fontsize='9')


# Setup plot
plt.suptitle(r's and p state mixing for $n=2$, $|F, m_F\rangle$ states')

fig.legend(custom_lines, ['s', 'p'])

fig.text(0.09, 0.8, '2s', fontsize='9')
fig.text(0.09, 0.6, '2p', fontsize='9')
fig.text(0.09, 0.41, '2p', fontsize='9')
fig.text(0.09, 0.23, '2p', fontsize='9')
fig.text(0.08, 0.92, 'n={}'.format(len(n)), fontsize='9')

# Labels
fig.text(0.5, 0.04, r'$\frac{\mu_B B}{A_{hfs}^{(n=2)}}$', ha='center', fontsize='12')
fig.text(0.04, 0.5, r'Probability of measurement in s or p state', va='center', rotation='vertical', fontsize='10')

plt.show()
