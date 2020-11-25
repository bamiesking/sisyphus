from sisyphus import Atom
import numpy as np
from scipy.constants import h, c
import subprocess
import os

# n values to iterate over
ns = [2, 3, 4, 5]

# l state symbols
ls = ['s', 'p', 'd', 'f', 'g']

# Replace a label of the form 1s0.5 with a nicely formatted LaTeX version
def format_label(label):
    clean = r''
    pre, post = label.split('.')
    clean += pre[:-1]
    frac = r'_\frac{{{}}}{{2}}'.format(2*int(pre[-1])+1)
    clean += frac
    return clean

# NIST Hydrogen Fine Structure energy levels in inverse cm
# https://physics.nist.gov/PhysRefData/Handbook/Tables/hydrogentable5.htm
nist_data = {
    '1s0.5': 0.0000,
    '2s0.5': 82258.9544,
    '2p0.5': 82258.9191,
    '2p1.5': 82259.2850,
    '3s0.5': 97492.2217,
    '3p0.5': 97492.2112,
    '3p1.5': 97492.3196,
    '3d1.5': 97492.3195,
    '3d2.5': 97492.3556,
    '4s0.5': 102823.8530,
    '4p0.5': 102823.8486,
    '4p1.5': 102823.8943,
    '4d1.5': 102823.8942,
    '4d2.5': 102823.9095,
    '4f2.5': 102823.9095,
    '4f3.5': 102823.9171,
    '5s0.5': 105291.6309,
    '5p0.5': 105291.6287,
    '5p1.5': 105291.6521,
    '5d1.5': 105291.6520,
    '5d2.5': 105291.6599,
    '5f2.5': 105291.6598,
    '5f3.5': 105291.6637,
    '5g3.5': 105291.6637,
    '5g4.5': 105291.6661
}



ground = Atom(1, 0).eigen()[0].mean()
g_I = int(2*0.5+1)

# Initialise energies dictionary for storing calculated values
energies = {'1s0.5': 0}

# Sample and average eigenvalues for each value of n,l,j
prev = 0
for n in ns:
    locs = [0]
    atom = Atom(n, n-1)
    for l in range(n):
        for j in list(set([np.abs(l-0.5), l+0.5])):
            current = prev + g_I*int(2*j+1)
            energies['{}{}{}'.format(n, ls[l], j)] = np.around((np.flip(np.array(atom.eigen()[0]))[prev:current].mean() - ground)/(h*c*1e2), 4)
            prev = current
        prev = 0

# Combine data into a single easy-to-read dictionary
final = {}
for label in list(energies.keys())[1:]:
    final[label] = {
        'nist': nist_data[label],
        'calculated': energies[label],
        'difference': np.around(100*(nist_data[label] - energies[label])/energies[label], 4),
    }


# LaTeX Header
latex = r"""\documentclass{standalone}
\usepackage{siunitx}
\begin{document}
\begin{tabular}{ |ccccc|}
    \hline 
    Energy level & NIST value ($\text{cm}^{-1}$) & Calculated value ($\text{cm}^{-1}$) & Absolute difference ($\text{cm}^{-1}$) & Percentage difference \\ [0.5ex]
    \hline 
    \hline
"""
# Populate LaTeX tabular with data
for label in final.keys():
    data = final[label]
    latex += r"""${}$ & {} & {} & {} & {} \\
\hline
""".format(format_label(label), data['nist'], data['calculated'], np.around(data['nist'] - data['calculated'],4), data['difference'])

# LaTeX Footer
latex += r"""\end{tabular}
\end{document}"""


vals = []
for label in final.keys():
    vals.append(final[label]['difference'])

# Save generated LaTeX to a file
with open('comparison_table.tex', 'w+') as f:
    f.write(latex)

# Compiles LaTeX file and deletes all files but the output .pdf
subprocess.run('pdflatex comparison_table.tex && rm comparison_table.tex comparison_table.log comparison_table.aux', cwd=os.getcwd(), shell=True)












