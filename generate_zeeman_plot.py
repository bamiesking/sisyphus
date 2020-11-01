from sisyphus import BField, Atom, A_hfs
from scipy.constants import physical_constants
import numpy as np

profile = np.array([lambda x: 0, lambda y: 0, lambda z: z*A_hfs/physical_constants['Bohr magneton'][0]]) 
B = BField(profile)
atom = Atom(1, 0, B_field=B)
nums = [[0.5, -0.5], [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5]]

atom.showZeemanEnergyShift(nums=nums)