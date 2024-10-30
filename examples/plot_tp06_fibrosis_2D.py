"""
This script runs a 2D simulation of cardiac tissue
with 20% fibrosis using the TP06 model.
Fibrosis is simulated as mesh nodes with values = 2.
You can generate it manulally
or use one of the predefined patterns (like Diffuse2DPattern).
The simulation results are visualized as a potential map.
"""

import finitewave as fw
import matplotlib.pyplot as plt
import numpy as np

# number of nodes on the side
n = 300

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n])

# generate 20% of fibrosis in the tissue:
fibrosis_pattern = fw.Diffuse2DPattern(0, n, 0, n, 0.20)
fibrosis_pattern.generate(tissue.mesh.shape, tissue.mesh)

tissue.add_boundaries()

# create model object:
tp06 = fw.TP062D()
# set up numerical parameters:
tp06.dt = 0.01
tp06.dr = 0.25
tp06.t_max = 40

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, -40, 0, int(n * 0.03), 0, n))
# add the tissue and the stim parameters to the model object:
tp06.cardiac_tissue = tissue
tp06.stim_sequence = stim_sequence

tp06.run()

# show the potential map at the end of calculations:
plt.imshow(tp06.u)
plt.show()
