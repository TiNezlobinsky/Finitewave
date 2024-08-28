
#
# You can apply matrix area stimulation using StimVoltageMatrix2D.
# stim_area - a boolean matrix of 0 (non-activated) and 1 (activated) points.
#


import matplotlib.pyplot as plt
from skimage import draw
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 400

tissue = fw.CardiacTissue2D([n, n], mode='aniso')
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n])
tissue.add_boundaries()
# add numeric method stencil for weights computations
tissue.D_al = 1

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 30
# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_area = np.full([400, 400], False, dtype=bool)
ii, jj = draw.disk([200, 200], 10) # center/radius
stim_area[ii, jj] = True
stim_sequence.add_stim(fw.StimVoltageMatrix2D(0, 1, stim_area))
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# show the potential map at the end of calculations:
# plt.figure()
plt.imshow(aliev_panfilov.u)
plt.show()
