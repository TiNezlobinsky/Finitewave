
"""
Spiral Wave in 2D
=================

This example demonstrates how to use Aliev-Panfilov model in 2D to initiate a
spiral wave.
"""

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# set up the tissue:
n = 256
tissue = fw.CardiacTissue2D([n, n])


# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=0, volt_value=1,
                                             x1=0, x2=n, y1=0, y2=5))
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=50, volt_value=1,
                                             x1=n//2, x2=n, y1=0, y2=n))

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()
# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.3
aliev_panfilov.t_max = 150
# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence

aliev_panfilov.run()

# show the potential map at the end of calculations:
plt.imshow(aliev_panfilov.u)
plt.show()
