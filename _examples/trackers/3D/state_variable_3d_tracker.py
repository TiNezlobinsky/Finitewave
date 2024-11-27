
#
# Here we use the Variable2DTracker and MultiVariable2DTracker classes to track
# the values of the variables u and v at the specified cell indices.
#

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 100
nk = 10

# create tissue object:
tissue = fw.CardiacTissue3D([n, n, nk])
tissue.mesh = np.ones([n, n, nk], dtype="uint8")
tissue.add_boundaries()

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()

# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 100

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 1, 3, 1, n, 1, nk))

tracker_sequence = fw.TrackerSequence()
# add one variable tracker:
variable_tracker = fw.Variable3DTracker()
variable_tracker.var_name = "u"
variable_tracker.cell_ind = [40, 40, 5]
tracker_sequence.add_tracker(variable_tracker)

# add the multi variable tracker:
multivariable_tracker = fw.MultiVariable3DTracker()
# to specify the mesh node under the measuring - use the cell_ind field:
multivariable_tracker.cell_ind = [30, 30, 5]
multivariable_tracker.var_list = ["u", "v"]
tracker_sequence.add_tracker(multivariable_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# plot the action potential and state variable v at the measuring point
time = np.arange(len(multivariable_tracker.output["u"])) * aliev_panfilov.dt

plt.plot(time, variable_tracker.output, label="u")
plt.plot(time, multivariable_tracker.output["u"], label="u")
plt.plot(time, multivariable_tracker.output["v"], label="v")
plt.legend(title=aliev_panfilov.__class__.__name__)
plt.show()
