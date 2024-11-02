
#
# Use the Velocity2DTracker measure the wave front velocity.
# The Velocity2DTracker gives a list of velocities for each wave front node.
#

import matplotlib.pyplot as plt
import numpy as np

import finitewave as fw

# number of nodes on the side
n = 200

tissue = fw.CardiacTissue2D([n, n])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, n], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# don't forget to add the fibers array even if you have an anisotropic tissue:
tissue.fibers = np.zeros([n, n, 2])

# create model object:
aliev_panfilov = fw.AlievPanfilov2D()

# set up numerical parameters:
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 15

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 95, 105, 95, 105))

tracker_sequence = fw.TrackerSequence()
act_time_tracker = fw.ActivationTime2DTracker()
act_time_tracker.threshold = 0.5
tracker_sequence.add_tracker(act_time_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

velocity_calculation = fw.PlanarWaveVelocity2DCalculation()
velocity = velocity_calculation.compute_velocity_front(act_time_tracker.output,
                                                       aliev_panfilov.dr)
print("Mean wave front velocity: ", np.mean(velocity))
