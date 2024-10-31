
#
# Use the VTKFrame3DTracker to create a snapshot folder with vtk files suitable
# for building animation.
# Load the snapshot dir in paraview as series (it's possible to create
# animation with series).
#

import numpy as np

import finitewave as fw

# number of nodes on the side
n = 100
nj = 100
nk = 10

tissue = fw.CardiacTissue3D([n, nj, nk])
# create a mesh of cardiomyocytes (elems = 1):
tissue.mesh = np.ones([n, nj, nk], dtype="uint8")
# add empty nodes on the sides (elems = 0):
tissue.add_boundaries()

# create model object:
aliev_panfilov = fw.AlievPanfilov3D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 100

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, n, 0, n//2, 0, nk))
stim_sequence.add_stim(fw.StimVoltageCoord3D(31, 1, 0, n//2, 0, n, 0, nk))

tracker_sequence = fw.TrackerSequence()
vtk_frame_tracker = fw.VTKFrame3DTracker()
# We want to write the animation for the voltage variable. Use string value
# to specify the required array.anim_data
vtk_frame_tracker.variable_name = "u"
# write every 3 time unit.
vtk_frame_tracker.start_time = 40
vtk_frame_tracker.end_time = 100
vtk_frame_tracker.step = 100
tracker_sequence.add_tracker(vtk_frame_tracker)

aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()