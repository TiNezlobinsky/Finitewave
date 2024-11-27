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
aliev_panfilov.t_max = 120

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord3D(0, 1, 0, n, 0, n//2, 0, nk))
stim_sequence.add_stim(fw.StimVoltageCoord3D(31, 1, 0, n//2, 0, n, 0, nk))

tracker_sequence = fw.TrackerSequence()
period_map_tracker = fw.PeriodAnimation3DTracker()
period_map_tracker.dir_name = "period_map"
period_map_tracker.threshold = 0.3
period_map_tracker.step = 100
tracker_sequence.add_tracker(period_map_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

period_map_tracker.write(clim=[0, 120], cmap="jet", clear=True)
