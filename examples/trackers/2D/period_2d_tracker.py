
#
# Use the Period2DTracker to measure wave period (e.g spiral wave) in
# particular cells. To measure the wave period in a cell,
# the LocalActivationTime2DTracker is more suitable.
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
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 300

# induce spiral wave:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(0, 1, 0, n, 0, n//2))
stim_sequence.add_stim(fw.StimVoltageCoord2D(31, 1, 0, n//2, 0, n))

tracker_sequence = fw.TrackerSequence()
period_tracker = fw.Period2DTracker()
positions = np.array([[1, 1], [5, 5], [7, 3], [9, 1],
                      [100, 100], [150, 3], [100, 150]])
period_tracker.cell_ind = positions
period_tracker.threshold = 0.5
period_tracker.start_time = 100
period_tracker.step = 10
tracker_sequence.add_tracker(period_tracker)

# add the tissue and the stim parameters to the model object:
aliev_panfilov.cardiac_tissue = tissue
aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

# get the wave period as a pandas Series with the cell index as the index:
periods = period_tracker.output

# plot the wave period:
plt.figure()
plt.errorbar(range(len(positions)),
             periods.apply(lambda x: x.mean()),
             yerr=periods.apply(lambda x: x.std()),
             fmt='o')
plt.xticks(range(len(positions)), [f'({x[0]}, {x[1]})' for x in positions],
           rotation=45)
plt.xlabel('Cell Index')
plt.ylabel('Period')
plt.title('Wave period')
plt.tight_layout()
plt.show()
