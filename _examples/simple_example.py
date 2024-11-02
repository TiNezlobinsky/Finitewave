import numpy as np
import matplotlib.pyplot as plt
import finitewave as fw

n = 100
tissue = fw.CardiacTissue2D([n, n])
tissue.mesh = np.ones([n, n])
tissue.add_boundaries()

aliev_panfilov = fw.AlievPanfilov2D()
aliev_panfilov.dt = 0.01
aliev_panfilov.dr = 0.25
aliev_panfilov.t_max = 10
aliev_panfilov.cardiac_tissue = tissue

stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord2D(time=0,
                                             volt_value=1,
                                             x1=1, x2=n-1, y1=1, y2=3))

act_time_tracker = fw.ActivationTime2DTracker()
act_time_tracker.threshold = 0.5
act_time_tracker.step = 100

tracker_sequence = fw.TrackerSequence()
tracker_sequence.add_tracker(act_time_tracker)

aliev_panfilov.stim_sequence = stim_sequence
aliev_panfilov.tracker_sequence = tracker_sequence

aliev_panfilov.run()

fig, axs = plt.subplots(ncols=2)
axs[0].imshow(aliev_panfilov.u, cmap='coolwarm')
axs[0].set_title("u")

axs[1].imshow(act_time_tracker.output, cmap='viridis')
axs[1].set_title("Activation time")

fig.suptitle("Aliev-Panfilov 2D isotropic")
plt.tight_layout()
plt.show()
