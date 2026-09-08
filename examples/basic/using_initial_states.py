import matplotlib.pyplot as plt
import gc
import shutil

import finitewave as fw

# number of nodes on the side
# create a tissue of size 400x400 with cardiomycytes:
n = 400
tissue = fw.CardiacTissue([n, n], dr=0.25)

# set up stimulation parameters:
stim_sequence = fw.StimSequence()
stim_sequence.add_stim(fw.StimVoltageCoord(0, 1,
                                           n//2 - 5, n//2 + 5,
                                           n//2 - 5, n//2 + 5))

# create model object and set up parameters:
simulation = fw.CardiacSimulation(dt=0.01, t_max=20)
# add the tissue and the stim parameters to the model object:
simulation.cardiac_model = fw.AlievPanfilov()
simulation.cardiac_tissue = tissue
simulation.stim_sequence = stim_sequence

# run the model:
simulation.run()

# u_before = simulation.cardiac_model.output("u")
u_before = simulation.cardiac_model.u.copy()

v_map = simulation.cardiac_model.v.copy()
v_map[0:n//2, 0:n//2] = 2
simulation.cardiac_model.v = v_map

simulation.t_max = 40
simulation.run(initialize=False)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# plot the results:
axs[0].imshow(u_before, cmap='hot')
axs[0].set_title('Before re-initialization (t=20)')
axs[1].imshow(simulation.cardiac_model.u, cmap='hot')
axs[1].set_title('After re-initialization (t=40)')
plt.show()