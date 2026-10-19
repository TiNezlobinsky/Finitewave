import finitewave as fw
import numpy as np
import matplotlib.pyplot as plt

cell = fw.SingleCellSimulation(model=fw.LuoRudy91(), dt=0.01)
cell.add_stimulation(
    n_beats=10,
    cycle_length=1000,
    current=20,
    duration=2,
)
state_vars = cell.run(history=True)

plt.plot(cell.times, cell.u_history)
plt.xlabel("Time (ms)")
plt.ylabel("Transmembrane Potential (mV)")
plt.title(f"Single Cell Simulation: {cardiac_model.__class__.__name__}")
plt.show()
