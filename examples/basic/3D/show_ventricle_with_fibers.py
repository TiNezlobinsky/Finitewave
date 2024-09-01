from pathlib import Path
import numpy as np
import pyvista as pv
import finitewave as fw


path = Path(__file__).parent

fibers = np.load(path.joinpath("data", "fibers.npy"))
distance = np.load(path.joinpath("data", "distance.npy"))
mesh = np.load(path.joinpath("data", "mesh.npy"))

distance_mesh = np.zeros_like(mesh, dtype=float)
distance_mesh[mesh > 0] = distance

fibers_mesh = np.zeros(mesh.shape + (3,), dtype=float)
fibers_mesh[mesh > 0] = fibers

# Show every 3rd fiber to reduce the number of arrows for better 
# visualization
cent = np.argwhere(mesh[::3, ::3, ::3] > 0)
cent = cent * 3
# Select every 3rd fiber
direction = fibers_mesh[cent[:, 0], cent[:, 1], cent[:, 2]]
# Normalized distance from endocardium used to color the arrows
dist = distance_mesh[cent[:, 0], cent[:, 1], cent[:, 2]]

mesh_grid = fw.VisMeshBuilder3D().build_mesh(mesh)

arrow_grid = pv.StructuredGrid()
arrow_grid.points = cent
arrow_grid['direction'] = direction
arrow_grid['Endo Distance'] = dist

arrow_grid = arrow_grid.glyph(orient='direction', factor=20, scale=True,
                              geom=pv.Arrow(tip_resolution=3,
                                            shaft_resolution=3))

pl = pv.Plotter(shape=(1, 2))
pl.add_mesh(mesh_grid, clim=[0, 2], cmap='viridis', scalars='mesh')

pl.subplot(0, 1)
pl.add_mesh(arrow_grid, scalars='Endo Distance', cmap='viridis', clim=[0, 1])

pl.link_views()
pl.show()
