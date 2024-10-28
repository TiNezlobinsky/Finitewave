from pathlib import Path
import numpy as np
import pyvista as pv
import finitewave as fw


path = Path(__file__).parents[1].joinpath("basic", "3D", "data")

fibers = np.load(path.joinpath("fibers.npy"))
u = np.load(path.joinpath("ap_rotor.npy"))
mesh = np.load(path.joinpath("mesh.npy"))
distance = np.load(path.joinpath("distance.npy"))

u_mesh = np.zeros_like(mesh, dtype=float)
u_mesh[mesh > 0] = u

fibers_mesh = np.zeros(mesh.shape + (3,), dtype=float)
fibers_mesh[mesh > 0] = fibers

distance_mesh = np.zeros_like(mesh, dtype=float)
distance_mesh[mesh > 0] = distance

mesh_builder = fw.VisMeshBuilder3D()
mesh_grid = mesh_builder.build_mesh(mesh)
mesh_grid = mesh_builder.add_scalar(u_mesh, name='U')
mesh_grid = mesh_builder.add_scalar(distance_mesh, name='Endo Distance')
mesh_grid = mesh_builder.add_vector(fibers_mesh, name='Fibers')

# # Save the mesh to a file
# mesh_grid.save('mesh.vtk')

# Show every 3rd fiber to reduce the number of arrows for better
# performance. This is not necessary if number of fibers is small.
# mesh_grid can be used directly to create the glyphs.
cent = np.argwhere(mesh[::3, ::3, ::3] > 0)
cent = cent * 3
arrow_mesh = np.zeros_like(mesh)
arrow_mesh[cent[:, 0], cent[:, 1], cent[:, 2]] = 1

mesh_builder = fw.VisMeshBuilder3D()
mesh_builder.build_mesh(arrow_mesh)
mesh_builder.add_scalar(distance_mesh, 'Endo Distance')
mesh_builder.add_vector(fibers_mesh, 'direction')
arrow_grid = mesh_builder.grid
arrow_grid = arrow_grid.glyph(orient='direction', factor=20, scale=True,
                              geom=pv.Arrow(tip_resolution=3,
                                            shaft_resolution=3))

pl = pv.Plotter(shape=(1, 2))
pl.add_mesh(mesh_grid, clim=[0, 2], cmap='viridis', scalars='mesh')

pl.subplot(0, 1)
pl.add_mesh(arrow_grid, scalars='Endo Distance', cmap='viridis', clim=[0, 1])

pl.link_views()
pl.show()
