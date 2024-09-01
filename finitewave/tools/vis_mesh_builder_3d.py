import pyvista as pv
import numpy as np


class VisMeshBuilder3D:
    """Class to build a 3D mesh for visualization with pyvista.
    """
    def __init__(self):
        pass

    def build_mesh(self, mesh):
        """Build a Unstructured Grid from 3D mesh where mesh > 0.

        Args:
            mesh (np.array): 3D mesh with cardiomyocytes (elems = 1),
                empty space (elems = 0), and fibrosis (elems = 2).

        Returns:
            grid (pv.UnstructuredGrid): pyvista Unstructured Grid.
        """
        grid = pv.UniformGrid()
        grid.dimensions = np.array(mesh.shape) + 1
        grid.spacing = (1, 1, 1)
        grid.cell_data['mesh'] = mesh.astype(float).flatten(order='F')
        # Threshold the mesh to remove empty space
        self.grid = grid.threshold(0.5)
        self._mesh_mask = (mesh > 0).flatten(order='F')
        self._mesh = mesh
        return self.grid

    def add_scalar(self, scalars, name='Scalars'):
        """Add a scalar field to the mesh. The scalar field is flattened
        and only the values of the non-empty space are added to the mesh.

        Args:
            scalar (np.array): 3D scalar field.
            name (str): Name of the scalar.

        Returns:
            grid (pv.UnstructuredGrid): pyvista Unstructured Grid.
        """

        if scalars.shape != self._mesh.shape:
            raise ValueError("Scalars must have the same shape asthe mesh.")

        self.grid.cell_data[name] = scalars.flatten(order='F')[self._mesh_mask]
        self.grid.set_active_scalars(name)
        return self.grid
