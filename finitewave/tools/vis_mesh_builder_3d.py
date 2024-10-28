import pyvista as pv
import numpy as np


class VisMeshBuilder3D:
    """Class to build a 3D mesh for visualization with pyvista.

    Attributes:
        grid (pv.UnstructuredGrid): pyvista Unstructured Grid.
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
        grid = pv.ImageData()
        grid.dimensions = np.array(mesh.shape) + 1
        grid.spacing = (1, 1, 1)
        grid.cell_data['mesh'] = mesh.astype(float).flatten(order='F')
        # Threshold the mesh to remove empty space
        self.grid = grid.threshold(0.5)
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

        scalars_flat = scalars.T[self._mesh.T > 0].flatten(order='F')
        self.grid.cell_data[name] = scalars_flat
        self.grid.set_active_scalars(name)
        return self.grid

    def add_vector(self, vectors, name='Vectors'):
        """Add a vector field to the mesh. The vector field is flattened
        and only the values of the non-empty space are added to the mesh.

        Args:
            vectors (np.array): 3D vector field.
            name (str): Name of the vector.

        Returns:
            grid (pv.UnstructuredGrid): pyvista Unstructured Grid.
        """

        if vectors.shape != self._mesh.shape + (3,):
            raise ValueError("Vectors must have the same shape as the mesh.")

        vectors_list = []
        for i in range(3):
            x = vectors[:, :, :, i].T
            x_flat = x[self._mesh.T > 0].flatten(order='F')
            vectors_list.append(x_flat)

        vectors_flat = np.vstack(vectors_list).T

        self.grid.cell_data[name] = vectors_flat
        self.grid.set_active_vectors(name)
        return self.grid
