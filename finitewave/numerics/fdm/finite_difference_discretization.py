from abc import abstractmethod
import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from finitewave.core.numerics.spatial_discretization import SpatialDiscretization


class FiniteDifferenceDiscretization(SpatialDiscretization):
    """
    Base class for finite difference discretization stencils.
    """

    def __init__(self):
        pass

    def compute_weights(self, tissue):
        """
        Computes the weights for the diffusion operator.

        Parameters
        ----------
        tissue : CardiacTissueBase
            The tissue object containing the mesh and diffusion tensor.

        Returns
        -------
        scipy.sparse.csr_matrix
            The stiffness matrix in tissue indexing.
        scipy.sparse.csr_matrix
            The identity mass matrix in tissue indexing.
        """
        mesh = tissue.mesh
        diffusion = tissue.diffusion_tensor
        connectivity = tissue.connectivity
        dr = tissue.dr
        indexes = tissue.myo_indexes

        stiffness = self.compute_diffusion_operator(mesh, dr, indexes, diffusion, connectivity)

        # mass matrix: we keep it to preserve the interface, but it is just an identity matrix.
        mass = sp.eye(stiffness.shape[0], dtype=stiffness.dtype, format='csr')
        return stiffness, mass

    @abstractmethod
    def compute_diffusion_operator(self, mesh, dr, indexes, diffusion, connectivity):
        """
        Computes the weights for calculating the diffusion operator.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            The diffusion tensor as a (*mesh.shape, ndim, ndim).
        dr : float
            The grid spacing.
        indexes : numpy.ndarray
            The indexes of the non-empty nodes in the mesh.

        Returns
        -------
        rows : numpy.ndarray
            The central node indexes.
        cols : numpy.ndarray
            The node indexes involved in diffusion calculation.
        weights : numpy.ndarray
            The weights for connections between central and neighboring nodes.
        """
        raise NotImplementedError()

    def nonzero_weights(self, mesh, ijk, ijk_list, w_list, index_map=None, direction=1):
        # TODO: check performance
        """
        Collects non-zero weights.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        ijk : numpy.ndarray
            The indexes of the non-empty nodes in the mesh.
        ijk_list : list
            The list of ijk coordinates of the neighboring nodes in the mesh.
        w_list : list
            The list of weights for the neighboring nodes.
        direction : int, optional
            The direction of the flux (1 for major-to-center, -1 for center-to-major)

        Returns
        -------
        rows : numpy.ndarray
            The row indexes for the sparse matrix.
        cols : numpy.ndarray
            The column indexes for the sparse matrix.
        weights : numpy.ndarray
            The non-zero weights for the sparse matrix.
        """
        if index_map is None:
            index_map = - np.ones_like(mesh, dtype=np.int64)
            index_map[mesh > 0] = np.arange(np.count_nonzero(mesh > 0))

        rows = []
        cols = []
        weights = []

        for neighbor_ijk, weight in zip(ijk_list, w_list):
            weight = np.asarray(weight)
            nonzero = np.flatnonzero(weight)
            if nonzero.size == 0:
                continue

            center_ijk = tuple(ijk[:, nonzero])
            active_neighbor_ijk = tuple(neighbor_ijk[:, nonzero])
            rows.append(index_map[center_ijk])
            cols.append(index_map[active_neighbor_ijk])
            weights.append(direction * weight[nonzero])

        if not rows:
            weight_dtype = np.asarray(w_list[0]).dtype
            empty_indexes = np.empty(0, dtype=np.int64)
            return empty_indexes, empty_indexes.copy(), np.empty(
                0, dtype=weight_dtype
            )

        return (
            np.concatenate(rows),
            np.concatenate(cols),
            np.concatenate(weights),
        )
             
    def build_neighbor(self, ijk, shift, axis):
        """
        Builds the neighbor ijk coordinates by shifting along a given axis.

        Parameters
        ----------
        ijk : numpy.ndarray
            The ijk coordinates of the current cell.
        shift : int
            The shift to apply along the specified axis.
        axis : int
            The axis along which to apply the shift.

        Returns
        -------
        numpy.ndarray
            The ijk coordinates of the neighbor.
        """

        ijk = ijk.copy()
        ijk[axis] += shift
        return ijk

    def is_valid_index(self, index, mesh):
        """
        Checks if the index coordinates are valid (within bounds and
        in a non-empty cell).

        Parameters
        ----------
        index : numpy.ndarray
            The ijk coordinates of the nodes.
        mesh : numpy.ndarray
            The mesh of the simulation.

        Returns
        -------
        numpy.ndarray
            A boolean array indicating the validity of each index.
        """
        valid = is_valid_indexes_numba(index, mesh)
        return valid
    
    def reindex_matrix(self, mesh, rows, cols, indexes):
        """
        Reindexes the rows and columns of the sparse matrix to avoid zero
        rows in the sparse matrix.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        rows : numpy.ndarray
            The row indices of the sparse matrix.
        cols : numpy.ndarray
            The column indices of the sparse matrix.
        indexes : numpy.ndarray
            The indexes of the non-empty cells in the mesh.

        Returns
        -------
        numpy.ndarray
            The reindexed row indices.
        numpy.ndarray
            The reindexed column indices.
        """
        c_indexes = np.zeros(mesh.size, dtype=np.int64)
        c_indexes[indexes] = np.arange(len(indexes))
        rows = c_indexes[rows]
        cols = c_indexes[cols]
        return rows, cols


@njit
def is_valid_index(multi_index, limits, mesh):
    for axis in range(mesh.ndim):
        coord = multi_index[axis]
        limit = limits[axis]
        if coord < 0 or coord >= limit:
            return False

    flat_index = ravel_multi_index_numba(multi_index, mesh.shape)
    return mesh.flat[flat_index] == 1
    

@njit(parallel=True)
def is_valid_indexes_numba(multi_indexes, mesh):
    n_points = multi_indexes.shape[1]
    limits = np.array(mesh.shape)
    mask = np.zeros(n_points, dtype=np.bool_)
    for i in prange(n_points):
        index = multi_indexes[:, i]
        mask[i] = is_valid_index(index, limits, mesh)
    return mask


@njit
def ravel_multi_index_numba(multi_index, shape):
    flat_index = 0
    for axis in range(len(shape)):
        flat_index = flat_index * shape[axis] + multi_index[axis]
    return flat_index
