from abc import abstractmethod
import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from finitewave.core.numerics.spatial_discretization import SpatialDiscretization


class FiniteDifferenceDiscretization(SpatialDiscretization):
    """Base class for finite-difference diffusion discretizations.

    Subclasses assemble a sparse operator ``K`` on the compressed tissue
    indexing formed by ``mesh > 0``.  Rows are assembled only for excitable
    cells (``mesh == 1``); non-excitable tissue points (``mesh == 2``) remain
    in the matrix so that all numerical components use the same indexing.
    """

    def __init__(self):
        """Initialize the stateless base discretization."""
        pass

    def compute_weights(self, tissue):
        """Assemble the stiffness and mass matrices for ``tissue``.

        The stiffness matrix represents ``-div(D grad(u))`` in compressed
        tissue indexing.  FDM uses one degree of freedom per tissue point, so
        its mass matrix is the identity.  Time integrators combine these two
        matrices with the reaction term and the time step.

        Parameters
        ----------
        tissue : CardiacTissueBase
            Tissue providing ``mesh``, ``dr``, ``myo_indexes``,
            ``diffusion_tensor``, and ``connectivity``.

        Returns
        -------
        stiffness : scipy.sparse.csr_matrix
            Discrete positive diffusion operator ``K`` with shape
            ``(n_tissue, n_tissue)``, where ``n_tissue = count(mesh > 0)``.
        mass : scipy.sparse.csr_matrix
            Identity matrix ``M`` with the same shape and dtype as ``K``.
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
        """Assemble the sparse operator approximating ``-div(D grad(u))``.

        Implementations must return a square CSR matrix in compressed tissue
        indexing.  This differs from flat grid indexing: tissue index ``p``
        addresses the ``p``-th entry of ``flatnonzero(mesh > 0)``.

        Parameters
        ----------
        mesh : numpy.ndarray
            Integer grid: ``0`` is outside tissue, ``1`` is active tissue, and
            ``2`` is a retained but non-excitable/decoupled point.
        dr : float
            Uniform grid spacing; must be positive.
        indexes : numpy.ndarray
            One-dimensional positions of active cells in compressed tissue
            indexing, normally ``tissue.myo_indexes``.
        diffusion : scalar or numpy.ndarray
            Scalar coefficient, constant tensor, full-grid tensor field, or
            tensor field already stored in compressed tissue indexing.
        connectivity : scalar or numpy.ndarray
            Multipliers for positive-axis edges.  A zero value removes the
            corresponding flux without removing either endpoint from tissue.

        Returns
        -------
        scipy.sparse.csr_matrix
            Matrix ``K`` of shape ``(n_tissue, n_tissue)``.
        """
        raise NotImplementedError()

    def nonzero_weights(self, mesh, ijk, ijk_list, w_list, index_map=None, direction=1):
        # TODO: check performance
        """Convert stencil contributions to COO triplets.

        For every pair ``(neighbor_ijk, weight)`` this method selects entries
        for which ``weight != 0`` and emits
        ``(row=center, column=neighbor, value=direction * weight)``.  Duplicate
        triplets are intentional: ``scipy.sparse.csr_matrix`` sums them during
        assembly, which is how contributions from different faces and axes are
        accumulated.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        ijk : numpy.ndarray
            Center coordinates with shape ``(mesh.ndim, n_points)``.
        ijk_list : sequence of numpy.ndarray
            Neighbor-coordinate arrays, each with the same shape as ``ijk``.
        w_list : sequence of numpy.ndarray
            One-dimensional weight arrays of length ``n_points``.  The entry
            order must correspond to ``ijk_list``.
        index_map : numpy.ndarray, optional
            Grid-shaped map from coordinates to compressed tissue indexes.
            If omitted, it is constructed from ``mesh > 0``.
        direction : int, optional
            Sign applied to every emitted weight.  The asymmetric scheme uses
            ``1`` for the center row and ``-1`` for the adjacent face row.

        Returns
        -------
        rows, cols : numpy.ndarray
            One-dimensional compressed tissue indexes of nonzero entries.
        weights : numpy.ndarray
            One-dimensional values corresponding elementwise to ``rows`` and
            ``cols``.  All three arrays are empty when every weight is zero.
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
        """Return coordinates shifted along one grid axis.

        A copy is made before applying the shift; the input ``ijk`` is never
        modified.  Coordinates may temporarily lie outside the mesh and must
        be checked with :meth:`is_valid_index` before they are dereferenced.

        Parameters
        ----------
        ijk : numpy.ndarray
            Coordinate array with shape ``(mesh.ndim, n_points)``.
        shift : int
            The shift to apply along the specified axis.
        axis : int
            Grid axis in ``range(mesh.ndim)``.

        Returns
        -------
        numpy.ndarray
            Shifted copy with the same shape and dtype as ``ijk``.
        """

        ijk = ijk.copy()
        ijk[axis] += shift
        return ijk

    def is_valid_index(self, index, mesh):
        """Test whether coordinate columns refer to active cells.

        A coordinate is valid only when every component is in bounds and its
        mesh value is exactly ``1``.  Consequently both empty space
        (``mesh == 0``) and retained non-excitable points (``mesh == 2``) act
        as unavailable stencil neighbors.

        Parameters
        ----------
        index : numpy.ndarray
            Coordinate array with shape ``(mesh.ndim, n_points)``.
        mesh : numpy.ndarray
            The mesh of the simulation.

        Returns
        -------
        numpy.ndarray
            Boolean vector of length ``n_points``.
        """
        valid = is_valid_indexes_numba(index, mesh)
        return valid
    
    def reindex_matrix(self, mesh, rows, cols, indexes):
        """Map flat-grid COO indexes to a compact index set.

        This legacy helper is useful when ``rows`` and ``cols`` are expressed
        as indexes into ``mesh.ravel()``.  New FDM assembly normally uses a
        grid-shaped ``tissue_index_map`` directly.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        rows : numpy.ndarray
            The row indices of the sparse matrix.
        cols : numpy.ndarray
            The column indices of the sparse matrix.
        indexes : numpy.ndarray
            Flat-grid indexes to retain, in desired compact order.

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
    """Return whether one coordinate is in bounds and has ``mesh == 1``."""
    for axis in range(mesh.ndim):
        coord = multi_index[axis]
        limit = limits[axis]
        if coord < 0 or coord >= limit:
            return False

    flat_index = ravel_multi_index_numba(multi_index, mesh.shape)
    return mesh.flat[flat_index] == 1
    

@njit(parallel=True)
def is_valid_indexes_numba(multi_indexes, mesh):
    """Vectorized Numba kernel for validating coordinate columns."""
    n_points = multi_indexes.shape[1]
    limits = np.array(mesh.shape)
    mask = np.zeros(n_points, dtype=np.bool_)
    for i in prange(n_points):
        index = multi_indexes[:, i]
        mask[i] = is_valid_index(index, limits, mesh)
    return mask


@njit
def ravel_multi_index_numba(multi_index, shape):
    """Convert one C-order multidimensional index to a flat index."""
    flat_index = 0
    for axis in range(len(shape)):
        flat_index = flat_index * shape[axis] + multi_index[axis]
    return flat_index
