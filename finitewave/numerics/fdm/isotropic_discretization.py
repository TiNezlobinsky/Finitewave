import numpy as np
from .asymmetric_discretization import AsymmetricDiscretization


class IsotropicDiscretization(AsymmetricDiscretization):
    """Axis-aligned finite differences with mirrored boundary neighbors.

    Only diagonal tensor components contribute.  At an interior point this is
    the standard centered approximation of ``-div(D grad(u))``.  If exactly
    one axial neighbor is invalid, its contribution is mirrored onto the
    valid neighbor, corresponding to a reflected ghost value and a homogeneous
    Neumann boundary condition.  This produces the familiar doubled boundary
    coefficient.

    Use :class:`AsymmetricDiscretization` when off-diagonal tensor components
    are required.
    """
    
    def _diffusion_operator_component(self, mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map):
        """Emit COO triplets for one coordinate-axis contribution.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray [*mesh.shape, ndim, ndim]
            Normalized scalar or tissue-indexed diffusion tensor.
        connectivity : scalar or numpy.ndarray
            Normalized positive-edge connectivity.
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            Active-cell coordinates with shape ``(mesh.ndim, n_active)``.
        axis : int
            The axis along which to compute the diffusion weights.
        tissue_index_map : numpy.ndarray
            Grid-shaped map from coordinates to compressed tissue indexes.

        Returns
        -------
        rows : np.ndarray
            The row indexes for the sparse matrix.
        cols : np.ndarray
            The column indexes for the sparse matrix.
        weights : np.ndarray
            The weights for the sparse matrix.
        """
        ijk_list, w_list = self._flux_weights(mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map)
        rows, cols, weights = self.nonzero_weights(mesh, ijk, ijk_list, w_list, tissue_index_map, direction=1)

        weights = weights / dr
        return rows, cols, weights

    def _flux_weights(self, mesh, diffusion, connectivity, dr, ijk, axis, tissue_index_map):
        """Build the two-sided axial stencil before the final divergence.

        For a valid positive or negative neighbor the face contribution is

        ``D_face * (u_center - u_neighbor) / dr``.

        If only one neighbor is valid, the invalid coordinate and coefficient
        are replaced by those of the valid side.  Thus both face terms refer to
        the same neighbor and the eventual operator contribution is doubled.
        If neither side is valid, both contributions are zero.

        Parameters
        ----------
        mesh : numpy.ndarray
            The mesh of the simulation.
        diffusion : numpy.ndarray
            Normalized scalar or tissue-indexed diffusion tensor.
        connectivity : scalar or numpy.ndarray
            Normalized positive-edge connectivity.  Connectivity for the
            negative face is stored at the negative neighbor.
        dr : float
            The grid spacing.
        ijk : numpy.ndarray
            Active-cell coordinates with shape ``(mesh.ndim, n_active)``.
        axis : int
            The axis along which to compute the flux weights.
        tissue_index_map : numpy.ndarray
            Grid-shaped map from coordinates to compressed tissue indexes.

        Returns
        -------
        ijk_list : list
            Coordinates for center, positive neighbor, center, and negative
            neighbor, in that order.
        w_list : list
            Matching one-dimensional coefficients.  Each already contains one
            factor ``1 / dr``; the divergence adds the second factor later.
        """
        ijk_pos = self.build_neighbor(ijk, shift=1, axis=axis)
        ijk_neg = self.build_neighbor(ijk, shift=-1, axis=axis)

        valid_pos = self.is_valid_index(ijk_pos, mesh)
        valid_neg = self.is_valid_index(ijk_neg, mesh)

        d_pos = self._diffusion_tensor_component(
            diffusion, connectivity, ijk, ijk_pos, valid_pos, axis,
            tissue_index_map
        )[:, axis]
        d_neg = self._diffusion_tensor_component(
            diffusion, connectivity, ijk_neg, ijk, valid_neg, axis,
            tissue_index_map
        )[:, axis]

        # Keep writable vectors because boundary coefficients are mirrored below.
        d_pos = np.broadcast_to(d_pos, valid_pos.shape).copy()
        d_neg = np.broadcast_to(d_neg, valid_neg.shape).copy()

        invalid_pos = (~valid_pos) & valid_neg
        invalid_neg = (~valid_neg) & valid_pos
        
        ijk_pos[:, invalid_pos] = ijk_neg[:, invalid_pos]
        ijk_neg[:, invalid_neg] = ijk_pos[:, invalid_neg]

        d_pos[invalid_pos] = d_neg[invalid_pos]
        d_neg[invalid_neg] = d_pos[invalid_neg]

        d_pos /= dr
        d_neg /= dr

        ijk_list = [ijk, ijk_pos, ijk, ijk_neg]
        w_list = [d_pos, - d_pos, d_neg, - d_neg]

        return ijk_list, w_list
