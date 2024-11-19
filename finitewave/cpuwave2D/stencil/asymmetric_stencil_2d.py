import numpy as np
from numba import njit, prange

from finitewave.core.stencil.stencil import Stencil


@njit
def minor_corners(m0, m1, m2, m3):
    """
    Computes the coefficients for minor (secondary) partial derivatives.

    For example for corner point ``i-1, j-1`` which is used in the calculation
    of the ``du/dy`` at the point ``o = (i-1/2, j)``:

    .. code-block:: text
        m1 ----- m3 ------- x
        |         |         |
        |         |         |
        |         |         |
        x -- o -- x ------- x
        |         |         |
        |         |         |
        |         |         |
        m0 ------ m2 ------ x

    Coefficient ``m0 == 0`` if:
    - ``m0 == 0``
    - ``m1 == 0`` and ``m3 == 0``

    Coefficient ``m0 == 1`` if:
    - ``m0 == 1`` and ``m2 == 1``

    Coefficient ``m0 == 2`` if:
    - ``m0 == 1`` and ``m2 == 0``

    Parameters
    ----------
    m0 : int
        Target mesh point value.
    m1 : int
        Mesh point opposite to m0.
    m2 : int
        Mesh point adjacent to m0.
    m3 : int
        Mesh point adjacent to m1.

    Returns
    -------
    int
        Coefficient for the secondary partial derivatives.
    """
    return 0.25 * m0 * (m0 + (m2 == 0)) * ((m1 + m3) >= 1)
    # return 0.5 * m0 * m1


@njit
def minor_component(d, m, m0, m1, m2, m3):
    """
    Computes the minor component for the diffusion.

    Parameters
    ----------
    d : np.ndarray
        Diffusion at half-steps.
    m : np.ndarray
        Mesh point adjacent to the center point.
    m0 : int
        Mesh point value.
    m1 : int
        Mesh point opposite to m0.
    m2 : int
        Mesh point adjacent to m0.
    m3 : int
        Mesh point adjacent to m1.

    Returns
    -------
    np.ndarray
        Minor component for the diffusion.
    """
    # print(f'd={d}, m={m}, m0={m0}, m1={m1}, m2={m2}, m3={m3}')
    return d * m * minor_corners(m0, m1, m2, m3)


@njit
def major_component(d, m):
    """
    Computes the major component for the difussion current.

    Parameters
    ----------
    d : np.ndarray
        Diffusion at half-steps.
    m : np.ndarray
        Mesh point value.

    Returns
    -------
    np.ndarray
        Major component for the diffusion.
    """
    return d * m


@njit
def compute_major_components(d_xx_0, d_xx_1, d_yy_0, d_yy_1, m01, m21, m10,
                             m12):
    """
    Computes the major component for the diffusion current.

    .. code-block:: text
        m02 ------------- m12 --------------- m22
        |                  |                   |
        |               d_yy_1                 |
        |                  |                   |
        m01 --- d_xx_0 --- m11 --- d_xx_1 --- m21
        |                  |                   |
        |               d_yy_0                 |
        |                  |                   |
        m00 ------------- m10 --------------- m20
    """
    w1 = major_component(d_xx_0, m01)
    w3 = major_component(d_yy_0, m10)
    w5 = major_component(d_yy_1, m12)
    w7 = major_component(d_xx_1, m21)
    return w1, w3, w5, w7


@njit
def compute_minor_components(d_xy_0, d_xy_1, d_yx_0, d_yx_1, m00, m01, m02,
                             m10, m12, m20, m21, m22):
    """
    Computes the minor component for the diffusion current.

    .. code-block:: text
        m02 ------------- m12 --------------- m22
        |                  |                   |
        |               d_yx_1                 |
        |                  |                   |
        m01 --- d_xy_0 --- m11 --- d_xy_1 --- m21
        |                  |                   |
        |               d_yx_0                 |
        |                  |                   |
        m00 ------------- m10 --------------- m20

    """
    # m00 (i-1, j-1)
    w0 = (minor_component(d_xy_0, m01, m00, m02, m10, m12)
          + minor_component(d_yx_0, m10, m00, m20, m01, m21))
    # m01 (i-1, j)
    w1 = (minor_component(d_yx_0, m10, m01, m21, m00, m20)
          - minor_component(d_yx_1, m12, m01, m21, m02, m22))
    # m02 (i-1, j+1)
    w2 = (- minor_component(d_xy_0, m01, m02, m00, m12, m10)
          - minor_component(d_yx_1, m12, m02, m22, m01, m21))
    # m10 (i, j-1)
    w3 = (minor_component(d_xy_0, m01, m10, m12, m00, m02)
          - minor_component(d_xy_1, m21, m10, m12, m20, m22))
    # m12 (i, j+1)
    w5 = (- minor_component(d_xy_0, m01, m12, m10, m02, m00)
          + minor_component(d_xy_1, m21, m12, m10, m22, m20))
    # m20 (i+1, j-1)
    w6 = (- minor_component(d_xy_1, m21, m20, m22, m10, m12)
          - minor_component(d_yx_0, m10, m20, m00, m21, m01))
    # m21 (i+1, j)
    w7 = (- minor_component(d_yx_0, m10, m21, m01, m20, m00)
          + minor_component(d_yx_1, m12, m21, m01, m22, m02))
    # m22 (i+1, j+1)
    w8 = (minor_component(d_xy_1, m21, m22, m20, m12, m10)
          + minor_component(d_yx_1, m12, m22, m02, m21, m01))

    return w0, w1, w2, w3, w5, w6, w7, w8


@njit
def compute_local_weights(d_xx_0, d_xx_1, d_xy_0, d_xy_1, d_yx_0, d_yx_1,
                          d_yy_0, d_yy_1, m00, m01, m02, m10, m11, m12, m20,
                          m21, m22):
    """
    Computes the weights for central point.

    .. code-block:: text
        m02 ------------- m12 --------------- m22
        |                  |                   |
        |               d_yy_1                 |
        |               d_yx_1                 |
        |                  |                   |
        |       d_xx_0     |       d_xx_1      |
        m01 --- d_xy_0 --- m11 --- d_xy_1 --- m21
        |                  |                   |
        |                  |                   |
        |               d_yx_0                 |
        |               d_yy_0                 |
        |                  |                   |
        m00 ------------- m10 --------------- m20

    Parameters
    ----------
    d_xx_0 : np.ndarray
        Diffusion x component for x direction at half-step (i-1/2, j).
    d_xx_1 : np.ndarray
        Diffusion x component for x direction at half-step (i+1/2, j).
    d_xy_0 : np.ndarray
        Diffusion y component for x direction at half-step (i-1/2, j).
    d_xy_1 : np.ndarray
        Diffusion y component for x direction at half-step (i+1/2, j).
    d_yx_0 : np.ndarray
        Diffusion x component for y direction at half-step (i, j-1/2).
    d_yx_1 : np.ndarray
        Diffusion x component for y direction at half-step (i, j+1/2).
    d_yy_0 : np.ndarray
        Diffusion y component for y direction at half-step (i, j-1/2).
    d_yy_1 : np.ndarray
        Diffusion y component for y direction at half-step (i, j+1/2).
    m00 : int
        Mesh point value at (i-1, j-1).
    m01 : int
        Mesh point value at (i-1, j).
    m02 : int
        Mesh point value at (i-1, j+1).
    m10 : int
        Mesh point value at (i, j-1).
    m11 : int
        Mesh point value at (i, j).
    m12 : int
        Mesh point value at (i, j+1).
    m20 : int
        Mesh point value at (i+1, j-1).
    m21 : int
        Mesh point value at (i+1, j).
    m22 : int
        Mesh point value at (i+1, j+1).

    Returns
    -------
    tuple
        9-weight tuple for the central point.
    """
    # m00 (i-1, j-1)
    w0 = (minor_component(d_xy_0, m01, m00, m02, m10, m12) +
          minor_component(d_yx_0, m10, m00, m20, m01, m21))
    # m01 (i-1, j)
    w1 = major_component(d_xx_0, m01)
    w1 += minor_component(d_yx_0, m10, m01, m21, m00, m20)
    w1 -= minor_component(d_yx_1, m12, m01, m21, m02, m22)
    # m02 (i-1, j+1)
    w2 = - (minor_component(d_xy_0, m01, m02, m00, m12, m10) +
            minor_component(d_yx_1, m12, m02, m22, m01, m21))
    # m10 (i, j-1)
    w3 = major_component(d_yy_0, m10)
    w3 += minor_component(d_xy_0, m01, m10, m12, m00, m02)
    w3 -= minor_component(d_xy_1, m21, m10, m12, m20, m22)
    # m11 (i, j)
    w4 = - (major_component(d_xx_0, m01) +
            major_component(d_yy_0, m10) +
            major_component(d_xx_1, m21) +
            major_component(d_yy_1, m12))
    # m12 (i, j+1)
    w5 = major_component(d_yy_1, m12)
    w5 += (- minor_component(d_xy_0, m01, m12, m10, m02, m00)
           + minor_component(d_xy_1, m21, m12, m10, m22, m20))

    # m20 (i+1, j-1)
    w6 = - (minor_component(d_xy_1, m21, m20, m22, m10, m12) +
            minor_component(d_yx_0, m10, m20, m00, m21, m01))
    # m21 (i+1, j)
    w7 = major_component(d_xx_1, m21)
    w7 += (- minor_component(d_yx_0, m10, m21, m01, m20, m00)
           + minor_component(d_yx_1, m12, m21, m01, m22, m02))
    # m22 (i+1, j+1)
    w8 = (minor_component(d_xy_1, m21, m22, m20, m12, m10) +
          minor_component(d_yx_1, m12, m22, m02, m21, m01))

    return w0, w1, w2, w3, w4, w5, w6, w7, w8


@njit
def compute_weights(w, m, d_xx, d_xy, d_yx, d_yy):
    """
    Computes the weights for diffusion on a 2D mesh based on the asymmetric
    stencil.

    Parameters
    ----------
    w : np.ndarray
        3D array to store the weights for diffusion. Shape is (*mesh.shape, 9).
    m : np.ndarray
        2D array representing the mesh grid of the tissue. Non-tissue areas
        are set to 0.
    d_xx : np.ndarray
        Diffusion x component for x direction.
    d_xy : np.ndarray
        Diffusion y component for x direction.
    d_yx : np.ndarray
        Diffusion x component for y direction.
    d_yy : np.ndarray
        Diffusion y component for y direction.

    Returns
    -------
    np.ndarray
        3D array of weights for diffusion, with the shape of (*mesh.shape, 9).

    Notes
    -----
    The method assumes weights being used in the following order:
        ``w[0] : i-1, j-1``,
        ``w[1] : i-1, j``,
        ``w[2] : i-1, j+1``,
        ``w[3] : i, j-1``,
        ``w[4] : i, j``,
        ``w[5] : i, j+1``,
        ``w[6] : i+1, j-1``,
        ``w[7] : i+1, j``,
        ``w[8] : i+1, j+1``.
    """
    n_i = m.shape[0]
    n_j = m.shape[1]
    for ii in prange(n_i * n_j):
        i = int(ii / n_j)
        j = ii % n_j
        if m[i, j] != 1:
            continue

        # w[i, j, :] = compute_local_weights(d_xx[i-1, j], d_xx[i, j],
        #                                    d_xy[i-1, j], d_xy[i, j],
        #                                    d_yx[i, j-1], d_yx[i, j],
        #                                    d_yy[i-1, j], d_yy[i, j],
        #                                    m[i-1, j-1], m[i-1, j],
        #                                    m[i-1, j+1], m[i, j-1],
        #                                    m[i, j], m[i, j+1],
        #                                    m[i+1, j-1], m[i+1, j],
        #                                    m[i+1, j+1])

        w_major = compute_major_components(d_xx[i-1, j], d_xx[i, j],
                                           d_yy[i, j-1], d_yy[i, j],
                                           m[i-1, j], m[i+1, j],
                                           m[i, j-1], m[i, j+1])

        w_minor = compute_minor_components(d_xy[i-1, j], d_xy[i, j],
                                           d_yx[i, j-1], d_yx[i, j],
                                           m[i-1, j-1], m[i-1, j], m[i-1, j+1],
                                           m[i, j-1], m[i, j+1],
                                           m[i+1, j-1], m[i+1, j], m[i+1, j+1])
        # (i-1, j-1)
        w[i, j, 0] = w_minor[0]
        # (i-1, j)
        w[i, j, 1] = w_major[0] + w_minor[1]
        # (i-1, j+1)
        w[i, j, 2] = w_minor[2]
        # (i, j-1)
        w[i, j, 3] = w_major[1] + w_minor[3]
        # (i, j)
        w[i, j, 4] = - (w_major[0] + w_major[1] + w_major[2] + w_major[3])
        # (i, j+1)
        w[i, j, 5] = w_major[2] + w_minor[4]
        # (i+1, j-1)
        w[i, j, 6] = w_minor[5]
        # (i+1, j)
        w[i, j, 7] = w_major[3] + w_minor[6]
        # (i+1, j+1)
        w[i, j, 8] = w_minor[7]

    return w


class AsymmetricStencil2D(Stencil):
    """
    A class to represent a 2D asymmetric stencil for diffusion processes.
    The asymmetric stencil is used to handle anisotropic diffusion in the
    tissue.
    """

    def __init__(self):
        super().__init__()

    def get_weights(self, mesh, conductivity, fibers, D_al, D_ac, dt, dr):
        """
        Computes the weights for diffusion on a 2D mesh using an asymmetric
        stencil.

        Parameters
        ----------
        mesh : np.ndarray
            2D array representing the mesh grid of the tissue.
        conductivity : float
            Conductivity of the tissue, which scales the diffusion coefficient.
        fibers : np.ndarray
            Array representing fiber orientations. Used to compute directional
            diffusion coefficients.
        D_al : float
            Longitudinal diffusion coefficient.
        D_ac : float
            Cross-sectional diffusion coefficient.
        dt : float
            Temporal resolution.
        dr : float
            Spatial resolution.

        Returns
        -------
        np.ndarray
            3D array of weights for diffusion, with the shape of (mesh.shape[0], mesh.shape[1], 9).

        Notes
        -----
        The method assumes asymmetric diffusion where different coefficients are used for different directions.
        The weights are computed for eight surrounding directions and the central weight, based on the asymmetric stencil.
        Heterogeneity in the diffusion coefficients is handled by adjusting the weights based on fiber orientations.
        """
        mesh = mesh.copy()
        mesh[mesh != 1] = 0
        # fibers[np.where(mesh != 1)] = 0
        weights = np.zeros((*mesh.shape, 9))

        d_xx, d_xy = self.compute_half_step_diffusion(mesh, conductivity,
                                                      fibers, D_al, D_ac, 0)
        d_yx, d_yy = self.compute_half_step_diffusion(mesh, conductivity,
                                                      fibers, D_al, D_ac, 1)
        compute_weights(weights, mesh, d_xx, d_xy, d_yx, d_yy)
        weights *= dt/dr**2
        weights[:, :, 4] += 1

        return weights

    def compute_half_step_diffusion(self, mesh, conductivity, fibers, D_al,
                                    D_ac, axis, num_axes=2):
        """
        Computes the diffusion components for half-steps based on fiber
        orientations.

        Parameters
        ----------
        mesh : np.ndarray
            Array representing the mesh grid of the tissue.
        conductivity : np.ndarray
            Array representing the conductivity of the tissue.
        fibers : np.ndarray
            Array representing fiber orientations with shape
            ``(2, *mesh.shape)``.
        D_al : float
            Longitudinal diffusion coefficient.
        D_ac : float
            Cross-sectional diffusion coefficient.
        axis : int
            Axis index (0 for x, 1 for y).
        num_axes : int
            Number of axes.

        Returns
        -------
        np.ndarray
            Array of diffusion components for half-steps along the specified
            axis.

        Notes
        -----
        The index ``i`` in the returned array corresponds to ``i+1/2`` and
        ``i-1`` corresponds to ``i-1/2``.
        """
        if conductivity is None:
            conductivity = 1

        D = np.zeros((num_axes, *mesh.shape))
        for i in range(num_axes):
            D[i] = self.compute_diffusion_components(fibers, axis, i, D_al,
                                                     D_ac)
            D[i] *= conductivity
            D[i] = 0.5 * (D[i] + np.roll(D[i], -1, axis=axis))

        return D

    def compute_diffusion_components(self, fibers, ind0, ind1, D_al, D_ac):
        """
        Computes the diffusion components based on fiber orientations.

        Parameters
        ----------
        fibers : np.ndarray
            Array representing fiber orientations.
        ind0 : int
            First axis index (0 for x, 1 for y).
        ind1 : int
            Second axis index (0 for x, 1 for y).
        D_al : float
            Longitudinal diffusion coefficient.
        D_ac : float
            Cross-sectional diffusion coefficient.

        Returns
        -------
        np.ndarray
            Array of diffusion components based on fiber orientations
        """
        return (D_ac * (ind0 == ind1) +
                (D_al - D_ac) * fibers[ind0] * fibers[ind1])
