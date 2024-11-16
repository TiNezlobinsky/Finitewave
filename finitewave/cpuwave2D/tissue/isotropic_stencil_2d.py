import numpy as np


def heterogeneity(mesh, conductivity):
    """
    Adjusts the mesh values based on the conductivity of the tissue.

    Parameters
    ----------
    mesh : np.ndarray
        2D array representing the mesh grid of the tissue. Non-tissue areas
        are set to 0.

    conductivity : np.ndarray
        2D array representing the conductivity of the tissue.

    Returns
    -------
    np.ndarray
        A numpy array of mesh values adjusted based on the conductivity of
        the tissue.
    """
    return mesh * conductivity


def compute_weights(self, mesh, dt, dr, D, conductivity=None):
    """
    Computes the weights for diffusion on a 2D mesh using an isotropic
    stencil.

    Parameters
    ----------
    mesh : np.ndarray
        2D array representing the mesh grid of the tissue. Non-tissue areas
        are set to 0.

    dt : float
        Time step used in the simulation.

    dr : float
        Spatial resolution of the mesh.

    D : float
        Diffusion coefficient used in the simulation.

    conductivity : nd.ndarray, optional
        A 2D array representing the conductivity of the tissue.

    Returns
    -------
    np.ndarray
        A numpy array of stencil weights computed based on the provided
        parameters. The shape of the array is (K, N, M), where K is the
        number of stencil points, N and M are the dimensions of the mesh.

    Notes
    -----
        The method assumes isotropic diffusion if the conductivity is not
        provided. The weights are computed based on the isotropic stencil
        for diffusion processes in 2D with a 5-point stencil.
            o
            |
        o---o---o
            |
            o
        The heterogeneity in the diffusion coefficients can be handled by
        ``conductivity`` array.
    """
    mesh = mesh.copy()
    mesh[mesh != 1] = 0  # Set fibrosis to 0
    weights = np.zeros((5, *mesh.shape))

    # Compute the diffusion term
    diffuse = D * np.ones(mesh.shape)

    if conductivity is not None:
        diffuse *= conductivity

    # Assign weights based on diffusion
    weights[0, :, :] = diffuse * dt / (dr**2) * np.roll(mesh, 1, axis=0)
    weights[1, :, :] = diffuse * dt / (dr**2) * np.roll(mesh, 1, axis=1)
    weights[3, :, :] = diffuse * dt / (dr**2) * np.roll(mesh, -1, axis=1)
    weights[4, :, :] = diffuse * dt / (dr**2) * np.roll(mesh, -1, axis=0)

    if conductivity is not None:
        # Adjust weights for heterogeneity
        diff_i = np.roll(diffuse, 1, axis=0) - np.roll(diffuse, -1, axis=0)
        diff_j = np.roll(diffuse, 1, axis=1) - np.roll(diffuse, -1, axis=1)

        weights[0, :, :] -= dt / (2*dr) * diff_i
        weights[1, :, :] -= dt / (2*dr) * diff_j
        weights[3, :, :] += dt / (2*dr) * diff_j
        weights[4, :, :] += dt / (2*dr) * diff_i

    # Finalize the weights
    for i in [0, 1, 3, 4]:
        weights[i, :, :] *= mesh
        weights[2, :, :] -= weights[i, :, :]
    weights[2, :, :] += 1
    weights[2, :, :] *= mesh

    return weights
