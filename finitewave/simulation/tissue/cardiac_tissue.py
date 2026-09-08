from .cardiac_tissue_elements import CardiacTissueElements
from .cardiac_tissue_grid import CardiacTissueGrid


class CardiacTissue:
    """Create grid or element tissue from keyword parameters.

    Parameters
    ----------
    shape : tuple of int, optional
        Shape of the tissue grid (number of cells in each dimension). Required for grid tissue.
    dr : float, optional
        Spatial resolution of the tissue grid. Required for grid tissue.
    mesh : array-like, optional
        Optional mesh array for grid tissue. If not provided, a uniform grid is created.
    coords : array-like, optional
        Coordinates of the tissue elements. Required for element tissue.
    elems : array-like, optional
        Connectivity of the tissue elements. Required for element tissue.
    elem_type : str or ElementType, optional
        Type of the tissue elements (e.g., "Triangle", "Tetrahedron"). Required for element tissue.
    order : int, optional
        Order of the finite elements (1 for linear, 2 for quadratic).
        Default is 1. Only applicable for element tissue.

    Returns
    -------
    CardiacTissueGrid or CardiacTissueElements
        A fully initialized tissue. This class is a factory, not a base
        class; use ``CardiacTissueBase`` for shared type checks.

    Examples
    --------
    >>> tissue = CardiacTissue(shape=(100, 100), dr=0.25)
    >>> tissue = CardiacTissue(mesh=mesh, dr=0.25)
    >>> tissue = CardiacTissue(coords=coords, elems=elems, elem_type="Triangle")
    >>> tissue = CardiacTissue(coords=coords, elems=elems, elem_type=fw.ElementType.TETRAHEDRON)
    """

    def __new__(cls, shape=None, dr=None, *, mesh=None, coords=None, elems=None, elem_type=None, order=1):
        is_grid = (shape is not None or mesh is not None) and (dr is not None)
        is_elements = (coords is not None and elems is not None and elem_type is not None)
        if is_grid and is_elements:
            raise TypeError("Cannot mix grid and element tissue parameters.")
        if is_grid:
            return CardiacTissueGrid(shape=shape, dr=dr, mesh=mesh)
        if is_elements:
            return CardiacTissueElements(coords=coords, elems=elems, elem_type=elem_type, order=order)
        raise TypeError(
            "Supply dr and shape or mesh for grid tissue, or coords, elems and "
            "elem_type for element tissue."
        )
