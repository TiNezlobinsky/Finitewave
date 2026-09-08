from .cardiac_tissue_elements import CardiacTissueElements
from .cardiac_tissue_grid import CardiacTissueGrid


class CardiacTissue:
    """Create grid or element tissue from keyword parameters.

    Supply ``dr`` and either ``shape``, ``mesh``, or both for a grid.
    When both are supplied, ``shape`` must match ``mesh.shape``.
    Supply ``coords``, ``elems`` and
    ``elem_type`` for an element mesh (with optional ``order``).
    Mixing parameters from the two representations is not supported.

    Returns
    -------
    CardiacTissueGrid or CardiacTissueElements
        A fully initialized tissue. This class is a factory, not a base
        class; use ``CardiacTissueBase`` for shared type checks.

    Examples
    --------
    >>> tissue = CardiacTissue(shape=(100, 100), dr=0.25)
    """

    def __new__(cls, shape=None, dr=None, mesh=None, coords=None, elems=None, elem_type=None, order=1):
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
