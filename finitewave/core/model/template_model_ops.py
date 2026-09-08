
from importlib.metadata import entry_points


def _discover() -> dict:
    """Return installed Finitewave model entry points keyed by name."""
    eps = entry_points()
    group = "finitewave.models"
    if hasattr(eps, "select"):
        selected = eps.select(group=group)
    else:
        selected = eps.get(group, [])
    return {ep.name: ep for ep in selected}



def load_template_model_ops(model_name: str):
    """Load and validate a Finitewave model operations plugin.
    
    Parameters
    ----------
    model_name : str
        The name of the model to load, which should correspond to an entry
        point in the ``finitewave.models`` group.

    Returns
    -------
    module
        Operations module providing ``get_variables``, ``get_parameters``,
        ``get_diffusion_coefficient``, and ``ionic_step``.

    Raises
    ------
    KeyError
        If no installed entry point matches ``model_name``.
    ValueError
        If the plugin does not provide a required operation.
    """
    REQS = ("get_variables", "get_parameters", "ionic_step", "get_diffusion_coefficient")

    eps = _discover()
    if model_name not in eps:
        raise KeyError(
            f"Model '{model_name}' not found via entry point group "
            "'finitewave.models'."
        )
    
    mod = eps[model_name].load()   # ops package
    ops = getattr(mod, "ops", mod)

    for name in REQS:
        if not hasattr(ops, name):
            raise ValueError(f"Model '{model_name}' missing '{name}' in ops.")
    return ops
