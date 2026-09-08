

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


def load_ops(model_name: str):
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


def initialize_from_ops(model, ops):
    """Load defaults exposed by the model operations plugin.
    """
    model.ops = ops
    model.default_parameters = ops.get_parameters()
    model.default_variables = ops.get_variables()
    model.D_model = ops.get_diffusion_coefficient()["D_model"]

    model.state_vars = list(model.default_variables.keys())
    model.state_pars = list(model.default_parameters.keys())

    # expose parameters as direct attributes (scalar or array)
    for name, value in model.default_parameters.items():
        setattr(model, name, value)

    # expose initial conditions as init_*
    for name, value in model.default_variables.items():
        setattr(model, f"init_{name}", value)

    # declare arrays (optional, for readability/debug)
    for name in model.default_variables.keys():
        setattr(model, f"_{name}", None)
