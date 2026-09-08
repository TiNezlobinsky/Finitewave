from finitewave.core.model.template_model_ops import load_template_model_ops as load_ops
from .cardiac_model import CardiacModel

__all__ = [
    "AlievPanfilov",
    "Barkley",
    "BuenoOrovio",
    "Courtemanche",
    "FentonKarma",
    "LuoRudy91",
    "MitchellSchaeffer",
    "TenTusscherPanfilov2006",
]

class AlievPanfilov(CardiacModel):
    """
    Aliev-Panfilov model for cardiac electrophysiology.
    """
    def __init__(self):
        super().__init__()
        ops = load_ops("aliev_panfilov")
        initialize_from_ops(self, ops)


class Barkley(CardiacModel):
    """
    Barkley model for excitable media.
    """
    def __init__(self):
        super().__init__()
        ops = load_ops("barkley")
        initialize_from_ops(self, ops)


class BuenoOrovio(CardiacModel):
    """
    Bueno-Orovio model for cardiac electrophysiology.
    """
    def __init__(self):
        super().__init__()
        ops = load_ops("bueno_orovio")
        initialize_from_ops(self, ops)


class Courtemanche(CardiacModel):
    """
    Courtemanche model for human atrial electrophysiology.
    """
    def __init__(self):
        super().__init__()
        ops = load_ops("courtemanche")
        initialize_from_ops(self, ops)


class FentonKarma(CardiacModel):
    """
    Fenton-Karma model for cardiac electrophysiology.
    """
    def __init__(self):
        super().__init__()
        ops = load_ops("fenton_karma")
        initialize_from_ops(self, ops)


class LuoRudy91(CardiacModel):
    """
    Luo-Rudy 1991 model for cardiac electrophysiology.
    """
    def __init__(self):
        super().__init__()
        ops = load_ops("luo_rudy_91")
        initialize_from_ops(self, ops)


class MitchellSchaeffer(CardiacModel):
    """
    Mitchell-Schaeffer model for cardiac electrophysiology.
    """
    def __init__(self):
        super().__init__()
        ops = load_ops("mitchell_schaeffer")
        initialize_from_ops(self, ops)


class TenTusscherPanfilov2006(CardiacModel):
    """
    Ten Tusscher-Panfilov 2006 model for human ventricular electrophysiology.
    """
    def __init__(self):
        super().__init__()
        ops = load_ops("ten_tusscher_panfilov_2006")
        initialize_from_ops(self, ops)


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
