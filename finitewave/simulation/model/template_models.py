from finitewave.core.model.template_model import load_ops, initialize_from_ops
from .cardiac_model import CardiacModel


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
