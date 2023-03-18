import os
import numpy as np


class StateKeeper:
    def __init__(self):
        self.record_save = ""
        self.record_load = ""

    def save(self, model):
        if not os.path.exists(self.record_save):
            os.makedirs(self.record_save)
        for var in model.state_vars:
            self._save_variable(os.path.join(self.record_save, var + ".npy"),
                                model.__dict__[var])

    def load(self, model):
        for var in model.state_vars:
            setattr(model, var, self._load_variable(os.path.join(
                        self.record_load, var + ".npy")))

    def _save_variable(self, var_path, var):
        np.save(var_path, var)

    def _load_variable(self, var_path):
        return np.load(var_path)
