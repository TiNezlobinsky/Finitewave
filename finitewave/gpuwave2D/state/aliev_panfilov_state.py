from finitewave.core.state.state_keeper import StateKeeper

import os

class AlievPanfilovState(StateKeeper):
    def __init__(self):
        StateKeeper.__init__(self)

    def save(self, model):
        if not os.path.exists(self.record_save):
            os.makedirs(self.record_save)

        model.load_state()
        self._save_variable(os.path.join(self.record_save, "u.npy"), model.u)
        self._save_variable(os.path.join(self.record_save, "v.npy"), model.v)

    def load(self, model):
        model.u = self._load_variable(os.path.join(self.record_load, "u.npy")).astype('float32')
        model.v = self._load_variable(os.path.join(self.record_load, "v.npy")).astype('float32')
