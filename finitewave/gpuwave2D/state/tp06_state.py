from finitewave.core.state.state_keeper import StateKeeper

import os

class TP06State(StateKeeper):
    def __init__(self):
        StateKeeper.__init__(self)

    def save(self, model):
        if not os.path.exists(self.record_save):
            os.makedirs(self.record_save)

        model.load_state()

        self._save_variable(os.path.join(self.record_save, "u.npy"), model.u)
        self._save_variable(os.path.join(self.record_save, "Cai.npy"), model.Cai)
        self._save_variable(os.path.join(self.record_save, "CaSR.npy"), model.CaSR)
        self._save_variable(os.path.join(self.record_save, "CaSS.npy"), model.CaSS)
        self._save_variable(os.path.join(self.record_save, "Nai.npy"), model.Nai)
        self._save_variable(os.path.join(self.record_save, "Ki.npy"), model.Ki)
        self._save_variable(os.path.join(self.record_save, "M.npy"), model.M_)
        self._save_variable(os.path.join(self.record_save, "H.npy"), model.H_)
        self._save_variable(os.path.join(self.record_save, "J.npy"), model.J_)
        self._save_variable(os.path.join(self.record_save, "Xr1.npy"), model.Xr1)
        self._save_variable(os.path.join(self.record_save, "Xr2.npy"), model.Xr2)
        self._save_variable(os.path.join(self.record_save, "Xs.npy"), model.Xs)
        self._save_variable(os.path.join(self.record_save, "R.npy"), model.R_)
        self._save_variable(os.path.join(self.record_save, "S.npy"), model.S_)
        self._save_variable(os.path.join(self.record_save, "D.npy"), model.D_)
        self._save_variable(os.path.join(self.record_save, "F.npy"), model.F_)
        self._save_variable(os.path.join(self.record_save, "F2.npy"), model.F2_)
        self._save_variable(os.path.join(self.record_save, "FCass.npy"), model.FCass)
        self._save_variable(os.path.join(self.record_save, "RR.npy"), model.RR)
        self._save_variable(os.path.join(self.record_save, "OO.npy"), model.OO)

    def load(self, model):
        model.u     = self._load_variable(os.path.join(self.record_load, "u.npy")).astype('float32')
        model.Cai   = self._load_variable(os.path.join(self.record_load, "Cai.npy")).astype('float32')
        model.CaSR  = self._load_variable(os.path.join(self.record_load, "CaSR.npy")).astype('float32')
        model.CaSS  = self._load_variable(os.path.join(self.record_load, "CaSS.npy")).astype('float32')
        model.Nai   = self._load_variable(os.path.join(self.record_load, "Nai.npy")).astype('float32')
        model.Ki    = self._load_variable(os.path.join(self.record_load, "Ki.npy")).astype('float32')
        model.M_    = self._load_variable(os.path.join(self.record_load, "M.npy")).astype('float32')
        model.H_    = self._load_variable(os.path.join(self.record_load, "H.npy")).astype('float32')
        model.J_    = self._load_variable(os.path.join(self.record_load, "J.npy")).astype('float32')
        model.Xr1   = self._load_variable(os.path.join(self.record_load, "Xr1.npy")).astype('float32')
        model.Xr2   = self._load_variable(os.path.join(self.record_load, "Xr2.npy")).astype('float32')
        model.Xs    = self._load_variable(os.path.join(self.record_load, "Xs.npy")).astype('float32')
        model.R_    = self._load_variable(os.path.join(self.record_load, "R.npy")).astype('float32')
        model.S_    = self._load_variable(os.path.join(self.record_load, "S.npy")).astype('float32')
        model.D_    = self._load_variable(os.path.join(self.record_load, "D.npy")).astype('float32')
        model.F_    = self._load_variable(os.path.join(self.record_load, "F.npy")).astype('float32')
        model.F2_   = self._load_variable(os.path.join(self.record_load, "F2.npy")).astype('float32')
        model.FCass = self._load_variable(os.path.join(self.record_load, "FCass.npy")).astype('float32')
        model.RR    = self._load_variable(os.path.join(self.record_load, "RR.npy")).astype('float32')
        model.OO    = self._load_variable(os.path.join(self.record_load, "OO.npy")).astype('float32')
