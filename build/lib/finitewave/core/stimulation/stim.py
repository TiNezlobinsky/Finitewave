

class Stim:

    def __init__(self, start_time, current=0, voltage=0, duration=0):
        self.start_time = start_time
        self.end_time = start_time + duration
        self.duration = duration
        self.current = current
        self.voltage = voltage

    def initialize(self, model):
        self.prepare_coords(model.cardiac_tissue.mesh)
        self.prepare_end_time(model.dt)

    def check_status(self, time):
        return (time >= self.start_time) & (time < self.end_time)

    def stimulate(self, model):
        pass

    def prepare_coords(self, mesh):
        mask = mesh[tuple(self.coords.T)] == 1
        self.coords = tuple(self.coords[mask].T)

    def prepare_end_time(self, dt):
        self.duration = max(dt, self.duration)
        self.end_time = self.start_time + self.duration
