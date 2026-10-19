import numpy as np


class SingleCellStimulation:
    """Build a current trace for a single-cell simulation."""

    def __init__(self, dt):
        if dt <= 0:
            raise ValueError("dt must be positive.")
        self.dt = dt
        self._segments = []

    def add_stimulation(self, n_beats, cycle_length, current, duration):
        """Append a periodic rectangular-current segment."""
        if n_beats < 1:
            raise ValueError("n_beats must be positive.")
        if cycle_length <= 0:
            raise ValueError("cycle_length must be positive.")
        if duration < 0 or duration > cycle_length:
            raise ValueError("duration must be between zero and cycle_length.")

        steps = int(n_beats * cycle_length / self.dt)
        segment = np.zeros(steps, dtype=np.float64)

        for beat in range(n_beats):
            start = int(beat * cycle_length / self.dt)
            stop = int((beat * cycle_length + duration) / self.dt)
            segment[start:stop] = current

        self._segments.append(segment)
        return self

    @property
    def stim_current(self):
        if not self._segments:
            return np.empty(0, dtype=np.float64)
        return np.concatenate(self._segments)

    @stim_current.setter
    def stim_current(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("stim_current must be one-dimensional.")
        self._segments = [values.copy()]
