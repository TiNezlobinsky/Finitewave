import math
import numpy as np
from scipy.optimize import curve_fit


def _line(x, a, b):
    return a*x + b


class DriftVelocityCalculation:
    def __init__(self):
        self.swcore = []
        self.time_span = 0.

    def compute_drift(self):
        swcore = np.array(self.swcore)
        step   = swcore[1, 0] - swcore[0, 0]

        indx = int(self.time_span/step)

        time   = swcore[-indx:, 0]
        comp_x = swcore[-indx:, 2]
        comp_y = swcore[-indx:, 3]

        a_x, b_x = curve_fit(_line, time, comp_x)[0]
        a_y, b_y = curve_fit(_line, time, comp_y)[0]

        fit_x = _line(time, a_x, b_x)
        fit_y = _line(time, a_y, b_y)

        drift_x = (fit_x[-1] - fit_x[0])/(time[-1] - time[0])
        drift_y = (fit_y[-1] - fit_y[0])/(time[-1] - time[0])

        return drift_x, drift_y, math.sqrt(drift_x**2 + drift_y**2)
