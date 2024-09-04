import os
from math import sqrt
import numpy as np
import warnings
from numba import njit

from finitewave.core.tracker.tracker import Tracker


@njit
def _calc_tippos(vij, vi1j, vi1j1, vij1, vnewij, vnewi1j, vnewi1j1, vnewij1, V_iso1, V_iso2):
    """
    Calculate the position of the tip of a spiral wave in a 2D grid based on the voltage values.

    This function uses bilinear interpolation to determine the precise position of the spiral tip
    by finding the crossing point of voltage levels (`V_iso1` and `V_iso2`).

    Parameters
    ----------
    vij, vi1j, vi1j1, vij1 : float
        Old voltage values at the current and neighboring grid points.
    vnewij, vnewi1j, vnewi1j1, vnewij1 : float
        New voltage values at the current and neighboring grid points.
    V_iso1, V_iso2 : float
        Isoline voltage values used for detecting spiral tips.

    Returns
    -------
    int
        1 if a tip is found, 0 otherwise.
    list of float
        The (x, y) position of the tip if found; otherwise, [0, 0].
    """
    xy = [0, 0]
    # Compute various differences for both old and new voltage values
    AC = (vij - vij1 + vi1j1 - vi1j)
    GC = (vij1 - vij)
    BC = (vi1j - vij)
    DC = (vij - V_iso1)

    AD = (vnewij - vnewij1 + vnewi1j1 - vnewi1j)
    GD = (vnewij1 - vnewij)
    BD = (vnewi1j - vnewij)
    DD = (vnewij - V_iso2)

    # Compute intermediate values for solving the system of equations
    Q = (BC * AD - BD * AC)
    R = (GC * AD - GD * AC)
    S = (DC * AD - DD * AC)

    QOnR = Q / R
    SOnR = S / R

    T = AC * QOnR
    U = (AC * SOnR - BC + GC * QOnR)
    V = (GC * SOnR) - DC

    # Calculate the discriminant for the quadratic formula
    Disc = U * U - 4. * T * V
    if Disc < 0:
        return 0, xy  # No solution
    else:
        # Two possible solutions for (x, y) coordinates
        T2 = 2. * T
        sqrtDisc = sqrt(Disc)

        if T2 == 0.:
            return 0, [0, 0]
        xn = (-U - sqrtDisc) / T2
        xp = (-U + sqrtDisc) / T2
        yn = -QOnR * xn - SOnR
        yp = -QOnR * xp - SOnR

        # Ensure the points lie within the valid grid range
        if 0 <= xn <= 1 and 0 <= yn <= 1:
            xy[0] = xn
            xy[1] = yn
            return 1, xy
        elif 0 <= xp <= 1 and 0 <= yp <= 1:
            xy[0] = xp
            xy[1] = yp
            return 1, xy
        else:
            return 0, xy


@njit
def _track_tipline(size_i, size_j, var1, var2, tipvals, tipdata, tipsfound):
    """
    Track spiral wave tips in a 2D grid by detecting crossings of voltage isolines.

    This function searches for spiral tips in XY planes by detecting where the voltage crosses
    specified thresholds in both the old and new voltage values.

    Parameters
    ----------
    size_i, size_j : int
        Dimensions of the 2D grid.
    var1, var2 : np.ndarray
        2D arrays representing the old and new voltage values.
    tipvals : list of float
        Isoline voltage values used for detecting spiral tips.
    tipdata : np.ndarray
        Array to store the coordinates of detected tips.
    tipsfound : int
        Counter for the number of detected tips.

    Returns
    -------
    np.ndarray
        Updated array containing detected tip coordinates.
    int
        Updated count of detected tips.
    """
    iso1 = tipvals[0]
    iso2 = tipvals[1]
    delta = 5  # Safety margin to avoid boundary effects

    # Iterate through the grid to find spiral tips
    for xpos in range(delta, size_i - delta):
        for ypos in range(delta, size_j - delta):
            if tipsfound >= 100:  # Limit the number of tips detected
                break
            counter = 0
            # Check for crossings of the first isoline in the XY plane
            if var1[xpos][ypos] >= iso1 and \
                (var1[xpos + 1][ypos] < iso1 or
                 var1[xpos][ypos + 1] < iso1 or
                 var1[xpos + 1][ypos + 1] < iso1):
                counter = 1
            elif var1[xpos][ypos] < iso1 and \
                (var1[xpos + 1][ypos] >= iso1 or
                 var1[xpos][ypos + 1] >= iso1 or
                 var1[xpos + 1][ypos + 1] >= iso1):
                counter = 1

            if counter == 1:
                # Check for crossings of the second isoline in the XY plane
                if var2[xpos][ypos] >= iso2 and \
                    (var2[xpos + 1][ypos] < iso2 or
                     var2[xpos][ypos + 1] < iso2 or
                     var2[xpos + 1][ypos + 1] < iso2):
                    counter = 2
                elif var2[xpos][ypos] < iso2 and \
                    (var2[xpos + 1][ypos] >= iso2 or
                     var2[xpos][ypos + 1] >= iso2 or
                     var2[xpos + 1][ypos + 1] >= iso2):
                    counter = 2

                # If both crossings are detected, compute the precise position of the tip
                if counter == 2:
                    interp = _calc_tippos(var1[xpos, ypos], var1[xpos + 1, ypos], var1[xpos + 1, ypos + 1], var1[xpos, ypos + 1],
                                          var2[xpos, ypos], var2[xpos + 1, ypos], var2[xpos + 1, ypos + 1], var2[xpos, ypos + 1], iso1, iso2)
                    if interp[0] == 1:
                        tipdata[tipsfound][0] = xpos + interp[1][0]
                        tipdata[tipsfound][1] = ypos + interp[1][1]
                        tipsfound += 1

    return tipdata, tipsfound


class Spiral2DTracker(Tracker):
    """
    A class to track spiral wave tips in a 2D cardiac tissue model.

    This tracker identifies and records the positions of spiral wave tips by analyzing
    voltage isoline crossings in the simulated 2D grid over time.

    Attributes
    ----------
    size_i, size_j : int
        Dimensions of the 2D grid.
    dr : float
        Grid spacing in the model.
    threshold : float
        Voltage threshold value for detecting spiral tips.
    file_name : str
        Name of the output file where spiral tip data is saved.
    swcore : list
        List to store detected spiral wave core positions.
    all : bool
        Flag to determine whether all tips or only first few are tracked.
    step : int
        Interval of steps for saving the spiral wave tips.
    _t : float
        Internal timer to track the current simulation time.
    _u_prev_step : np.ndarray
        Array to store the voltage values from the previous time step.
    _tipdata : np.ndarray
        Array to store the detected tip coordinates.

    Methods
    -------
    initialize(model):
        Initializes the tracker with the simulation model.
    track_tipline(var1, var2, tipvals, tipdata, tipsfound):
        Wrapper function for the low-level _track_tipline function.
    track():
        Tracks spiral tips at each simulation step.
    write():
        Saves the tracked spiral tip data to a file.
    output:
        Property that returns the tracked spiral core data.
    """

    def __init__(self):
        """
        Initializes the Spiral2DTracker with default parameters.
        """
        Tracker.__init__(self)
        self.size_i = 100
        self.size_j = 100
        self.dr = 0.25
        self.threshold = 0.2
        self.file_name = "swcore.txt"
        self.swcore = []

        self.all = False
        self.step = 1
        self._t = 0
        self._u_prev_step = np.array([])

    def initialize(self, model):
        """
        Initialize the tracker with the given cardiac tissue model.

        Parameters
        ----------
        model : object
            The cardiac tissue simulation model containing the grid and voltage data.
        """
        self.model = model
        self.size_i, self.size_j = self.model.cardiac_tissue.shape
        self.dt = self.model.dt
        self.dr = self.model.dr
        self._u_prev_step = np.zeros([self.size_i, self.size_j])
        self._tipdata = np.zeros([102, 2])

    def track_tipline(self, var1, var2, tipvals, tipdata, tipsfound):
        """
        High-level function to track spiral tips in the 2D grid.

        Parameters
        ----------
        var1, var2 : np.ndarray
            2D arrays representing the old and new voltage values.
        tipvals : list of float
            Isoline voltage values used for detecting spiral tips.
        tipdata : np.ndarray
            Array to store the coordinates of detected tips.
        tipsfound : int
            Counter for the number of detected tips.

        Returns
        -------
        tuple
            Updated array of detected tip coordinates and the count of detected tips.
        """
        return _track_tipline(self.size_i, self.size_j, var1, var2, tipvals, tipdata, tipsfound)

    def track(self):
        """
        Track spiral tips at each simulation step by analyzing voltage data.

        The tracker is updated at each simulation step, detecting any spiral tips
        based on the voltage data from the previous and current steps.
        """
        if self._t > self.step:
            tipvals = [0, 0]
            tipvals[0] = self.threshold
            tipvals[1] = self.threshold

            tipsfound = 0

            self._tipdata, tipsfound = self.track_tipline(self._u_prev_step, self.model.u, tipvals, self._tipdata, tipsfound)

            if self.all:
                if not tipsfound:
                    self.swcore.append([self.model.t, 0, -1, -1])

            for i in range(tipsfound):
                self.swcore.append([self.model.t, i])
                for j in range(2):
                    self.swcore[-1].append(self._tipdata[i][j] * self.dr)

            self._u_prev_step = np.copy(self.model.u)
            self._t = 0
        else:
            self._t += self.dt

    def write(self):
        """
        Save the tracked spiral tip data to a file.
        """
        np.savetxt(os.path.join(self.path, self.file_name), np.array(self.swcore))

    @property
    def output(self):
        """
        Get the tracked spiral core data.

        Returns
        -------
        list
            List of tracked spiral core data.
        """
        return self.swcore
