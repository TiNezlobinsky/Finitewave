import os
from math import sqrt
from numba import njit
import numpy as np
import warnings
from math import sqrt

from finitewave.core.tracker.tracker import Tracker


# @njit
def _calc_tippos(vij, vi1j, vi1j1, vij1, vnewij, vnewi1j,
             vnewi1j1, vnewij1, V_iso1, V_iso2):
    xy = [0, 0]
    # For old voltage values of point and neighbours
    AC=(vij-vij1+vi1j1-vi1j) # upleft-downleft + downright-upright
    GC=(vij1-vij)               # downleft-upleft
    BC=(vi1j-vij)               # upright-upleft
    DC=(vij-V_iso1)                 # upleft-iso

    # For current voltage values of point and neighbours
    AD=(vnewij-vnewij1+vnewi1j1-vnewi1j)
    GD=(vnewij1-vnewij)
    BD=(vnewi1j-vnewij)
    DD=(vnewij-V_iso2) # adapted here

    Q=(BC*AD-BD*AC)
    R=(GC*AD-GD*AC)
    S=(DC*AD-DD*AC)

    QOnR=Q/R
    SOnR=S/R

    T=AC*QOnR
    U=(AC*SOnR-BC+GC*QOnR)
    V=(GC*SOnR)-DC

    # Compute discriminant of the abc formula
    # with a=T, b=U and c=V
    Disc=U*U-4.*T*V
    if Disc<0:
        # If the discriminant is smaller than
        # zero there is no solution and a
        # failure flag should be returned
        return 0, xy
    else:

        # Otherwise two solutions for xvalues
        T2=2.*T
        sqrtDisc=sqrt(Disc)

        if T2 == 0.:
            return 0, [0,0]
        xn=(-U-sqrtDisc)/T2
        xp=(-U+sqrtDisc)/T2
        # Leading to two solutions for yvalues
        yn=-QOnR*xn-SOnR
        yp=-QOnR*xp-SOnR

        # demand that fractions lie in interval [0,1]
        if xn>=0 and xn<=1 and yn>=0 and yn<=1:
            # If the first point fulfills these
            # conditions take that point
            xy[0]=xn
            xy[1]=yn
            return 1, xy
        elif xp>=0 and xp<=1 and yp>=0 and yp<=1:
            # If the second point fulfills these
            # conditions take that point
            xy[0]=xp
            xy[1]=yp
            return 1, xy
        else:
            # If neither point fulfills these
            # conditions return a failure flag.
            return 0, xy


# @njit
def _track_tipline(size_i, size_j, var1, var2, tipvals, tipdata, tipsfound, mesh):
    iso1 = tipvals[0]
    iso2 = tipvals[1]

    # each row contains data for a point of the tipline
    # stored in columns: type, posx posy posz, dxu dyu dzu, dxv dyv dzv, tx ty tx
    # possible types: 1 = YZ, 11 = YZ at lower medium boundary, 21 = YZ at upper medium boundary,
    # 31 at lower domain boundary, 41 at upper domain boundary
    # 2 = XZ, 12 = XZ at lower medium boundary, 22 = XZ at upper medium boundary
    # 3 = XY, 13 = XY at lower medium boundary, 23 = XY at upper medium boundary
    # sign of type equals < T , e_i > (i.e. entering or leaving the voxel)

    tipsfound = 0
    delta = 5

    # check XY-planes
    for xpos in range(delta, size_i-delta):
        for ypos in range(delta, size_j-delta):
            # Just to fix some strange behaviour. Should be deleted.
            if mesh[xpos][ypos] != 1:
                continue
            if mesh[xpos+1][ypos] != 1:
                continue
            if mesh[xpos-1][ypos] != 1:
                continue
            if mesh[xpos][ypos+1] != 1:
                continue
            if mesh[xpos][ypos-1] != 1:
                continue
            if mesh[xpos+1][ypos+1] != 1:
                continue
            if mesh[xpos-1][ypos-1] != 1:
                continue
            if tipsfound >= 100:
                break
            counter = 0
            if var1[xpos][ypos] >= iso1 and \
                (var1[xpos+1][ypos] < iso1 or \
                var1[xpos][ypos+1] < iso1 or \
                var1[xpos+1][ypos+1] < iso1):
                counter = 1
            else:
                if var1[xpos][ypos] < iso1 and \
                    (var1[xpos+1][ypos] >= iso1 or \
                    var1[xpos][ypos+1] >= iso1 or \
                    var1[xpos+1][ypos+1] >= iso1):
                    counter = 1
            if counter == 1:
                if var2[xpos][ypos] >= iso2 and \
                    (var2[xpos+1][ypos] < iso2 or \
                    var2[xpos][ypos+1] < iso2 or \
                    var2[xpos+1][ypos+1] < iso2):
                    counter = 2
                else:
                    if var2[xpos][ypos] < iso2 and \
                        (var2[xpos+1][ypos] >= iso2 or \
                        var2[xpos][ypos+1] >= iso2 or \
                        var2[xpos+1][ypos+1] >= iso2):
                        counter = 2

                if counter == 2:
                    interp = _calc_tippos(var1[xpos, ypos], var1[xpos+1, ypos], var1[xpos+1, ypos+1], var1[xpos, ypos+1],
                                          var2[xpos, ypos], var2[xpos+1, ypos], var2[xpos+1, ypos+1], var2[xpos, ypos+1], iso1, iso2)
                    if interp[0] == 1:
                        tipdata[tipsfound][0] = xpos + interp[1][0]
                        tipdata[tipsfound][1] = ypos + interp[1][1]
                        tipsfound += 1

    return tipdata, tipsfound


class Spiral3DTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)
        self.size_i = 100
        self.size_j = 100
        self.size_k = 100
        self.dr = 0.25
        self.threshold = 0.2
        self.file_name = "swcore.txt"
        self.swcore = []

        self.all = False

        self.step = 1
        self._t   = 0

        self._u_prev_step = np.array([])

    def initialize(self, model):
        self.model = model

        self.size_i, self.size_j, self.size_k = self.model.cardiac_tissue.shape
        self.dt = self.model.dt
        self.dr = self.model.dr
        self._u_prev_step = np.zeros([self.size_i, self.size_j, self.size_k])
        self._tipdata = np.zeros([102, 2])

    def track_tipline(self, var1, var2, tipvals, tipdata, tipsfound, mesh):
        return _track_tipline(self.size_i, self.size_j, var1, var2, tipvals, tipdata, tipsfound, mesh)

    def track(self):
        if self._t > self.step:

            tipvals = [0, 0]
            tipvals[0] = self.threshold
            tipvals[1] = self.threshold

            tipsfound = 0
            sum = 0

            for k in range(self.size_k):

                self._tipdata, tipsfound = self.track_tipline(self._u_prev_step[:,:,k], self.model.u[:,:,k], tipvals, self._tipdata, tipsfound, self.model.cardiac_tissue.mesh[:,:,k])

                if self.all:
                    if not tipsfound:
                        self.swcore.append([self.model.t, 0, -1, -1])
                for i in range(tipsfound):
                    self.swcore.append([self.model.t, i+sum])
                    for j in range(2):
                        self.swcore[-1].append(self._tipdata[i][j]*self.dr)
                    self.swcore[-1].append(k*self.dr)

                sum += tipsfound
            self._u_prev_step = np.copy(self.model.u)
            self._t = 0
        else:
            self._t += self.dt

    def write(self):
        np.savetxt(os.path.join(self.path, self.file_name), np.array(self.swcore))

    @property
    def output(self):
        return self.swcore
