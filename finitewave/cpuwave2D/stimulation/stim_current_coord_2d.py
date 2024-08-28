from finitewave.core.stimulation.stim_current import StimCurrent


class StimCurrentCoord2D(StimCurrent):
    def __init__(self, time, curr_value, curr_time, x1, x2, y1, y2):
        StimCurrent.__init__(self, time, curr_value, curr_time)
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def stimulate(self, model):
        if not self.passed:
            # ROI - region of interest
            roi_x1, roi_x2 = self.x1, self.x2
            roi_y1, roi_y2 = self.y1, self.y2

            roi_mesh = model.cardiac_tissue.mesh[roi_x1:roi_x2, roi_y1:roi_y2]

            mask = (roi_mesh == 1)

            model.u[roi_x1:roi_x2, roi_y1:roi_y2][mask] += self._dt*self.curr_value
