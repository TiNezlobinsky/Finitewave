from pathlib import Path
import shutil as shatilib
from finitewave.cpuwave2D.tracker.period_animation_2d_tracker import (
    PeriodAnimation2DTracker
)
from finitewave.tools.animation_3d_builder import Animation3DBuilder


class PeriodAnimation3DTracker(PeriodAnimation2DTracker):
    """
    Class for tracking 3D period map. Initializes PeriodAnimation2DTracker.
    """
    def __init__(self):
        super().__init__()

    def write(self, clim=[0, 1], cmap="viridis", scalar_bar=False, clear=True,
              prog_bar=True, **kwargs):

        animation_builder = Animation3DBuilder()
        animation_builder.write(Path(self.path, self.dir_name),
                                path_save=self.path,
                                mask=self.model.cardiac_tissue.mesh,
                                scalar_name='Period',
                                clim=clim, cmap=cmap,
                                scalar_bar=scalar_bar, format='mp4',
                                prog_bar=prog_bar, **kwargs)

        if clear:
            shatilib.rmtree(Path(self.path, self.dir_name))
