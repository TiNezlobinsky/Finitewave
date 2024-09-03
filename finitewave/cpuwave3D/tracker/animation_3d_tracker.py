from pathlib import Path
import numpy as np
import pyvista as pv
import shutil as shatilib

from finitewave.core.tracker.tracker import Tracker
from finitewave.tools.vis_mesh_builder_3d import VisMeshBuilder3D
from finitewave.tools.animation_3d_builder import Animation3DBuilder


class Animation3DTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)
        self.step = 1
        self.start = 0
        self.target_array = ""
        self.dir_name = "animation"

        self._t = 0
        self._frame_n = 0

    def initialize(self, model):
        self.model = model

        self._t = 0
        self._frame_n = 0
        self._dt = self.model.dt
        self._step = self.step - self._dt

        if not Path(self.path).joinpath(self.dir_name).exists():
            Path(self.path).joinpath(self.dir_name).mkdir(parents=True)

    def track(self):
        path = Path(self.path)
        if not self.model.t >= self.start:
            return

        if self._t > self._step:
            frame = self.model.__dict__[self.target_array]
            np.save(path.joinpath(self.dir_name, f"{self._frame_n}.npy"),
                    frame)
            self._frame_n += 1
            self._t = 0
        else:
            self._t += self._dt

    def write(self, path=None, clim=[0, 1], cmap="viridis", scalar_bar=False,
              format="mp4", clear=False, **kwargs):
        """Write the animation to a file.

        Args:
            path (str, optional): Path to save the animation.
                Defaults is path of the tracker.
            clim (list, optional): Color limits. Defaults to [0, 1].
            cmap (str, optional): Color map. Defaults to "viridis".
            scalar_bar (bool, optional): Show scalar bar. Defaults to False.
            format (str, optional): Format of the animation. Defaults to "mp4".
                Other options are "gif".
            clear (bool, optional): Clear the snapshot folder after writing
                the animation. Defaults to False.
            **kwargs: Additional arguments for the animation writer.
        """

        if path is None:
            path = self.path

        animation_builder = Animation3DBuilder()
        animation_builder.write(Path(self.path).joinpath(self.dir_name),
                                path_save=path,
                                mask=self.model.cardiac_tissue.mesh,
                                scalar_name=self.target_array,
                                clim=clim, cmap=cmap,
                                scalar_bar=scalar_bar, format=format, **kwargs)

        if clear:
            shatilib.rmtree(Path(self.path).joinpath(self.dir_name))
