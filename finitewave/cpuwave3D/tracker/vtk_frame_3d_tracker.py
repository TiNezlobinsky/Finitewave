from pathlib import Path

from finitewave.tools.vis_mesh_builder_3d import VisMeshBuilder3D
from finitewave.core.tracker.tracker import Tracker


class VTKFrame3DTracker(Tracker):
    """
    A class for tracking and saving VTK frames in a 3D model.

    Attributes:
        step (int): The step size for tracking frames.
        file_name (str): The name of the file to save the frames
                         ".vtk" or ".vtu"
        target_array (str): The name of the target array to be tracked.
        file_type (str): The file type of the saved frames.
    """
    def __init__(self):
        Tracker.__init__(self)
        self.step = 5
        self.file_name = "vtk_frames"
        self.target_array = ""
        self.file_type = ".vtk"

        self._t = 0
        self._frame_n = 0

    def initialize(self, model):
        self.model = model

        self._t = 0
        self._frame_n = 0
        self._dt = self.model.dt

        self.path = Path(self.path)

        if not self.path.joinpath(self.file_name).exists():
            self.path.joinpath(self.file_name).mkdir(parents=True)

        if self.target_array == "":
            raise ValueError("Please specify the target array to be tracked.")

        if self.target_array not in self.model.__dict__:
            raise ValueError(f"Array {self.target_array} not found in model.")

    def track(self):
        if self._t > self.step:
            frame_name = self.path.joinpath(self.file_name,
                                            f"frame{self._frame_n}"
                                            ).with_suffix(self.file_type)

            self.write_frame(frame_name)
            self._frame_n += 1
            self._t = 0
        else:
            self._t += self._dt

    def write_frame(self, frame_name):
        state_var = self.model.__dict__[self.target_array]

        vtk_mesh_builder = VisMeshBuilder3D()
        vtk_mesh = vtk_mesh_builder.build_mesh(self.model.cardiac_tissue.mesh)
        vtk_mesh = vtk_mesh_builder.add_scalar(state_var, self.target_array)
        vtk_mesh.save(frame_name)
