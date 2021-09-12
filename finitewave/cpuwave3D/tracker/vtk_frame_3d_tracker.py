import os
import vtk
import numpy as np

from finitewave.core.tracker.tracker import Tracker


class VTKFrame3DTracker(Tracker):
    def __init__(self):
        Tracker.__init__(self)
        self.step = 5
        self._t   = 0

        self.file_name = "vtk_frames"

        self._frame_n = 0

        self.target_array = ""

    def initialize(self, model):
        self.model = model
        
        self._t   = 0
        self._frame_n = 0
        self._dt  = self.model.dt

        if not os.path.exists(os.path.join(self.path, self.file_name)):
            os.makedirs(os.path.join(self.path, self.file_name))

    def track(self):
        if self._t > self.step:
            frame_name = os.path.join(self.path, self.file_name, "frame" + str(self._frame_n) + ".vtk")
            self.write_vtk_unstructured_grid(frame_name, self.create_vtk_mesh_3D(self.model.cardiac_tissue.mesh,
                                                                                 self.model.u))

            self._frame_n += 1
            self._t = 0
        else:
            self._t += self._dt

    def create_vtk_mesh_3D(self, np_mesh, np_scalar=None):
        unstructured_grid = vtk.vtkUnstructuredGrid()

        points= vtk.vtkPoints()
        number = np_mesh[np_mesh == 1].size
        points.SetNumberOfPoints(number)

        if np_scalar is not None:
            scalar_array = vtk.vtkFloatArray()
            scalar_array.SetNumberOfComponents(1)
            scalar_array.SetName("Scalars")

        n, m, s = np_mesh.shape
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(s):
                    if np_mesh[i, j, k] == 1:
                        points.InsertPoint(idx, i, j, k)
                        vertex = vtk.vtkVertex()
                        vertex.GetPointIds().SetId(0, idx)
                        unstructured_grid.InsertNextCell(vertex.GetCellType(),
                                                         vertex.GetPointIds())
                        idx += 1
                        if np_scalar is not None:
                            scalar_array.InsertNextTuple1(np_scalar[i, j, k])

        unstructured_grid.SetPoints(points)

        if np_scalar is not None:
            unstructured_grid.GetPointData().SetScalars(scalar_array)

        return unstructured_grid

    def write_vtk_unstructured_grid(self, file_name, unstructured_grid):
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(file_name)
        writer.SetInputData(unstructured_grid)
        writer.Write()
