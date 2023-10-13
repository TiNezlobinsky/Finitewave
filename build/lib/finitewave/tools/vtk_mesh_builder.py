import vtk


class VTKMeshBuilder:
    def __init__(self):
        pass

    @staticmethod
    def create_vtk_mesh_3D(np_mesh, np_fibers=None):
        unstructured_grid = vtk.vtkUnstructuredGrid()

        points= vtk.vtkPoints()
        number = np_mesh[np_mesh == 1].size
        points.SetNumberOfPoints(number)

        if np_fibers is not None:
            fibers_array = vtk.vtkDoubleArray()
            fibers_array.SetNumberOfComponents(3)
            fibers_array.SetName("Fibers")

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
                        if np_fibers is not None:
                            fibers_array.InsertNextTuple3(np_fibers[i, j, k, 0], np_fibers[i, j, k, 1], np_fibers[i, j, k, 2])

        unstructured_grid.SetPoints(points)

        if np_fibers is not None:
            unstructured_grid.GetPointData().SetVectors(fibers_array)

        return unstructured_grid

    @staticmethod
    def write_vtk_unstructured_grid(file_name, unstructured_grid):
        writer = vtk.vtkUnstructuredGridWriter()
        writer.SetFileName(file_name)
        writer.SetInputData(unstructured_grid)
        writer.Write()
