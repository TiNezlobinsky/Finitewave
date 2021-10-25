import unittest
import numpy as np
import matplotlib.pyplot as plt
import sys

from finitewave.cpuwave2D.tissue.cardiac_tissue_2d import CardiacTissue2D
from finitewave.cpuwave2D.fibrosis.diffuse_2d_pattern import Diffuse2DPattern 


class TestFibrosisGeneration2D(unittest.TestCase):
    def setUp(self):

        self.n = 500
        self.tissue = CardiacTissue2D([self.n, self.n])
        self.tissue.mesh = np.ones([self.n, self.n], dtype="uint8")
        self.tissue.add_boundaries()

    def test_diffuse_pattern(self):
        sys.stdout.write("---> Check the diffuse fibrosis pattern\n")
        diffuse = Diffuse2DPattern(0, self.n, 0, self.n, 0.37)
        matrix = diffuse.generate([self.n, self.n])
        percentage = len(matrix[matrix == 2])/self.n**2 
        self.assertAlmostEqual(percentage, 0.37,
                               msg="Diffuse fibrosis percentage is incorrect! (matrix getter)",
                               delta=0.01)

        diffuse.apply(self.tissue)
        matrix = self.tissue.mesh[1:499, 1:499]
        percentage = len(matrix[matrix == 2])/(self.n-2)**2 
        self.assertAlmostEqual(percentage, 0.37,
                               msg="Diffuse fibrosis percentage is incorrect! (apply method)",
                               delta=0.01)


