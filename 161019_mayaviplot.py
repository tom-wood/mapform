import numpy as np
from mayavi import mlab

class AtomStructure:
    def __init__(self, anions, cations):
        self.anions = anions
        self.cations = cations
    def get_uc_vertices(self):
        x, y, z = [l / 2. for l in self.anions.shape]
        ucv = np.array([[0, 0, 0], [x, 0, 0], [0, y, 0], [0, 0, z],
                        [x, y, 0], [x, 0, z], [0, y, z], [x, y, z]])
        return ucv

test_anions = np.array([[[-8, 0],
                         [0, -8]],
                        [[0, -8],
                         [-8, 0]]])
test_cats = np.array([[[1, 1],
                       [1, 1]],
                      [[2, 0],
                       [0, 2]]])
key = {1:'Li', 2:'Zn', -8:'O'}


unit_cell_vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                               [1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]])
unit_cell_vertices *= test_anions.shape[0]

test = AtomStructure(test_anions, test_cats)

