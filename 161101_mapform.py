import numpy as np
from mayavi import mlab

class AtomStructure:
    def __init__(self, anions, cations):
        self.anions = anions
        self.cations = cations
    def get_uc_vertices(self):
        x, y, z = self.anions.shape
        ucv = np.array([[0, 0, 0], [x, 0, 0], [0, y, 0], [0, 0, z],
                        [x, y, 0], [x, 0, z], [0, y, z], [x, y, z]])
        return ucv
    def get_ucpaths(self):
        verts = self.get_uc_vertices()
        faces = np.vstack((verts[:2], verts[4], verts[2], verts[0], verts[3],
                           verts[5], verts[7], verts[6], verts[3]))
        strut1 = np.vstack((verts[1], verts[5]))
        strut2 = np.vstack((verts[2], verts[6]))
        strut3 = np.vstack((verts[4], verts[7]))
        return [np.array(np.transpose(arr), dtype='float64') for arr in 
                 [faces, strut1, strut2, strut3]]
    def get_anions_xyz(self):
        return test_anions.nonzero()
    def plot_cell(self):
        ucpaths = self.get_ucpaths()
        mlab.figure(size=(350, 350))
        ucp1 = mlab.plot3d(ucpaths[0][0], ucpaths[0][1], ucpaths[0][2], 
                           color=(0, 0, 0))
        ucp2 = mlab.plot3d(ucpaths[1][0], ucpaths[1][1], ucpaths[1][2], 
                           color=(0, 0, 0))
        ucp3 = mlab.plot3d(ucpaths[2][0], ucpaths[2][1], ucpaths[2][2], 
                           color=(0, 0, 0))
        ucp4 = mlab.plot3d(ucpaths[3][0], ucpaths[3][1], ucpaths[3][2], 
                           color=(0, 0, 0))
        mlab.show()

test_anions = np.array([[[-8, 0],
                         [0, -8]],
                        [[0, -8],
                         [-8, 0]]])
test_cats = np.array([[[1, 1],
                       [1, 1]],
                      [[2, 0],
                       [0, 2]]])
key = {1:'Li', 2:'Zn', -8:'O'}


test = AtomStructure(test_anions, test_cats)

