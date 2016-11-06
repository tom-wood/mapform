#Version 0.1.1---includes local cation environment plotting

import numpy as np
from mayavi import mlab
import itertools as it

class AtomStructure:
    def __init__(self, anions, cations, key={2:[(0.3, 0.3, 0.3), 0.5],
                                             1:[(0.3, 0.6, 0.3), 0.5],
                                             -8:[(1, 0, 0), 0.5]},
                 costs=None):
        self.anions = anions
        self.cations = cations
        self.key = key
        self.costs = costs
    def get_uc_vertices(self):
        """Return array with rows corresponding to unit cell vertices"""
        x, y, z = self.anions.shape
        ucv = np.array([[0, 0, 0], [x, 0, 0], [0, y, 0], [0, 0, z],
                        [x, y, 0], [x, 0, z], [0, y, z], [x, y, z]])
        return ucv
    def get_ucpaths(self):
        """Return series of paths which combined plot out a unit cell"""
        verts = self.get_uc_vertices()
        faces = np.vstack((verts[:2], verts[4], verts[2], verts[0], verts[3],
                           verts[5], verts[7], verts[6], verts[3]))
        strut1 = np.vstack((verts[1], verts[5]))
        strut2 = np.vstack((verts[2], verts[6]))
        strut3 = np.vstack((verts[4], verts[7]))
        return [np.array(np.transpose(arr), dtype='float64') for arr in 
                 [faces, strut1, strut2, strut3]]
    def plot_cell(self):
        """Plots the unit cell"""
        ucpaths = self.get_ucpaths()
        ucp1 = mlab.plot3d(ucpaths[0][0], ucpaths[0][1], ucpaths[0][2], 
                           color=(0, 0, 0))
        ucp2 = mlab.plot3d(ucpaths[1][0], ucpaths[1][1], ucpaths[1][2], 
                           color=(0, 0, 0))
        ucp3 = mlab.plot3d(ucpaths[2][0], ucpaths[2][1], ucpaths[2][2], 
                           color=(0, 0, 0))
        ucp4 = mlab.plot3d(ucpaths[3][0], ucpaths[3][1], ucpaths[3][2], 
                           color=(0, 0, 0))
        return
    def repeat_anions(self, anions):
        """Returns expanded anion array (atoms at zero also at one)"""
        exp_anions = np.zeros([i + 1 for i in anions.shape], 
                              dtype='int64')
        exp_anions[anions.nonzero()] = anions[anions.nonzero()]
        exp_anions[-1, :, :] = exp_anions[0, :, :]
        exp_anions[:, -1, :] = exp_anions[:, 0, :]
        exp_anions[:, :, -1] = exp_anions[:, :, 0]
        return exp_anions
    def plot_anions(self):
        """Plots anion array"""
        x, y, z = [np.array(arr, dtype='float64') for arr in 
                   self.repeat_anions(self.anions).nonzero()]
        anions = mlab.points3d(x, y, z, color=(1, 0, 0), resolution=32)
    def plot_cations(self):
        """Plots cations"""
        for i in self.key:
            x, y, z = [arr + 0.5 for arr in np.where(self.cations == i)]
            mlab.points3d(x, y, z, color=self.key[i][0], resolution=32,
                          scale_factor=self.key[i][1])
    def plot_bonds(self):
        """Plots bonds between anions and cations"""
        cat_is = list(it.product([0, -1], repeat=3))
        for xyz in np.transpose(self.repeat_anions(self.anions).nonzero()):
            cats = np.array([arr + xyz for arr in cat_is])
            cat_bools = [np.logical_and(cats[:, i] >= 0, 
                                        cats[:, i] < self.cations.shape[i])
                         for i in range(3)]
            cats = cats[np.all(cat_bools, axis=0)]
            for c in cats:
                if self.cations[tuple(c)]:
                    x, y, z = tuple(np.transpose(np.vstack((xyz, c + 0.5))))
                    mlab.plot3d(x, y, z, color=(0.7, 0.7, 0.7))
    def plot_anion_costs(self):
        """Plots anion costs on anion positions by size and colormap"""
        xyz = self.repeat_anions(self.anions).nonzero()
        costs = self.repeat_anions(self.costs)[xyz]
        costs_s = np.abs(costs)**(1./3) + 1
        costs_max = np.abs(costs).max()
        #The following is a fudge to ensure that the colour is 
        #independent from the size of the spheres
        pts = mlab.quiver3d(xyz[0], xyz[1], xyz[2], costs_s, costs_s,
                            costs_s, scalars=costs, mode='sphere', 
                            colormap='RdBu', resolution=32, vmin=-costs_max,
                            vmax=costs_max)
        pts.glyph.color_mode = 'color_by_scalar'
        pts.glyph.glyph_source.glyph_source.center = [0, 0, 0]
    def plot_nearest_cations(self, anion_indices):
        """Given anion indices plots anion with surrounding cations"""
        if self.anions[anion_indices]:
            mlab.points3d(anion_indices[0], anion_indices[1],
                          anion_indices[2], color=(1, 0, 0), resolution=32,
                          scale_factor=0.5)
        a, b, c = self.anions.shape
        cat_pos = np.array(list(it.product([0, -1], repeat=3)))
        cat_pos += np.array(anion_indices)
        cat_is = np.column_stack((np.mod(cat_pos[:, 0], a),
                                  np.mod(cat_pos[:, 1], b),
                                  np.mod(cat_pos[:, 2], c))) 
        #return cat_pos, cat_is
        for i in self.key:
            cats = np.array([arr for i1, arr in enumerate(cat_pos) if 
                             list(cat_is[i1]) in 
                             [list(arr2) for arr2 in 
                              np.transpose(np.where(self.cations == i))]]) \
                    + 0.5
            if cats.shape[0]:
                x, y, z = np.transpose(cats)
                mlab.points3d(x, y, z, color=self.key[i][0], resolution=32,
                              scale_factor=self.key[i][1])
                if self.anions[anion_indices]:
                    for cat in cats:
                        x, y, z = tuple(np.transpose(np.vstack((cat,
                                        anion_indices))))
                        mlab.plot3d(x, y, z, color=(0.7, 0.7, 0.7))

"""
#Example follows:
test_anions = np.array([[[-8, 0],
                         [0, -8]],
                        [[0, -8],
                         [-8, 0]]])
test_cats = np.array([[[1, 1],
                       [1, 1]],
                      [[2, 0],
                       [0, 2]]])
costs = np.arange(-4, 4).reshape((2, 2, 2))

#key takes the form of cation value, with list of a tuple to determine
#colour (normalized rgb values) and then a scalar for the atom size wanted)
key={2:[(0.3, 0.3, 0.3), 0.5], 1:[(0.3, 0.6, 0.3), 0.5]}
test = AtomStructure(test_anions, test_cats, costs=costs, key=key)

mlab.figure()
test.plot_cell()
test.plot_anions()
test.plot_cations()
test.plot_bonds()
mlab.show()

mlab.figure()
test.plot_cell()
test.plot_anion_costs()
test.plot_cations()
test.plot_bonds()
mlab.show()


mlab.figure()
test.plot_cell()
test.plot_nearest_cations((0, 0, 0))
mlab.show()
"""
