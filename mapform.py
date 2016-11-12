#Version 0.2.0 alpha---renamed AtomStructure to FccStructure;
#added some more user scalability options and a scale attribute;
#also added a cation_holes attribute to affect other methods

import numpy as np
from mayavi import mlab
import itertools as it

class FccStructure:
    def __init__(self, anions, cations, key={2:[(0.3, 0.3, 0.3), 0.5],
                                             1:[(0.3, 0.6, 0.3), 0.5],
                                             -8:[(1, 0, 0), 0.5]},
                 costs=None, scale=1, cation_holes='tetrahedral'):
        self.anions = anions
        self.cations = cations
        self.key = key
        self.costs = costs
        self.scale = scale
        self.cation_holes = cation_holes
    def get_uc_vertices(self):
        """Return array with rows corresponding to unit cell vertices"""
        x, y, z = self.anions.shape
        ucv = np.array([[0, 0, 0], [x, 0, 0], [0, y, 0], [0, 0, z],
                        [x, y, 0], [x, 0, z], [0, y, z], [x, y, z]])
        ucv *= self.scale
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
    def repeat_anions(self, anions):#this method is now deprecated
        """Returns expanded anion array (atoms at zero also at one)"""
        exp_anions = np.zeros([i + 1 for i in anions.shape], 
                              dtype='int64')
        exp_anions[anions.nonzero()] = anions[anions.nonzero()]
        exp_anions[-1, :, :] = exp_anions[0, :, :]
        exp_anions[:, -1, :] = exp_anions[:, 0, :]
        exp_anions[:, :, -1] = exp_anions[:, :, 0]
        return exp_anions
    def repeat_array(self, ion_arr):
        """Returns expanded array of atoms (those at zero also at one)"""
        a, b, c = ion_arr.shape
        exp_arr = np.zeros((a + 1, b + 1, c + 1), dtype='int64')
        exp_arr[:a, :b, :c] = ion_arr
        exp_arr[-1, :, :] = exp_arr[0, :, :]
        exp_arr[:, -1, :] = exp_arr[:, 0, :]
        exp_arr[:, :, -1] = exp_arr[:, :, 0]
        return exp_arr
    def plot_anions(self):
        """Plots anion array"""
        x, y, z = [np.array(arr, dtype='float64') * self.scale for arr in 
                   self.repeat_array(self.anions).nonzero()]
        colour = self.key[self.anions.min()][0]
        atom_scale = self.key[self.anions.min()][1]
        anions = mlab.points3d(x, y, z, color=colour, resolution=32,
                              scale_factor=atom_scale)
    def plot_cations(self):
        """Plots cations"""
        for i in self.key:
            if self.cation_holes == 'tetrahedral':
                x, y, z = [(arr + 0.5) * self.scale for arr in 
                           np.where(self.cations == i)]
            elif self.cation_holes == 'octahedral':
                exp_cats = self.repeat_array(self.cations)
                xyz_ans = np.transpose(np.where(self.repeat_array(\
                                        self.anions) == 0))
                xyz = []
                for xyz_a in xyz_ans:
                    if exp_cats[tuple(xyz_a)] == i:
                        xyz.append(xyz_a)
                if xyz:
                    x, y, z = tuple(np.transpose(xyz) * self.scale)
                else:
                    continue
            mlab.points3d(x, y, z, color=self.key[i][0], resolution=32,
                          scale_factor=self.key[i][1])
    def plot_bonds(self, bond_radius=None):
        """Plots bonds between anions and cations"""
        if self.cation_holes == 'tetrahedral':
            cat_is = list(it.product([0, -1], repeat=3))
        elif self.cation_holes == 'octahedral':
            cat_is = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                      (0, 0, 1), (0, 0, -1)]
        for xyz in np.transpose(self.repeat_array(self.anions).nonzero()):
            cats = np.array([arr + xyz for arr in cat_is])
            if self.cation_holes == 'tetrahedral':
                cat_bools = [np.logical_and(cats[:, i] >= 0, \
                             cats[:, i] < self.cations.shape[i])
                             for i in range(3)]
            elif self.cation_holes == 'octahedral':
                cat_bools = [np.logical_and(cats[:, i] >= 0, \
                             cats[:, i] <= self.cations.shape[i])
                             for i in range(3)]
            cats = cats[np.all(cat_bools, axis=0)]
            for c in cats:
                if self.cation_holes == 'tetrahedral':
                    if self.cations[tuple(c)]:
                        x, y, z = tuple(np.transpose(np.vstack((xyz, 
                                        c + 0.5))) * self.scale)
                elif self.cation_holes == 'octahedral':
                    if self.repeat_array(self.cations)[tuple(c)]:
                        x, y, z = tuple(np.array(np.transpose(\
                                    np.vstack((xyz, c))), dtype='float64')\
                                       * self.scale)
                mlab.plot3d(x, y, z, color=(0.7, 0.7, 0.7), 
                            tube_radius=bond_radius)
    def plot_anion_costs(self, colour_min=(0, 0, 1), colour_max=(1, 0, 0),
                         colour_zero=(0, 0, 0)):
        """Plots anion costs on anion positions by size and colours"""
        xyz = self.repeat_array(self.anions).nonzero()
        costs = self.repeat_array(self.costs)[xyz]
        costs_s = (1.1 * np.abs(costs)) / np.abs(costs).max() + 0.1
        xyz = [arr * self.scale for arr in xyz]
        for i, c in enumerate(costs): 
            if c < 0:
                mlab.points3d(xyz[0][i], xyz[1][i], xyz[2][i], 
                              resolution=32, color=colour_min,
                              scale_factor=costs_s[i])
            elif c > 0:
                mlab.points3d(xyz[0][i], xyz[1][i], xyz[2][i], 
                              resolution=32, color=colour_max,
                              scale_factor=costs_s[i])
            else:
                mlab.points3d(xyz[0][i], xyz[1][i], xyz[2][i], 
                              resolution=32, color=colour_zero,
                              scale_factor=costs_s[i])
    def plot_anion_costs_cmap(self, cm='seismic'):
        """Plots anion costs on anion positions by size and colourmap"""
        xyz = self.repeat_array(self.anions).nonzero()
        costs = self.repeat_array(self.costs)[xyz]
        costs_max = np.abs(costs).max()
        costs_s = (1.1 * np.abs(costs)) / np.abs(costs).max() + 0.1
        xyz = [arr * self.scale for arr in xyz]
        #The following is a fudge to ensure that the colour is 
        #independent from the size of the spheres
        pts = mlab.quiver3d(xyz[0], xyz[1], xyz[2], costs_s, costs_s,
                            costs_s, scalars=costs, mode='sphere', 
                            colormap='RdBu', resolution=32, vmin=-costs_max,
                            vmax=costs_max)
        pts.glyph.color_mode = 'color_by_scalar'
        pts.glyph.glyph_source.glyph_source.center = [0, 0, 0]
    def plot_nearest_cations(self, anion_indices, bond_radius=None):
        """Given anion indices plots anion with surrounding cations"""
        if self.anions[anion_indices]:
            ais = [ai * self.scale for ai in anion_indices]
            mlab.points3d(ais[0], ais[1], ais[2], 
                          color=self.key[self.anions.min()][0], 
                          resolution=32, 
                          scale_factor=self.key[self.anions.min()][1])
        a, b, c = self.anions.shape
        if self.cation_holes == 'tetrahedral':
            cat_pos = np.array(list(it.product([0, -1], repeat=3)))
        elif self.cation_holes == 'octahedral':
            cat_pos = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], 
                                [0, -1, 0], [0, 0, 1], [0, 0, -1]])
        cat_pos += np.array(anion_indices)
        cat_is = np.column_stack((np.mod(cat_pos[:, 0], a),
                                  np.mod(cat_pos[:, 1], b),
                                  np.mod(cat_pos[:, 2], c))) 
        #return cat_pos, cat_is
        for i in self.key:
            cats = np.array([arr for i1, arr in enumerate(cat_pos) if 
                             list(cat_is[i1]) in 
                             [list(arr2) for arr2 in 
                              np.transpose(np.where(self.cations == i))]],
                            dtype='float64')
            if self.cation_holes == 'tetrahedral':
                cats += 0.5
            if cats.shape[0]:
                x, y, z = np.transpose(cats) * self.scale
                mlab.points3d(x, y, z, color=self.key[i][0], resolution=32,
                              scale_factor=self.key[i][1])
                if self.anions[anion_indices]:
                    for cat in cats:
                        x, y, z = tuple(np.transpose(np.vstack((cat,
                                        anion_indices))) * self.scale)
                        mlab.plot3d(x, y, z, color=(0.7, 0.7, 0.7),
                                    tube_radius=bond_radius)

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
oct_cats = np.where(test_anions == 0, 2, 0)                        

#key takes the form of cation value, with list of a tuple to determine
#colour (normalized rgb values) and then a scalar for the atom size wanted)
key={2:[(0.3, 0.3, 0.3), 0.5], 1:[(0.3, 0.6, 0.3), 0.5],
     -8:[(1, 0, 0), 0.5]}
test = FccStructure(test_anions, test_cats, costs=costs, key=key)
test_oct = FccStructure(test_anions, oct_cats, costs=costs, key=key,
                        cation_holes='octahedral')

mlab.figure()
test.plot_cell()
test.plot_anions()
test.plot_cations()
test.plot_bonds()
mlab.orientation_axes()                    
mlab.show()

mlab.figure()
test.plot_cell()
test.plot_anion_costs()
test.plot_cations()
test.plot_bonds()
mlab.orientation_axes()                    
mlab.show()

mlab.figure()
test.plot_cell()
test.plot_nearest_cations((0, 0, 0))
mlab.orientation_axes()                    
mlab.show()

mlab.figure()
test_oct.plot_cell()
test_oct.plot_anions()
test_oct.plot_cations()
test_oct.plot_bonds()
mlab.orientation_axes()                    
mlab.show()

mlab.figure()
test_oct.plot_cell()
test_oct.plot_nearest_cations((0, 0, 0))
mlab.orientation_axes()                    
mlab.show()
"""
