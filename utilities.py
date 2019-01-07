'''
a collection of simple utilities
mostly stolen from stack overflow
'''
import numpy as np

# https://stackoverflow.com/a/36960495
def onehot_initialization(mat):
    '''
    to get a 3d matrix of the world view
    '''
    ncols = mat.max()+1
    out = np.zeros(mat.shape + (ncols,), dtype=int)
    out[all_idx(mat, axis=2)] = 1
    return out

# https://stackoverflow.com/a/46103129/ @Divakar
def all_idx(idx, axis):
    '''
    to get the necessary indices
    '''
    grid = np.ogrid[tuple(map(slice, idx.shape))]
    grid.insert(axis, idx)
    return tuple(grid)
