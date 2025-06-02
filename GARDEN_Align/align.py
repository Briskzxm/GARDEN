from .utils import *
import numpy as np

def align_slice(slice1,slice2,index=None,pis=None,renamed_spatial = 'spatial', slice_only = True):

    assert index is not None or pis is not None, "index and pis both None"

    if pis is not None:
        best =  np.array(np.argmax(pis, axis=1))
        index = np.array([range(slice2.shape[0]), best])
        R,T = find_rigid_transform(slice1.obsm['spatial'][index[1,:]],slice2.obsm['spatial'])
    else:
        R,T = find_rigid_transform(slice1.obsm['spatial'][index[1,:]],slice2.obsm['spatial'][index[0,:]])
    
    slice1.obsm[renamed_spatial] = np.dot(slice1.obsm['spatial'], R.T) + T
    if not slice_only:
        return slice1,R,T,index
    else:
        return slice1

# def align_slice(slice1,slice2,index=None,pis=None,renamed_spatial = 'spatial', step1 = True):

#     assert index is not None or pis is not None, "index and pis both None"

#     if pis is not None:
#         best =  np.array(np.argmax(pis, axis=1))
#         index = np.array([range(slice2.shape[0]), best])
#         
#     R,T = find_rigid_transform(slice1.obsm['spatial'][index[1,:]],slice2.obsm['spatial'])
#    
    
#     slice1.obsm[renamed_spatial] = np.dot(slice1.obsm['spatial'], R.T) + T

#     return slice1,R,T,index