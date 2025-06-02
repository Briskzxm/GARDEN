from .model import *
from .utils import *
from argparse import ArgumentParser
from .preprocess import *
from .align import *
import scanpy as sc
import pandas as pd
from .GARDEN_Align import GARDEN_Align
from .utils import  spatial_match

def train_multi_slices(slices_init,args,fraction_ratio=1,double_r=False,step=1):
    slices = slices_init.copy()
    assert len(slices)>=2
    rotation1_list = []
    rotation2_list = []
    transform1_list = []
    transform2_list = []
    matchinglist1 = []
    matchinglist2 = []
    for i in range(len(slices)-step):
        print(f'------ Train slice {i} and slice {i+1} ------')
        slice1 = slices[i+step].copy()
        slice2 = slices[i].copy()
        slice1.obs_names_make_unique()
        slice2.obs_names_make_unique()
        sc.pp.subsample(slice1,fraction = fraction_ratio)
        sc.pp.subsample(slice2,fraction = fraction_ratio)
        model = GARDEN_Align(slice1,slice2,args)
        out1,out2,similarity = model.train()
        slice1,R1,T1,matching1 = align_slice(slice1,slice2,pis = similarity, renamed_spatial='spatial', slice_only = False)
        best, index, distance = spatial_match(embds = [out1,out2], adatas=[slice1,slice2], scale_coord=False, smooth=False, reorder=False,top_n=30,smooth_range=30)
        matching2 = np.array([range(index.shape[0]), best])
        slice1,R2,T2,matching2  = align_slice(slice1,slice2,index = matching2 ,renamed_spatial='spatial', slice_only = False)

        matching3 = get_closest_half_matches_median(slice1,slice2)
        slice1,R3,T3,matching3  = align_slice(slice1,slice2,index = matching3 ,renamed_spatial='spatial', slice_only = False)

        slices[i+1].obsm['spatial'] = np.dot(slices[i+1].obsm['spatial'], R1.T) + T1

        if(double_r):
            slices[i+1].obsm['spatial'] = np.dot(slices[i+1].obsm['spatial'], R2.T) + T2
            slices[i+1].obsm['spatial'] = np.dot(slices[i+1].obsm['spatial'], R2.T) + T2
        else: 
            slices[i+1].obsm['spatial'] = np.dot(slices[i+1].obsm['spatial'], R2.T) + T2
        slices[i+1].obsm['spatial'] = np.dot(slices[i+1].obsm['spatial'], R3.T) + T3

        rotation1_list.append(R1)
        rotation2_list.append(R2)
        transform1_list.append(T1)
        transform2_list.append(T2)
        matchinglist1.append(matching1)
        matchinglist2.append(matching2)
    return slices, rotation1_list, rotation2_list, transform1_list, transform2_list, matchinglist1, matchinglist2


def train_multi_slices_v2(slices_init,args,fraction_ratio=1,double_r=False):
    slices = slices_init.copy()
    assert len(slices)>=2
    rotation1_list = []
    rotation2_list = []
    transform1_list = []
    transform2_list = []
    matchinglist1 = []
    matchinglist2 = []
    for i in range(len(slices)-1):
        print(f'------ Train slice {i} and slice {i+1} ------')
        slice1 = slices[i+1].copy()
        slice2 = slices[i].copy()
        slice1.obs_names_make_unique()
        slice2.obs_names_make_unique()
        sc.pp.subsample(slice2,fraction = fraction_ratio)
        sc.pp.subsample(slice1,fraction = fraction_ratio)
        model = GARDEN_Align(slice1,slice2,args)
        out1, out2,similarity = model.train()
        slice1,R1,T1,matching1 = align_slice(slice1,slice2,pis = similarity, renamed_spatial='spatial', slice_only = False)
        matching2 = get_closest_half_matches(slice1,slice2,keep_ratio=0.5)
        slice1,R2,T2,matching2  = align_slice(slice1,slice2,index = matching2 ,renamed_spatial='spatial', slice_only = False)
        slices[i+1].obsm['spatial'] = np.dot(slices[i+1].obsm['spatial'], R1.T) + T1

        if(double_r):
            slices[i+1].obsm['spatial'] = np.dot(slices[i+1].obsm['spatial'], R2.T) + T2
            slices[i+1].obsm['spatial'] = np.dot(slices[i+1].obsm['spatial'], R2.T) + T2
        else: 
            slices[i+1].obsm['spatial'] = np.dot(slices[i+1].obsm['spatial'], R2.T) + T2

        rotation1_list.append(R1)
        rotation2_list.append(R2)
        transform1_list.append(T1)
        transform2_list.append(T2)
        matchinglist1.append(matching1)
        matchinglist2.append(matching2)
    return slices, rotation1_list, rotation2_list, transform1_list, transform2_list, matchinglist1, matchinglist2