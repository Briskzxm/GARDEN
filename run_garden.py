import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
sc.set_figure_params(scanpy=True, dpi=100, dpi_save=150, frameon=True, vector_friendly=True, fontsize=8)

import os
import numpy as np
import torch
import pandas as pd
from sklearn import metrics
import multiprocessing as mp
from GARDEN import GARDEN
import scanpy as sc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['R_HOME'] = "/home/zhangxinming/anaconda3/envs/Test/lib/R"

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--input', dest='input', type=str, default = None)
parser.add_argument('--n_clusters', dest='n_clusters', type=int, default = 6)
args, unknown = parser.parse_known_args()

if args.input is not None:
    dataset = str(args.input)
    adata = sc.read_h5ad(dataset)
else:
    adata = sc.datasets.visium_sge(sample_id="V1_Human_Brain_Section_2")

model = GARDEN.GARDEN(adata,model_select='Radius',k_cl=7,rad_cutoff=500,device=device)
adatas1 = model.train()

n_clusters = args.n_clusters 
radius = 30
tool = 'mclust' # mclust, leiden, and louvain

# clustering
from GARDEN.utils import clustering

if tool == 'mclust':
   clustering(adatas1, n_clusters, radius=radius, method=tool, refinement=True) # For DLPFC dataset, we use optional refinement step.
elif tool in ['leiden', 'louvain']:
   clustering(adatas1, n_clusters, radius=radius, method=tool, start=0.1, end=0.8, increment=0.01, refinement=False)

sc.pl.spatial(adatas1,basis='spatial',color = ['domain'],spot_size = 200)