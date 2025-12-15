import scanpy as sc
from GraphST import GraphST
from GraphST.preprocess import filter_with_overlap_gene
from GraphST.utils import project_cell_to_spot
import pandas as pd
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

file_fold ="Breast_Caner"
adata = sc.read_visium(file_fold, count_file='CytAssist_FFPE_Human_Breast_Cancer_filtered_feature_bc_matrix.h5', load_images=True)
adata.var_names_make_unique()

file_path = 'scRNA.h5ad'
adata_sc = sc.read(file_path)
adata_sc.var_names_make_unique()

GraphST.preprocess(adata)
GraphST.construct_interaction(adata)
GraphST.add_contrastive_label(adata)

GraphST.preprocess(adata_sc)
adata, adata_sc = filter_with_overlap_gene(adata, adata_sc)

GraphST.get_feature(adata)

model = GraphST.GraphST(adata, adata_sc, epochs=1200, random_seed=50, device=device, deconvolution=True)
adata, adata_sc = model.train_map()

adata_sc.obs['cell_type'] = adata_sc.obs.Annotation
project_cell_to_spot(adata, adata_sc, retain_percent=0.15)

import matplotlib as mpl
import matplotlib.pyplot as plt
with mpl.rc_context({'axes.facecolor':  'black',
                     'figure.figsize': [4.5, 5]}):
            sc.pl.embedding(adata, cmap='magma',basis='spatial',
                  # selected cell types
                  color=['DCIS 1', 'DCIS 2', 'Invasive Tumor','B Cells'],
                  ncols=5, size=50,
                  #img_key='hires',
                  # limit color scale at 99.2% quantile of cell abundance
                  vmin=0, vmax='p99.2',
                  show=True
                 )