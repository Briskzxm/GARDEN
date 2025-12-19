import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import torch
sys.path.append('./')  # uncomment for local import
import tangram as tg

tg.__version__

adata = sc.read('MOB_3D.h5ad')
ad_sp = adata[adata.obs['section'].isin([3])]
ad_sc = sc.read('mouse_bulb_scRNA.h5ad')

sc.pp.normalize_total(ad_sc)
genes = set(ad_sc.var_names) & set(ad_sp.var_names)

sc.tl.rank_genes_groups(ad_sc,groupby='cellType')
cell_types = ad_sc.obs['cellType'].unique().tolist()

top_genes_dict = {}

for ct in cell_types:
    df = sc.get.rank_genes_groups_df(ad_sc, group=ct)
    df_top = df.sort_values('scores', ascending=False).head(50)
    top_genes_dict[ct] = df_top['names'].tolist()

all_top_genes = [gene for gene_list in top_genes_dict.values() for gene in gene_list]
markers = list(set(all_top_genes))
len(markers)

tg.pp_adatas(ad_sc, ad_sp, genes=markers)

ad_map = tg.map_cells_to_space(
    adata_sc=ad_sc,
    adata_sp=ad_sp,
    device='cuda',
)

tg.project_cell_annotations(ad_map, ad_sp, annotation='cellType.sub')
annotation_list = list(pd.unique(ad_sc.obs['cellType.sub']))
tg.plot_cell_annotation_sc(ad_sp, annotation_list,x='x', y='y',spot_size= 20, scale_factor=2,perc=0.02)