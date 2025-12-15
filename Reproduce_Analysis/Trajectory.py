import omicverse as ov
import scanpy as sc
import scvelo as scv
import numpy as np
import cellrank as cr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

ov.utils.ov_plot_set()
adata = sc.read_h5ad('cancer.h5ad')
adata = adata[:, adata.var['highly_variable']]
adata.var_names_make_unique()

v0 = ov.single.pyVIA(adata=adata,adata_key='emb_pca',adata_ncomps=80, basis='spatial',
                         clusters='domain',knn=30,random_seed=4,root_user=[4823],)

v0.run()
plot_color = [
    "#001219",
    "#005F73",
    "#0a9396",
    "#94d2bd",
    "#e9d8a6",
    "#ee9b00",
    "#ca6702",
    "#ae2012",
    "#9b2226",
    "#Cbc106",
    "#27993C",
    "#1c6838",
    "#8ebcb5",
    "#389ca7",
    "#4d83aB",
    "#Cb7b26",
    "#Bf565D",
    "#9e163C"
]

custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", plot_color, N=256)
custom_cmap


from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("custom_cmap", plot_color, N=256)
fig,ax=v0.plot_stream(basis='spatial',clusters='domain',arrow_color='w',arrow_size=1.0, density_grid=1.0, scatter_size=30, scatter_alpha=1.0, linewidth=1.1,cmap_str=custom_cmap,title='')