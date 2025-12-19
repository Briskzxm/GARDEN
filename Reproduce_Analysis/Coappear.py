import scanpy as sc
import squidpy as sq
adata = sc.read_h5ad("cancer.h5ad")
adata = adata[:,adata.var['highly_variable']]
adata.X = adata.obsm['feat']

import squidpy as sq
sq.gr.co_occurrence(adata, cluster_key="domain")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Define your custom colormap using hex color codes
colors = ['#75d9e8', '#71b499', '#15a1c0', '#3563fa','#fabbef',
              '#f49989', '#a0ccca', '#b88b97', '#094a5f',
              '#cdc0d5', '#ffb301','#f2656a',  '#5665b5', '#dc1400']

# Create a colormap object
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)

sq.pl.co_occurrence(
    adata,
    cluster_key="domain",
    clusters=[12],
    figsize=(5, 4)
)