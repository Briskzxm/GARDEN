import scanpy as sc
import squidpy as sq
import numpy as np

adata = sc.read_h5ad('cancer.h5ad')
adata = adata[:,adata.var['highly_variable']]
adata.X = adata.obsm['emb']
adata.obs['domain'] = adata.obs['domain'].astype(str)
adata.obs['domain'] = adata.obs['domain'].astype('category')

sq.gr.ligrec(
    adata,
    use_raw=False,
    cluster_key="domain"
)


sq.pl.ligrec(
    adata,
    cluster_key="domain",
    source_groups=['11'],
    target_groups=['11',"14"],
    means_range = [0.99,1.0],
    swap_axes=True,
)