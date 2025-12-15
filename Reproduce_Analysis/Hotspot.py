import scanpy as sc
import stereo as st
data = st.io.read_h5ad("cancer.h5ad")
data.adata.obs_names_make_unique()
data.adata.var_names_make_unique()
data.tl.spatial_hotspot(
                    use_highly_genes=True,
                    use_raw=False,
                    hvg_res_key='highly_variable_genes',
                    model='normal',
                    n_neighbors=30,
                    n_jobs=20,
                    fdr_threshold=0.05,
                    min_gene_threshold=10,
                    res_key='spatial_hotspot',
                    )
data.plt.hotspot_local_correlations(width=3000,height=3000)
data.plt.hotspot_modules()