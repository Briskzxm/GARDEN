import warnings
warnings.filterwarnings('ignore')
import numpy as np
import spateo as st
import scanpy as sc

adata = sc.read("40Slices.h5ad")
palette = [
    '#B10026', '#E31A1C', '#FC4E2A', '#FD8D3C', '#FEB24C',  
    '#FED976', '#FFFFB2', '#41AB5D', '#78C679', '#ADDD8E',  
    '#D9F0A3', '#006837', '#238443', '#4DAC26', '#7FBC41',  
    '#B2E2E2', '#66C2A4', '#2CA25F', '#006D2C', '#8C2D04',  
    '#CC4C02', '#E6550D', '#FD8D3C', '#FDAE6B', '#FDD0A2',  
    '#54278F', '#756BB1', '#9E9AC8', '#CBC9E2', '#DADAEB',  
    '#6A51A3', '#807DBA', '#9E9AC8', '#BFD3E6', '#F7FBFF',  
    '#4292C6', '#2171B5', '#084594', '#08306B', '#C6DBEF'   
]

embryo_pc, plot_cmap = st.tdr.construct_pc(adata=adata.copy(), spatial_key="spatial", groupby="annotation", key_added="tissue",colormap=palette)
st.pl.three_d_plot(model=embryo_pc, key="tissue", model_style="points",cpo = [80,200,-20], model_size=3 ,colormap=plot_cmap, jupyter="static",off_screen=True)
st.pl.three_d_plot(model=embryo_pc, key="tissue", model_style="points",cpo = 'yz', model_size=3 ,colormap=plot_cmap, jupyter="static",off_screen=True)
embryo_mesh, _, _ = st.tdr.construct_surface(pc=embryo_pc, key_added="tissue", alpha=0.3, cs_method="marching_cube", cs_args={"mc_scale_factor": 1}, smooth=10000, scale_factor=1.08)
st.pl.three_d_plot(model=st.tdr.collect_models([embryo_mesh, embryo_pc]), cpo = [-50,-150,-30],key="tissue", model_style=["surface", "points"], jupyter="static")

subtype = "LA"
subtype_rpc = st.tdr.three_d_pick(model=embryo_pc, key="tissue", picked_groups=subtype)[0]
subtype_mesh, subtype_pc, _ = st.tdr.construct_surface(
    pc=subtype_rpc, key_added="tissue", label=subtype, color="purple", alpha=0.6, cs_method="marching_cube", cs_args={"mc_scale_factor": 0.15}, smooth=10000, scale_factor=1)
st.pl.three_d_multi_plot(
    model=st.tdr.collect_models(
        [
            st.tdr.collect_models([embryo_mesh, subtype_pc]),
            st.tdr.collect_models([embryo_mesh, subtype_mesh]),
            st.tdr.collect_models([embryo_mesh, subtype_mesh, subtype_pc])
        ]
    ),
    key="tissue",
    model_style=[["surface", "points"], "surface", ["surface", "surface", "points"]],
    model_size=5,
    shape=(1, 3),
    jupyter="static"
)
