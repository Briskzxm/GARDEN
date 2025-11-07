import torch 
import numpy as np
import ruptures as rpt
import scipy
import anndata
import sklearn
import random
import numpy as np
import scanpy as sc
import pandas as pd
from typing import List, Optional, Union
import scipy.sparse as sp
from torch.backends import cudnn
from torch_geometric.data import Data
from torch import Tensor
from scipy.sparse import coo_matrix,issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.extmath import randomized_svd
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected
import ot
from scipy.spatial import cKDTree

from sklearn.neighbors import NearestNeighbors 

def construct_interaction_KNN(adata, n_neighbors=3):
    position = adata.obsm['spatial']
    n_spot = position.shape[0]
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(position)  
    _ , indices = nbrs.kneighbors(position)
    x = indices[:, 0].repeat(n_neighbors)
    y = indices[:, 1:].flatten()
    interaction = np.zeros([n_spot, n_spot])
    interaction[x, y] = 1
    
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)
    np.fill_diagonal(adj, 1)
    return adj

def seed_everything(random_seed=2025):
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    torch.cuda.empty_cache()

def Cal_Spatial_Net(
    adata,
    rad_cutoff: Optional[Union[None, int]] = None,
    k_cutoff: Optional[Union[None, int]] = None,
    model: Optional[str] = "Radius",
    return_data: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> None:
    r"""
    Construct the spatial neighbor networks.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff.
        When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """
    assert model in ["Radius", "KNN"]
    if verbose:
        print("Calculating spatial neighbor graph ...")

    if model == "KNN":
        edge_index = knn_graph(
            x=torch.tensor(adata.obsm["spatial"]),
            flow="target_to_source",
            k=k_cutoff,
            loop=True,
            num_workers=8,
        )
        edge_index = to_undirected(
            edge_index, num_nodes=adata.shape[0]
        )  # ensure the graph is undirected
    elif model == "Radius":
        edge_index = radius_graph(
            x=torch.tensor(adata.obsm["spatial"]),
            flow="target_to_source",
            r=rad_cutoff,
            loop=True,
            num_workers=8,
        )

    graph_df = pd.DataFrame(edge_index.numpy().T, columns=["Cell1", "Cell2"])
    id_cell_trans = dict(zip(range(adata.n_obs), adata.obs_names))
    graph_df["Cell1"] = graph_df["Cell1"].map(id_cell_trans)
    graph_df["Cell2"] = graph_df["Cell2"].map(id_cell_trans)
    adata.uns["Spatial_Net"] = graph_df

    if verbose:
        print(f"The graph contains {graph_df.shape[0]} edges, {adata.n_obs} cells.")
        print(f"{graph_df.shape[0]/adata.n_obs} neighbors per cell on average.")

    if return_data:
        return adata
    
def Cal_Feature_Net(adata, k_cutoff=20, verbose=True):
    
    feat = pd.DataFrame(adata.obsm['feat'])  # 取出特征矩阵
    feat.index = adata.obs.index  # 确保索引匹配
    feat.columns = [f'feat_{i}' for i in range(feat.shape[1])]  # 生成列名
    
    # 计算 KNN
    nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1, metric='correlation').fit(feat)
    distances, indices = nbrs.kneighbors(feat)

    # 构造邻接表
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1], indices[it, :], distances[it, :])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    
    # 去掉自身连接（对角线）
    Feature_Net = KNN_df.copy()
    Feature_Net = Feature_Net.loc[Feature_Net['Distance'] > 0, ]

    # 映射索引到真实的细胞 ID
    id_cell_trans = dict(zip(range(feat.shape[0]), np.array(feat.index)))
    Feature_Net['Cell1'] = Feature_Net['Cell1'].map(id_cell_trans)
    Feature_Net['Cell2'] = Feature_Net['Cell2'].map(id_cell_trans)

    # 存入 adata
    adata.uns['Spatial_Net'] = Feature_Net
    return Feature_Net  

def dual_SVD(
    X: np.ndarray,
    Y: np.ndarray,
    dim: Optional[int] = 50,
    singular: Optional[bool] = False,
    backend: Optional[str] = "sklearn",
    use_gpu: Optional[bool] = True,
    device: Optional[str] = "cpu" 
) -> List[Tensor]:
    r"""
    Dual PCA for batch correction

    Parameters
    ----------
    X
        expr matrix 1 in shape of (cells, genes)
    Y
        expr matrix 2 in shape of (cells, genes)
    dim
        dimension of embedding
    singular
        if multiply the singular value
    backend
        backend to calculate singular value
    use_gpu
        if calculate in gpu

    Returns
    ----------
    embd1, embd2: Tensors of embedding

    References
    ----------
    Thanks Xin-Ming Tu for his [blog](https://xinmingtu.cn/blog/2022/CCA_dual_PCA/)
    """
    assert X.shape[1] == Y.shape[1]
    device = device
    X = torch.Tensor(X).to(device=device)
    Y = torch.Tensor(Y).to(device=device)
    cor_var = X @ Y.T
    if backend == "torch":
        U, S, Vh = torch.linalg.svd(cor_var)
        if not singular:
            return U[:, :dim], Vh.T[:, :dim]
        Z_x = U[:, :dim] @ torch.sqrt(torch.diag(S[:dim]))
        Z_y = Vh.T[:, :dim] @ torch.sqrt(torch.diag(S[:dim]))
        return Z_x.cpu(), Z_y.cpu()
        # torch.dist(cor_var, Z_x @ Z_y.T)  # check the information loss
    elif backend == "sklearn":
        cor_var = cor_var.cpu().numpy()
        U, S, Vh = randomized_svd(cor_var, n_components=dim, random_state=0)
        if not singular:
            return Tensor(U), Tensor(Vh.T)
        Z_x = U @ np.sqrt(np.diag(S))
        Z_y = Vh.T @ np.sqrt(np.diag(S))
        return Tensor(Z_x), Tensor(Z_y)
    

def SVD_based_preprocess(adatas,
    dim: Optional[int] = 50,
    self_loop: Optional[bool] = False,
    SVD = True,
    join: Optional[str] = "inner",
    backend: Optional[str] = "sklearn",
    mincells_ratio : Optional[float] = 0.01,
    use_highly_variable: Optional[bool]=True,
    singular: Optional[bool] = True,
    check_order: Optional[bool] = True,
    n_top_genes: Optional[int] = 2500,
    device: Optional[str] = "cpu"):

    gpu_flag = True if torch.cuda.is_available() else False

    edgeLists = []
    for adata in adatas:
        G_df = adata.uns["Spatial_Net"].copy()
        cells = np.array(adata.obs_names)
        cells_id_tran = dict(zip(cells, range(cells.shape[0])))
        G_df["Cell1"] = G_df["Cell1"].map(cells_id_tran)
        G_df["Cell2"] = G_df["Cell2"].map(cells_id_tran)

        # build adjacent matrix
        G = scipy.sparse.coo_matrix(
            (np.ones(G_df.shape[0]), (G_df["Cell1"], G_df["Cell2"])),
            shape=(adata.n_obs, adata.n_obs),
        )
        if self_loop:
            G = G + scipy.sparse.eye(G.shape[0])
        edgeList = np.nonzero(G)
        edgeLists.append(edgeList)

    adata_all = adatas[0].concatenate(adatas[1], join=join)

    if use_highly_variable is True:
        #print("Select Highly Variable Peaks")
        min_cells = int(adata_all.shape[0] * mincells_ratio)
        # in-place filtering of regions
        sc.pp.filter_genes(adata_all, min_cells=min_cells)
        print('------ The shape of feature is ' + str(adata_all.shape[1])+' ------')
    # sc.pp.normalize_total(adata_all)
    # sc.pp.log1p(adata_all)
    adata_1 = adata_all[adata_all.obs['batch'] == '0']
    adata_2 = adata_all[adata_all.obs['batch'] == '1']
    # sc.pp.scale(adata_1)
    # sc.pp.scale(adata_2)
    # if gpu_flag:
    #     print(
    #         "Warning! Dual PCA is using GPU, which may lead to OUT OF GPU MEMORY in big dataset!"
    #     )
    if issparse(adata_1.X):
        adata_1.X = adata_1.X.todense()
    if issparse(adata_2.X):
        adata_2.X = adata_2.X.todense()
    if SVD:
        Z_x, Z_y = dual_SVD(
            adata_1.X.toarray(), adata_2.X.toarray(), dim=dim, singular=singular, backend=backend, device = device
        )
    data_x = Data(
        edge_index=torch.LongTensor(np.array([edgeLists[0][0], edgeLists[0][1]])), x=Z_x
    )
    data_y = Data(
        edge_index=torch.LongTensor(np.array([edgeLists[1][0], edgeLists[1][1]])), x=Z_y
    )
    datas = [data_x, data_y]    

    edges = [dataset.edge_index for dataset in datas]
    features = [dataset.x for dataset in datas]

    return edges, features


def lsi(
        X, n_components: int = 20,
        use_highly_variable = False, **kwargs
) -> None:
    r"""
    LSI analysis (following the Seurat v3 approach)

    Parameters
    ----------
    adata
        Input dataset
    n_components
        Number of dimensions to use
    use_highly_variable
        Whether to use highly variable features only, stored in
        ``adata.var['highly_variable']``. By default uses them if they
        have been determined beforehand.
    **kwargs
        Additional keyword arguments are passed to
        :func:`sklearn.utils.extmath.randomized_svd`
    """
    def tfidf(X):
        idf = X.shape[0] / X.sum(axis=0)
        if scipy.sparse.issparse(X):
            tf = X.multiply(1 / X.sum(axis=1))
            return tf.multiply(idf)
        else:
            tf = X / X.sum(axis=1, keepdims=True)
            return tf * idf
        
    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    X = tfidf(X)
    from sklearn.preprocessing import normalize
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    return X_lsi

def LSI_based_preprocess(adatas,
    dim: Optional[int] = 20,
    self_loop: Optional[bool] = False,
    join: Optional[str] = "inner",
    backend: Optional[str] = "sklearn",
    mincells_ratio : Optional[float] = 0.01,
    use_highly_variable: Optional[bool]=True,
    singular: Optional[bool] = True,
    check_order: Optional[bool] = True,
    n_top_genes: Optional[int] = 2500,
    device: Optional[str] = "cpu"):

    gpu_flag = True if torch.cuda.is_available() else False
    adata_all = adatas[0].concatenate(adatas[1], join=join)

    edgeLists = []
    for adata in adatas:
        G_df = adata.uns["Spatial_Net"].copy()
        cells = np.array(adata.obs_names)
        cells_id_tran = dict(zip(cells, range(cells.shape[0])))
        G_df["Cell1"] = G_df["Cell1"].map(cells_id_tran)
        G_df["Cell2"] = G_df["Cell2"].map(cells_id_tran)

        # build adjacent matrix
        G = scipy.sparse.coo_matrix(
            (np.ones(G_df.shape[0]), (G_df["Cell1"], G_df["Cell2"])),
            shape=(adata.n_obs, adata.n_obs),
        )
        if self_loop:
            G = G + scipy.sparse.eye(G.shape[0])
        edgeList = np.nonzero(G)
        edgeLists.append(edgeList)

    if use_highly_variable is True:
        print("Select Highly Variable Peaks")
        min_cells = int(adata_all.shape[0] * mincells_ratio)
        # in-place filtering of regions
        sc.pp.filter_genes(adata_all, min_cells=min_cells)
        print('The shape of Peak is ' + str(adata_all.shape[1]))

    adata_all.obsm['lsi'] = lsi(adata_all.X)
    adata_1 = adata_all[adata_all.obs['batch'] == '0']
    adata_2 = adata_all[adata_all.obs['batch'] == '1']

    Z_x, Z_y = Tensor(adata_1.obsm['lsi']).to(device), Tensor(adata_2.obsm['lsi']).to(device)

    data_x = Data(
        edge_index=torch.LongTensor(np.array([edgeLists[0][0], edgeLists[0][1]])), x=Z_x
    )
    data_y = Data(
        edge_index=torch.LongTensor(np.array([edgeLists[1][0], edgeLists[1][1]])), x=Z_y
    )
    datas = [data_x, data_y]    

    edges = [dataset.edge_index for dataset in datas]
    features = [dataset.x for dataset in datas]

    return edges, features

def find_best_matching(src, tgt, k_list=[3, 10, 40]):

    kd_tree = cKDTree(src.obsm['spatial'])
    knn_src_exp_base = src.obsm['emb'].copy()
    knn_src_exp = src.obsm['emb'].copy()
    if issparse(knn_src_exp_base):
        knn_src_exp_base = knn_src_exp_base.todense()
    if issparse(knn_src_exp):
        knn_src_exp = knn_src_exp.todense()
    if len(k_list) != 0:
        for k in k_list:
            distances, indices = kd_tree.query(src.obsm['spatial'], k=k)  # (source_num_points, k)
            knn_src_exp = knn_src_exp + np.array(np.mean(knn_src_exp_base[indices, :], axis=1))

    kd_tree = cKDTree(tgt.obsm['spatial'])
    knn_tgt_exp = tgt.obsm['emb'].copy()
    knn_tgt_exp_base = tgt.obsm['emb'].copy()
    if issparse(knn_tgt_exp_base):
        knn_tgt_exp_base = knn_tgt_exp_base.todense()
    if issparse(knn_tgt_exp):
        knn_tgt_exp = knn_tgt_exp.todense()
    if len(k_list) != 0:
        for k in k_list:
            distances, indices = kd_tree.query(tgt.obsm['spatial'], k=k)  # (source_num_points, k)
            knn_tgt_exp = knn_tgt_exp + np.array(np.mean(knn_tgt_exp_base[indices, :], axis=1))

    corr = np.corrcoef(knn_src_exp, knn_tgt_exp)[:knn_src_exp.shape[0],
           knn_src_exp.shape[0]:]  # (src_points, tgt_points)

    src.obsm['emb'] = knn_src_exp
    tgt.obsm['emb'] = knn_tgt_exp

    ''' find the spots which are possibly in the overlap region by L1 changepoint detection '''
    y = np.sort(np.max(corr, axis=0))[::-1]
    data = np.array(y).reshape(-1, 1)
    algo = rpt.Dynp(model="l1").fit(data)
    result = algo.predict(n_bkps=1)
    first_inflection_point = result[0]

    ### set1: For each of point in tgt, the corresponding best matched point in src
    set1 = np.array([[index, value]for index, value in enumerate(np.argmax(corr, axis=0))])
    set1 = np.column_stack((set1,np.max(corr, axis=0)))
    set1 = pd.DataFrame(set1,columns = ['tgt_index','src_index','corr'])
    set1.sort_values(by='corr',ascending=False,inplace=True)
    set1 = set1.iloc[:first_inflection_point,:]


    y = np.sort(np.max(corr, axis=1))[::-1]
    data = np.array(y).reshape(-1, 1)
    algo = rpt.Dynp(model="l1").fit(data)
    result = algo.predict(n_bkps=1)
    first_inflection_point = result[0]

    ### set2: For each of point in src, the corresponding best matched point in tgt
    set2 = np.array([[index, value]for index, value in enumerate(np.argmax(corr, axis=1))])
    set2 = np.column_stack((set2,np.max(corr, axis=1)))
    set2 = pd.DataFrame(set2,columns = ['src_index','tgt_index','corr'])
    set2.sort_values(by='corr',ascending=False,inplace=True)
    set2 = set2.iloc[:first_inflection_point,:]


    result = pd.merge(set1, set2, left_on=['tgt_index', 'src_index'], right_on=['tgt_index', 'src_index'], how='inner')
    src_sub = src[result['src_index'].to_numpy().astype(int), :]
    tgt_sub = tgt[result['tgt_index'].to_numpy().astype(int), :]

    slice1_index =  result['src_index'].values.flatten().astype(int)
    slice2_index =  result['tgt_index'].values.flatten().astype(int)
    pair_index = np.array([slice1_index,slice2_index])
    
    return pair_index.T

def dual_svd(adatas,
    dim: Optional[int] = 15,
    self_loop: Optional[bool] = False,
    join: Optional[str] = "inner",
    backend: Optional[str] = "sklearn",
    mincells_ratio : Optional[float] = 0.01,
    use_highly_variable: Optional[bool]=True,
    singular: Optional[bool] = True,
    device: Optional[str] = "cpu"):

    adata_all = adatas[0].concatenate(adatas[1], join=join)

    if use_highly_variable is True:
        print("Select Highly Variable Peaks")
        min_cells = int(adata_all.shape[0] * mincells_ratio)
        # in-place filtering of regions
        sc.pp.filter_genes(adata_all, min_cells=min_cells)
        print('The shape of Peak is ' + str(adata_all.shape[1]))
    
    sc.pp.normalize_total(adata_all)
    sc.pp.log1p(adata_all)
    adata_1 = adata_all[adata_all.obs['batch'] == '0']
    adata_2 = adata_all[adata_all.obs['batch'] == '1']
    sc.pp.scale(adata_1)
    sc.pp.scale(adata_2)
    if device != 'cpu':
        print(
            "Warning! Dual PCA is using GPU, which may lead to OUT OF GPU MEMORY in big dataset!"
        )
    if issparse(adata_1.X):
        adata_1.X = adata_1.X.todense()
    if issparse(adata_2.X):
        adata_2.X = adata_2.X.todense()

    Z_x, Z_y = dual_SVD(
        adata_1.X, adata_2.X, dim=dim, singular=singular, backend=backend, device = device
    )
    data_x = Data(x=Z_x)
    data_y = Data(x=Z_y)
    datas = [data_x, data_y]    

    features = [np.array(dataset.x) for dataset in datas]

    return features



def Transfer_pytorch_Data(adata):
    if('Spatial_Net' in adata.uns):
        G_df = adata.uns['Spatial_Net'].copy()
    elif('Feature_Net' in adata.uns):
        G_df = adata.uns['Feature_Net'].copy()
    else: 
        print('There is no Net that fulfills the conditions')  
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if type(adata.obsm['feat']) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['feat']))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(adata.obsm['feat'].toarray()))  # .todense()
    return data


def construct_interaction_Feature(adata, n_neighbors=6):
    # Step 1: 使用 adata.obsm['feat'] 作为特征来计算邻接矩阵
    feat = adata.obsm['feat']  # 这里假设 feat 是嵌入空间或特征矩阵
    n_spot = feat.shape[0]
    
    # Step 2: 计算 KNN，寻找最近的邻居
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(feat)  
    _, indices = nbrs.kneighbors(feat)
    
    # Step 3: 构建邻接矩阵
    x = indices[:, 0].repeat(n_neighbors)  # 重复邻居索引
    y = indices[:, 1:].flatten()  # 获取每个点的邻居索引
    interaction = np.zeros([n_spot, n_spot])  # 初始化邻接矩阵
    
    interaction[x, y] = 1  # 在邻接矩阵中标记连接关系
    
    adata.obsm['graph_neigh'] = interaction  # 将邻接矩阵存储到 adata 中

    # Step 4: 转换为对称的邻接矩阵（即双向边）
    adj = interaction
    adj = adj + adj.T  # 对称化
    adj = np.where(adj > 1, 1, adj)  # 确保不会有大于1的值
    np.fill_diagonal(adj, 1)  # 确保对角线为 1，表示每个点与自身相连
    
    return adj