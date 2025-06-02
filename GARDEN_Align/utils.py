import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import networkx as nx
import os
from tqdm import tqdm
from scipy.spatial import cKDTree
from typing import List, Optional, Union
import faiss
from anndata import AnnData
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

def build_nx_graph(edge_index, anchor_nodes, x=None):
    """
    Build a networkx graph from edge list and node attributes.
    :param edge_index: edge list of the graph
    :param anchor_nodes: anchor nodes
    :param x: node attributes of the graph
    :return: a networkx graph
    """

    G = nx.Graph()
    if x is not None:
        G.add_nodes_from(np.arange(x.shape[0]))
    G.add_edges_from(edge_index)
    G.x = x
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    G.anchor_nodes = anchor_nodes
    return G


def build_tg_graph(edge_index, x, rwr, dtype=torch.float32):
    """
    Build a PyG Data object from edge list and node attributes.
    :param edge_index: edge list of the graph
    :param x: node attributes of the graph
    :param rwr: random walk with restart scores
    :param dtype: data type
    :return: a PyG Data object
    """

    edge_index_tensor = torch.from_numpy(edge_index.T).to(torch.int64)
    x_tensor = torch.from_numpy(x).to(dtype)
    data = Data(x=x_tensor, edge_index=edge_index_tensor)
    data.rwr = torch.from_numpy(rwr).to(dtype)
    data.adj = to_dense_adj(edge_index_tensor).squeeze(0)
    return data



def get_rwr_matrix(G1, G2, anchor_links,device='cpu', dtype=np.float32):
    """
    Get distance matrix of the network
    :param G1: input graph 1
    :param G2: input graph 2
    :param anchor_links: anchor links
    :param dataset: dataset name
    :param ratio: training ratio
    :param dtype: data type
    :return: distance matrix (num of nodes x num of anchor nodes)
    """
    # if not os.path.exists(f'datasets/rwr'):
    #     os.makedirs(f'datasets/rwr')

    # rwr_path = f'datasets/rwr/rwr_emb_{dataset}_{ratio:.1f}.npz'
    # if os.path.exists(rwr_path):
    #     print(f"Loading RWR scores from {rwr_path}...", end=" ")
    #     data = np.load(rwr_path)
    #     rwr1, rwr2 = data['rwr1'], data['rwr2']
    #     print("Done")
    # else:
    rwr1, rwr2 = rwr_scores(G1, G2, anchor_links,device, dtype)
    # print(f"Saving RWR scores to {rwr_path}...", end=" ")
    # np.savez(rwr_path, rwr1=rwr1, rwr2=rwr2)
    # print("Done")

    return rwr1, rwr2

def rwr_scores(G1, G2, anchor_links,device='cpu',dtype=torch.float32):
    """
    Compute initial node embedding vectors by random walk with restart
    :param G1: network G1, i.e., networkx graph
    :param G2: network G2, i.e., networkx graph
    :param anchor_links: anchor links
    :param dtype: data type
    :return: rwr_score1, rwr_score2: RWR vectors of the networks
    """

    rwr_score1 = rwr_score_gpu(G1, anchor_links[:, 0], desc="Computing RWR scores for G1",device=device, dtype=dtype)
    rwr_score2 = rwr_score_gpu(G2, anchor_links[:, 1], desc="Computing RWR scores for G2",device=device, dtype=dtype)

    return rwr_score1, rwr_score2


def find_rigid_transform(A, B):
    assert A.shape == B.shape

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)


    A_centered = A - centroid_A
    B_centered = B - centroid_B

    H = A_centered.T @ B_centered

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t
def rotation_angle_2d(R):
    theta = np.arctan2(R[1, 0], R[0, 0])
    return np.degrees(theta)
def evaluation(src_cor, tgt_cor, src_exp, tgt_exp, src_cell_type, tgt_cell_type):

    kd_tree = cKDTree(src_cor)
    distances, indices = kd_tree.query(tgt_cor, k=1) 
    corr = np.corrcoef(np.concatenate((tgt_exp,src_exp[indices]), axis=0))[:tgt_exp.shape[0],tgt_exp.shape[0]:]
    acc = corr.trace()/tgt_exp.shape[0]
    cri = np.mean((tgt_cell_type == src_cell_type[indices])+0)
    #euc = np.mean((ori_src_cor-src_cor)**2)
    
    return acc, cri
def simulate_stitching(adata,axis = 0, from_low = True, threshold = 0.5):
    cadata = adata.copy()
    coo = cadata.obsm['spatial']
    scale = np.max(coo[:, axis]) - np.min(coo[:, axis])
    if from_low:
        chosen_indices = coo[:,axis] > (scale * threshold + np.min(coo[:, axis]))
    else:
        chosen_indices = coo[:,axis] < (np.max(coo[:, axis]) - scale * threshold)
    cadata = cadata[chosen_indices,:]
    return cadata
def intersect(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3

def spatial_match(embds:List[torch.Tensor],
                  reorder:Optional[bool]=True,
                  top_n:Optional[int]=20,
                  smooth:Optional[bool]=True,
                  smooth_range:Optional[int]=20,
                  scale_coord:Optional[bool]=True,
                  adatas:Optional[List[AnnData]]=None,
                  verbose:Optional[bool]=False
    )-> List[Union[np.ndarray,torch.Tensor]]:
    r"""
    Use embedding to match cells from different datasets based on cosine similarity
    
    Parameters
    ----------
    embds
        list of embeddings
    reorder
        if reorder embedding by cell numbers
    top_n
        return top n of cosine similarity
    smooth
        if smooth the mapping by Euclid distance
    smooth_range
        use how many candidates to do smooth
    scale_coord
        if scale the coordinate to [0,1]
    adatas
        list of adata object
    verbose
        if print log
    
    Note
    ----------
    Automatically use larger dataset as source
    
    Return
    ----------
    Best matching, Top n matching and cosine similarity matrix of top n  
    
    Note
    ----------
    Use faiss to accelerate, refer https://github.com/facebookresearch/faiss/issues/95
    """
    if reorder and embds[0].shape[0] < embds[1].shape[0]:
        embd0 = embds[1]
        embd1 = embds[0]
        adatas = adatas[::-1] if adatas is not None else None
    else:
        embd0 = embds[0]
        embd1 = embds[1]
    index = faiss.index_factory(embd1.shape[1], "Flat", faiss.METRIC_INNER_PRODUCT)
    embd0_np = embd0.detach().cpu().numpy() if torch.is_tensor(embd0) else embd0
    embd1_np = embd1.detach().cpu().numpy() if torch.is_tensor(embd1) else embd1
    embd0_np = embd0_np.copy().astype('float32')
    embd1_np = embd1_np.copy().astype('float32')
    faiss.normalize_L2(embd0_np)
    faiss.normalize_L2(embd1_np)
    index.add(embd0_np)
    distance, order = index.search(embd1_np, top_n)
    best = []
    adata1_coord = adatas[0].obsm['spatial'].copy()
    adata2_coord = adatas[1].obsm['spatial'].copy()
    if smooth and adatas != None:
        smooth_range = min(smooth_range, top_n)
        if verbose:
            print('Smoothing mapping, make sure object is in same direction')
        if scale_coord:
            # scale spatial coordinate of every adata to [0,1]
            adata1_coord = adatas[0].obsm['spatial'].copy()
            adata2_coord = adatas[1].obsm['spatial'].copy()
            for i in range(2):
                    adata1_coord[:,i] = (adata1_coord[:,i]-np.min(adata1_coord[:,i]))/(np.max(adata1_coord[:,i])-np.min(adata1_coord[:,i]))
                    adata2_coord[:,i] = (adata2_coord[:,i]-np.min(adata2_coord[:,i]))/(np.max(adata2_coord[:,i])-np.min(adata2_coord[:,i]))
        for query in range(embd1_np.shape[0]):
            ref_list = order[query, :smooth_range]
            dis = euclidean_distances(adata2_coord[query,:].reshape(1, -1), 
                                      adata1_coord[ref_list,:])
            best.append(ref_list[np.argmin(dis)])
    else:
        best = order[:,0]
    return np.array(best), order, distance

import numpy as np
import ruptures as rpt
from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity

def detect_inflection_point(corr):
    """使用 Dynp 变点检测算法找出第一突变点"""
    y = np.sort(np.max(corr, axis=0))[::-1]  # 取相关性最大值并排序
    data = np.array(y).reshape(-1, 1)  # 变成列向量

    algo = rpt.Dynp(model="l1").fit(data)
    result = algo.predict(n_bkps=1)  # 计算变点
    first_inflection_point = result[0]  # 变点位置
    return first_inflection_point


def generate_positive_pairs_spatial_and_feature(src, k_spatial=6): 
    """
    在空间邻域内筛选特征相似性PCC大于阈值的 positive 锚点对，并基于变点检测进行过滤。
    :param src: 切片数据（包含空间坐标和特征）
    :param k_spatial: 空间坐标上的最近邻数量
    :return: positive 锚点对 (n_pairs, 2)
    """
    spatial_coords = src.obsm['spatial']  # 空间坐标 (n_cells, 2)
    features = src.obsm['emb']  # 特征矩阵 (n_cells, n_features)

    # 构建 k 近邻树
    kd_tree = cKDTree(spatial_coords)
    distances, spatial_indices = kd_tree.query(spatial_coords, k=k_spatial+1)  # 包含自身，所以 k+1

    # 计算 Pearson 相关系数矩阵（PCC）
    feature_similarity = np.corrcoef(features)  # 计算 PCC 矩阵
    mean_feature_similarity = np.mean(feature_similarity)

    # 计算变点检测的 inflection point
    first_inflection_point = detect_inflection_point(feature_similarity)

    # 生成 positive 锚点对
    positive_pairs = []
    for i in range(len(spatial_coords)):
        spatial_neighbors = spatial_indices[i][1:]  # 跳过自身
        
        for j in spatial_neighbors:
            if feature_similarity[i, j] > mean_feature_similarity:
                positive_pairs.append([i, j])

    # 变点检测过滤
    positive_pairs = positive_pairs[:first_inflection_point]

    return np.array(positive_pairs)

def get_closest_half_matches_median(slice1, slice2):
    # 提取空间坐标
    coord1 = slice1.obsm['spatial']
    coord2 = slice2.obsm['spatial']

    # 计算距离矩阵
    dist_matrix = euclidean_distances(coord2, coord1)

    # 找到每个 slice2 点到 slice1 最近点的索引和距离
    closest_indices = np.argmin(dist_matrix, axis=1)
    closest_distances = np.min(dist_matrix, axis=1)

    # 计算中位数
    median_dist = np.median(closest_distances)

    # 保留距离小于等于中位数的索引
    keep_indices = np.where(closest_distances <= median_dist)[0]

    # 构建匹配对
    matching = np.vstack([keep_indices, closest_indices[keep_indices]])

    return matching

def get_closest_half_matches(slice1, slice2, keep_ratio=0.8):
    # 提取空间坐标
    coord1 = slice1.obsm['spatial']
    coord2 = slice2.obsm['spatial']

    # 计算距离矩阵：每个slice2点到所有slice1点的距离
    dist_matrix = euclidean_distances(coord2, coord1)

    # 找到每个slice2点在slice1中最近的点
    closest_indices = np.argmin(dist_matrix, axis=1)  # 长度为 len(slice2)
    closest_distances = np.min(dist_matrix, axis=1)

    # 排序距离，保留前50%最近的匹配
    num_keep = int(len(closest_indices)*keep_ratio)
    keep_indices = np.argsort(closest_distances)[:num_keep]

    # 构建匹配对：第一行是slice2的索引，第二行是对应slice1最近邻的索引
    matching = np.vstack([keep_indices, closest_indices[keep_indices]])

    return matching

def rwr_score_gpu(G, anchors, restart_prob=0.15, desc='Computing RWR scores', dtype=torch.float32, device='cpu'):
    """
    GPU-accelerated RWR using PyTorch
    :param G: networkx Graph
    :param anchors: anchor node list
    :param restart_prob: restart probability
    :param desc: tqdm description
    :param dtype: torch dtype
    :param device: 'cuda' or 'cpu'
    :return: torch.Tensor of shape [n_nodes, len(anchors)]
    """
    import scipy.sparse as sp

    # Get number of nodes and consistent ordering
    nodes = list(G.nodes())
    node_idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)

    # Construct sparse normalized adjacency matrix
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=np.float32)
    row_sum = np.array(A.sum(axis=1)).flatten()
    row_sum[row_sum == 0] = 1  # avoid division by zero
    D_inv = sp.diags(1.0 / row_sum)
    P = D_inv @ A  # transition probability matrix

    # Convert to torch sparse tensor
    P_coo = P.tocoo()
    indices = torch.tensor([P_coo.row, P_coo.col], dtype=torch.long)
    values = torch.tensor(P_coo.data, dtype=dtype)
    P_sparse = torch.sparse_coo_tensor(indices, values, size=(n, n), device=device)

    # Prepare output tensor
    rwr = torch.zeros((n, len(anchors)), dtype=dtype, device=device)

    for i, anchor in enumerate(tqdm(anchors, desc=desc)):
        # Initial restart vector
        e = torch.zeros(n, dtype=dtype, device=device)
        e[node_idx[anchor]] = 1.0

        # Iterative RWR
        p = e.clone()
        for _ in range(30):  # fixed iteration or add convergence check
            p_new = (1 - restart_prob) * torch.sparse.mm(P_sparse, p.unsqueeze(1)).squeeze(1) + restart_prob * e
            if torch.norm(p_new - p, p=1) < 1e-6:
                break
            p = p_new

        rwr[:, i] = p

    return rwr.cpu().numpy()