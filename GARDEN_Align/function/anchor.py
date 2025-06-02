import numpy as np
import torch


def get_random_anchorsets(num, c=0.5):
    """
    Generate random anchor-sets
    :param num: number of anchor nodes
    :param c: hyperparameter for the number of anchor-sets (k=clog^2(n))
    :return: anchorsets: list of anchor-sets (idx of anchor nodes, need to re-map to the original node idx)
    """
    m = int(np.log2(num))
    copy = int(c * m)
    anchorsets = []
    for i in range(m):
        anchor_size = int(num / np.exp2(i + 1))
        for j in range(copy):
            anchorsets.append(np.random.choice(num, size=anchor_size, replace=False))
    return anchorsets


def get_dist_max(anchorsets, dist, device):
    """
    Get max distance & node id for each anchor-set
    :param anchorsets: list of anchor-sets
    :param dist: distance matrix (num of nodes x num of anchor nodes)
    :param device: device
    :return:
        dist_max: max distance for each anchor-set (num of nodes x num of anchor-sets)
        dist_argmax: argmax distance for each anchor-set (num of nodes x num of anchor-sets), need to re-map to the original node idx
    """
    n, k = dist.shape[0], len(anchorsets)
    dist_max = torch.zeros((n, k)).to(device)
    dist_argmax = torch.zeros((n, k)).long().to(device)
    for i, anchorset in enumerate(anchorsets):
        anchorset = torch.as_tensor(anchorset, dtype=torch.long)
        dist_nodes_anchorset = dist[:, anchorset]
        dist_max_anchorset, dist_argmax_anchorset = torch.max(dist_nodes_anchorset, dim=-1)
        dist_max[:, i] = dist_max_anchorset
        dist_argmax[:, i] = anchorset[dist_argmax_anchorset]
    return dist_max, dist_argmax


def preselect_anchor(G1_data, G2_data, random=False, c=1, device='cpu'):
    """
    Preselect anchor-sets
    :param G1_data: PyG Data object for graph 1
    :param G2_data: PyG Data object for graph 2
    :param random: whether to sample random anchor-sets
    :param c: hyperparameter for the number of anchor-sets (k=clog^2(n))
    :param device: device
    :return:
        dists_max: max distance for each anchor-set (num of nodes x num of anchor-sets)
        dists_argmax: argmax distance for each anchor-set (num of nodes x num of anchor-sets)
    """
    assert G1_data.anchor_nodes.shape[0] == G2_data.anchor_nodes.shape[0], 'Number of anchor links of G1 and G2 should be the same'

    num_anchor_nodes = G1_data.anchor_nodes.shape[0]
    if random:
        anchorsets = get_random_anchorsets(num_anchor_nodes, c=c)
        G1_dists_max, G1_dists_argmax = get_dist_max(anchorsets, G1_data.dists, device)
        G2_dists_max, G2_dists_argmax = get_dist_max(anchorsets, G2_data.dists, device)
    else:
        G1_dists_max, G1_dists_argmax = G1_data.dists, torch.arange(num_anchor_nodes).repeat(G1_data.num_nodes, 1).to(device)
        G2_dists_max, G2_dists_argmax = G2_data.dists, torch.arange(num_anchor_nodes).repeat(G2_data.num_nodes, 1).to(device)

    return (G1_dists_max, G1_data.anchor_nodes[G1_dists_argmax].to(device),
            G2_dists_max, G2_data.anchor_nodes[G2_dists_argmax].to(device))


def test_consistency(G1_data, G2_data):
    """
    Test the consistency of the anchor nodes
    :param G1_data: PyG Data object for graph 1
    :param G2_data: PyG Data object for graph 2
    """
    anchor_links_list = [(G1_data.anchor_nodes[i], G2_data.anchor_nodes[i]) for i in range(G1_data.anchor_nodes.shape[0])]
    for node1, node2 in anchor_links_list:
        sampled_anchorsets_1 = G1_data.dists_argmax[node1]
        sampled_anchorsets_2 = G2_data.dists_argmax[node2]
        for rep_node1, rep_node2 in zip(sampled_anchorsets_1, sampled_anchorsets_2):
            assert (rep_node1, rep_node2) in anchor_links_list, 'Inconsistent anchor nodes, space disparity detected'
    print('Anchor nodes are consistent')
