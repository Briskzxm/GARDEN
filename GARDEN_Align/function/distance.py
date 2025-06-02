import numpy as np
import networkx as nx
import os
from tqdm import tqdm


def get_rwr_matrix(G1, G2, anchor_links, dtype=np.float32):
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
    rwr1, rwr2 = rwr_scores(G1, G2, anchor_links, dtype)
    # print(f"Saving RWR scores to {rwr_path}...", end=" ")
    # np.savez(rwr_path, rwr1=rwr1, rwr2=rwr2)
    # print("Done")

    return rwr1, rwr2


def rwr_scores(G1, G2, anchor_links, dtype=np.float32):
    """
    Compute initial node embedding vectors by random walk with restart
    :param G1: network G1, i.e., networkx graph
    :param G2: network G2, i.e., networkx graph
    :param anchor_links: anchor links
    :param dtype: data type
    :return: rwr_score1, rwr_score2: RWR vectors of the networks
    """

    rwr_score1 = rwr_score(G1, anchor_links[:, 0], desc="Computing RWR scores for G1", dtype=dtype)
    rwr_score2 = rwr_score(G2, anchor_links[:, 1], desc="Computing RWR scores for G2", dtype=dtype)

    return rwr_score1, rwr_score2


def rwr_score(G, anchors, restart_prob=0.15, desc='Computing RWR scores', dtype=np.float32):
    """
    Random walk with restart for a single graph
    :param G: network G, i.e., networkx graph
    :param anchors: anchor nodes
    :param restart_prob: restart probability
    :param desc: description for tqdm
    :param dtype: data type
    :return: rwr: rwr vectors of the network
    """

    n = G.number_of_nodes()
    rwr = np.zeros((n, len(anchors))).astype(dtype)

    for i, node in enumerate(tqdm(anchors, desc=desc)):
        s = nx.pagerank(G, personalization={node: 1}, alpha=1-restart_prob)
        for k, v in s.items():
            rwr[k, i] = v

    return rwr
