import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Any,Optional,List,Union
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
import pynvml

def run_LGCN(features: List[np.ndarray], edges: List[torch.Tensor], LGCN_layer: int = 2):
    """
    Run LGCN model

    Parameters
    ----------
    features
        list of graph node features
    edges
        list of graph edges
    LGCN_layer
        LGCN layer number, we suggest set 2 for barcode based and 4 for fluorescence based
    """
    if torch.cuda.is_available():
        gpu_index = get_free_gpu()
        #print(f"Choose GPU:{gpu_index} as device")
        device = torch.device(f"cuda:{gpu_index}")
    else:
        device = torch.device("cpu")
        print("GPU is not available")

    for i in range(len(features)):
        features[i] = features[i].to(device)
    for j in range(len(edges)):
        edges[j] = edges[j].to(device)

    LGCN_model = LGCN(K=LGCN_layer).to(device=device)

    time1 = time.time()
    embd0 = LGCN_model(features[0], edges[0])
    embd1 = LGCN_model(features[1], edges[1])

    run_time = time.time() - time1
    #print(f"LGCN time: {run_time}")
    return embd0, embd1, run_time

def get_free_gpu() -> int:
    r"""
    Get index of GPU with least memory usage

    Ref
    ----------
    https://stackoverflow.com/questions/58216000/get-total-amount-of-free-gpu-memory-and-available-using-pytorch
    """
    index = 0
    if torch.cuda.is_available():
        pynvml.nvmlInit()
        max = 0
        for i in range(torch.cuda.device_count()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            index = i if info.free > max else index
            max = info.free if info.free > max else max
    return index

def sym_norm(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weight: Optional[Union[Any, torch.Tensor]] = None,
    improved: Optional[bool] = False,
    dtype: Optional[Any] = None,
) -> List:
    r"""
    Replace `GCNConv.norm` from https://github.com/mengliu1998/DeeperGNN/issues/2
    """
    from torch_scatter import scatter_add

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

    fill_value = 1 if not improved else 2
    edge_index, edge_weight = add_remaining_self_loops(
        edge_index, edge_weight, fill_value, num_nodes
    )

    row, col = edge_index
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class CombUnweighted(MessagePassing):
    r"""
    LGCN (GCN without learnable and concat)

    Parameters
    ----------
    K
        K-hop neighbor to propagate
    """

    def __init__(
        self,
        K: Optional[int] = 1,
        cached: Optional[bool] = False,
        bias: Optional[bool] = True,
        **kwargs,
    ):
        super().__init__(aggr="add", **kwargs)
        self.K = K

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Union[torch.Tensor, None] = None,
    ):
        # edge_index, norm = GCNConv.norm(edge_index, x.size(0), edge_weight,
        #                                 dtype=x.dtype)
        edge_index, norm = sym_norm(edge_index, x.size(0), edge_weight, dtype=x.dtype)

        xs = [x]
        for k in range(self.K):
            xs.append(self.propagate(edge_index, x=xs[-1], norm=norm))
        return torch.cat(xs, dim=1)
        # return torch.stack(xs, dim=0).mean(dim=0)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return f"{self.__class__.__name__}(K={self.K})"

class LGCN(nn.Module):
    r"""
    Lightweight GCN which remove nonlinear functions and concatenate the embeddings of each layer:

        (:math:`Z = f_{e}(A, X) = Concat( [X, A_{X}, A_{2X}, ..., A_{KX}])W_{e}`)

    Parameters
    ----------
    K
        layers of LGCN
    """

    def __init__(self, K: Optional[int] = 8):
        super().__init__()
        self.conv1 = CombUnweighted(K=K)

    def forward(self, feature: torch.Tensor, edge_index: torch.Tensor):
        x = self.conv1(feature, edge_index)
        return x


