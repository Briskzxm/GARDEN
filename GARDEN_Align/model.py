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

class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(input_dim, hidden_dim)
        self.lin2 = nn.Linear(input_dim + hidden_dim, output_dim)
        self.act = nn.ReLU()
        
        # 增加 LayerNorm
        self.norm1 = nn.LayerNorm(hidden_dim)  # 对 lin1 的输出进行归一化
        self.norm2 = nn.LayerNorm(output_dim)  # 对 lin2 的输出进行归一化

    def forward(self, G1_data, G2_data):
        x1, x2 = G1_data.x, G2_data.x
        
        # Dropout
        x1 = F.dropout(x1, 0.2, self.training)
        x2 = F.dropout(x2, 0.2, self.training)
        
        # 第一层：线性变换 + 激活函数 + LayerNorm
        h1 = self.act(self.norm1(self.lin1(x1)))  # 对 lin1 的输出进行归一化
        h2 = self.act(self.norm1(self.lin1(x2)))
        
        # 拼接输入和隐藏层特征
        pos_emd1 = torch.cat([x1, h1], dim=1)
        pos_emd2 = torch.cat([x2, h2], dim=1)
        
        # 第二层：线性变换 + LayerNorm
        pos_emd1 = self.norm2(self.lin2(pos_emd1))  # 对 lin2 的输出进行归一化
        pos_emd2 = self.norm2(self.lin2(pos_emd2))
        
        # 归一化输出
        pos_emd1 = F.normalize(pos_emd1, p=2, dim=1)
        pos_emd2 = F.normalize(pos_emd2, p=2, dim=1)
        
        return pos_emd1, pos_emd2
    

class FusedGWLoss(torch.nn.Module):
    def __init__(self, G1_tg, G2_tg, anchor1, anchor2, gw_weight=20, gamma_p=1e-2, init_threshold_lambda=1, in_iter=5,
                 out_iter=10, total_epochs=250):
        super().__init__()
        self.device = G1_tg.x.device
        self.gw_weight = gw_weight
        self.gamma_p = gamma_p
        self.in_iter = in_iter
        self.out_iter = out_iter
        self.total_epochs = total_epochs

        self.n1, self.n2 = G1_tg.num_nodes, G2_tg.num_nodes
        self.threshold_lambda = init_threshold_lambda / (self.n1 * self.n2)
        self.adj1, self.adj2 = G1_tg.adj, G2_tg.adj
        self.H = torch.ones(self.n1, self.n2).to(torch.float64).to(self.device)
        self.H[anchor1, anchor2] = 0

    def forward(self, out1, out2):
        inter_c = torch.exp(-(out1 @ out2.T))
        intra_c1 = torch.exp(-(out1 @ out1.T)) * self.adj1
        intra_c2 = torch.exp(-(out2 @ out2.T)) * self.adj2
        with torch.no_grad():
            s = sinkhorn_stable(inter_c, intra_c1, intra_c2,
                                gw_weight=self.gw_weight,
                                gamma_p=self.gamma_p,
                                threshold_lambda=self.threshold_lambda,
                                in_iter=self.in_iter,
                                out_iter=self.out_iter,
                                device=self.device)
            self.threshold_lambda = 0.05 * self.update_lambda(inter_c, intra_c1, intra_c2, s, self.device) + 0.95 * self.threshold_lambda

        s_hat = s - self.threshold_lambda

        # Wasserstein Loss
        w_loss = torch.sum(inter_c * s_hat)

        # Gromov-Wasserstein Loss
        a = torch.sum(s_hat, dim=1)
        b = torch.sum(s_hat, dim=0)
        gw_loss = torch.sum(
            (intra_c1 ** 2 @ a.view(-1, 1) @ torch.ones((1, self.n2)).to(torch.float64).to(self.device) +
             torch.ones((self.n1, 1)).to(torch.float64).to(self.device) @ b.view(1, -1) @ intra_c2 ** 2 -
             2 * intra_c1 @ s_hat @ intra_c2.T) * s_hat)

        loss = w_loss + self.gw_weight * gw_loss + 20
        return loss, s, self.threshold_lambda

    def update_lambda(self, inter_c, intra_c1, intra_c2, s, device):
        k1 = torch.sum(inter_c)

        one_mat = torch.ones(self.n1, self.n2).to(torch.float64).to(device)
        mid = intra_c1 ** 2 @ one_mat * self.n2 + one_mat @ intra_c2 ** 2 * self.n1 - 2 * intra_c1 @ one_mat @ intra_c2.T
        k2 = torch.sum(mid * s)
        k3 = torch.sum(mid)

        return (k1 + 2 * self.gw_weight * k2) / (2 * self.gw_weight * k3)


def sinkhorn_stable(inter_c, intra_c1, intra_c2, threshold_lambda, in_iter=5, out_iter=10, gw_weight=20, gamma_p=1e-2,
                    device='cpu'):
    n1, n2 = inter_c.shape
    # marginal distribution
    a = torch.ones(n1).to(torch.float64).to(device) / n1
    b = torch.ones(n2).to(torch.float64).to(device) / n2
    # lagrange multiplier
    f = torch.ones(n1).to(torch.float64).to(device) / n1
    g = torch.ones(n2).to(torch.float64).to(device) / n2
    # transport plan
    s = torch.ones((n1, n2)).to(torch.float64).to(device) / (n1 * n2)

    def soft_min_row(z_in, eps):
        hard_min = torch.min(z_in, dim=1, keepdim=True)[0]
        soft_min = hard_min - eps * torch.log(torch.sum(torch.exp(-(z_in - hard_min) / eps), dim=1, keepdim=True))
        return soft_min.squeeze(-1)

    def soft_min_col(z_in, eps):
        hard_min = torch.min(z_in, dim=0, keepdim=True)[0]
        soft_min = hard_min - eps * torch.log(torch.sum(torch.exp(-(z_in - hard_min) / eps), dim=0, keepdim=True))
        return soft_min.squeeze(0)

    for i in range(out_iter):
        a_hat = torch.sum(s - threshold_lambda, dim=1)
        b_hat = torch.sum(s - threshold_lambda, dim=0)
        temp = (intra_c1 ** 2 @ a_hat.view(-1, 1) @ torch.ones((1, n2)).to(torch.float64).to(device) +
                torch.ones((n1, 1)).to(torch.float64).to(device) @ b_hat.view(1, -1) @ intra_c2 ** 2)
        L = temp - 2 * intra_c1 @ (s - threshold_lambda) @ intra_c2.T
        cost = inter_c + gw_weight * L

        Q = cost
        for j in range(in_iter):
            # log-sum-exp stabilization
            f = soft_min_row(Q - g.view(1, -1), gamma_p) + gamma_p * torch.log(a)
            g = soft_min_col(Q - f.view(-1, 1), gamma_p) + gamma_p * torch.log(b)
        s = 0.05 * s + 0.95 * torch.exp((f.view(-1, 1) + g.view(-1, 1).T - Q) / gamma_p)

    return s

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

class ReconDNN(nn.Module):
    r"""
    Data reconstruction network

    Parameters
    ----------
    hidden_size
        input dim
    feature_size
        output size (feature input size)
    hidden_size2
        hidden size
    """

    def __init__(self, hidden_size: int, feature_size: int, hidden_size2: Optional[int] = 512):
        super().__init__()
        self.hidden = nn.Linear(hidden_size, hidden_size2)
        self.output = nn.Linear(hidden_size2, feature_size)

    def forward(self, input_embd: torch.Tensor):
        return self.output(F.relu(self.hidden(input_embd)))

def feature_reconstruct_loss(
    embd: torch.Tensor, x: torch.Tensor, recon_model: torch.nn.Module
) -> torch.Tensor:
    r"""
    Reconstruction loss (MSE)

    Parameters
    ----------
    embd
        embd of a cell
    x
        input
    recon_model
        reconstruction model
    """
    recon_x = recon_model(embd)
    return torch.norm(recon_x - x, dim=1, p=2).mean()
