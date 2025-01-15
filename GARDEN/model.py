
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from .gat_conv_pyg import GATv2Conv

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits
    
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
          
        return F.normalize(global_emb, p=2, dim=1) 
    
class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, data ,dropout=0.0, act=F.relu    ):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        # after we use self.data
        self.data = data
        
        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        #print(self.out_features)
        self.zip_layer = GATv2Conv(self.in_features, self.out_features ,heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.eco_layer = GATv2Conv(self.out_features, self.in_features ,heads=1, concat=False,
                           dropout=0, add_self_loops=False, bias=False)
        
        #self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.zip_layer.weight)
        torch.nn.init.xavier_uniform_(self.eco_layer.weight)
        
    def forward(self, feat, feat_a, adj):
        z = self.zip_layer(feat,self.data.edge_index)
        z = torch.mm(adj, z)

        z_a = self.zip_layer(feat_a, self.data.edge_index)
        z_a = torch.mm(adj, z_a)

        hiden_emb = z

        h = self.eco_layer(z,self.data.edge_index)
        h = torch.mm(adj, h)

        emb = self.act(z)
        emb_a = self.act(z_a)

        g = self.read(emb, self.graph_neigh) 
        g = self.sigm(g)  

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)  

        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        
        return hiden_emb, h, ret, ret_a
    
class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh, data ,dropout=0.0, act=F.relu    ):
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act

        # after we use self.data
        self.data = data
        
        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        #print(self.out_features)
        self.zip_layer = GATv2Conv(self.in_features, self.out_features ,heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.eco_layer = GATv2Conv(self.out_features, self.in_features ,heads=1, concat=False,
                           dropout=0, add_self_loops=False, bias=False)
        
        #self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.zip_layer.weight)
        torch.nn.init.xavier_uniform_(self.eco_layer.weight)
        
    def forward(self, feat, feat_a, adj):
        z = self.zip_layer(feat,self.data.edge_index)
        z = torch.mm(adj, z)

        z_a = self.zip_layer(feat_a, self.data.edge_index)
        z_a = torch.mm(adj, z_a)

        hiden_emb = z

        h = self.eco_layer(z,self.data.edge_index)
        h = torch.mm(adj, h)

        emb = self.act(z)
        emb_a = self.act(z_a)

        g = self.read(emb, self.graph_neigh) 
        g = self.sigm(g)  

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)  

        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        
        return hiden_emb, h, ret, ret_a

class Batch_Encoder(Module):
    def __init__(self, in_features, out_features,dropout=0.0, act=F.relu    ):
        super(Batch_Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        # after we use self.data
        
        self.disc = Discriminator(self.out_features)
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()
        #print(self.out_features)
        self.zip_layer = GATv2Conv(self.in_features, self.out_features ,heads=1, concat=False,
                             dropout=0, add_self_loops=False, bias=False)
        self.eco_layer = GATv2Conv(self.out_features, self.in_features ,heads=1, concat=False,
                           dropout=0, add_self_loops=False, bias=False)
        
        
    def forward(self, feat, feat_a, graph_neigh, data, adj):
        z = self.zip_layer(feat,data.edge_index)
        z = torch.mm(adj, z)

        z_a = self.zip_layer(feat_a, data.edge_index)
        z_a = torch.mm(adj, z_a)

        hiden_emb = z

        h = self.eco_layer(z,data.edge_index)
        h = torch.mm(adj, h)

        emb = self.act(z)
        emb_a = self.act(z_a)

        g = self.read(emb, graph_neigh) 
        g = self.sigm(g)  

        g_a = self.read(emb_a, graph_neigh)
        g_a = self.sigm(g_a)  

        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        
        return hiden_emb, h, ret, ret_a
    
    def predict(self, feat, feat_a, data):
        z = self.zip_layer(feat,data.edge_index)
        h = self.eco_layer(z,data.edge_index)
        return h