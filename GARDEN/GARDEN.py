import torch
from .preprocess import preprocess_adj, preprocess_adj_sparse, preprocess, construct_interaction, add_contrastive_label, get_feature, permutation, fix_seed
import time
import gc
import random
import numpy as np
from .model import Dynamic_Atten_Autoencoder,Batch_Dynamic_Atten_Autoencoder
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
import pandas as pd
import torch.sparse as sp
from .utils import *
import math

class GARDEN():
    def __init__(self, 
        adata,
        adata_sc = None,
        device= torch.device('cpu'),
        learning_rate=0.001,
        weight_decay=0.00,
        epochs=800,
        dim_input=3000,
        dim_output=64,
        random_seed = 42,
        alpha = 10,
        beta = 4,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10X',
	    rad_cutoff = 150,
        k_dynamic = 3,
        radius = 50,
        k_cl = 4,
        model_select = 'Radius'
        ):
        
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype
        self.rad_cutoff=rad_cutoff
        self.k_dynamic=k_dynamic
        self.radius=radius
        self.k_cl=k_cl
        self.model_select = model_select
        self.dim_output = dim_output

        fix_seed(self.random_seed)
        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata)

        if datatype !='HD':
            if 'adj' not in adata.obsm.keys():
                construct_interaction(self.adata,radius=self.radius,neighborhood=self.k_cl)
                self.adj = self.adata.obsm['adj']
                self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
        if 'Spatial_Net' not in self.adata.uns:    
            Cal_Spatial_Net(self.adata, rad_cutoff=self.rad_cutoff, k_cutoff=self.k_dynamic, model = self.model_select)

        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
           get_feature(self.adata)

        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
        self.dim_input = self.features.shape[1]

        self.adj = preprocess_adj(self.adj)
        self.adj = torch.FloatTensor(self.adj).to(self.device)
            
    def train(self):
        self.adata.X = csr_matrix(self.adata.X)

        if 'highly_variable' in self.adata.var.columns:
            adata_Vars = self.adata[:, self.adata.var['highly_variable']]
        else:
            adata_Vars = self.adata

        if 'Spatial_Net' not in self.adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

        self.data = Transfer_pytorch_Data(adata_Vars).to(self.device) # include data.x, data.edge_index
        data = self.data

        self.model = Dynamic_Atten_Autoencoder(self.dim_input, self.dim_output, self.graph_neigh, data = data).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                         weight_decay=self.weight_decay)

        print('Begin to train ST data...')
        self.model.train()
        for epoch in tqdm(range(self.epochs)): 
            self.features_a = permutation(self.features)
            self.hiden_feat, self.emb, ret, ret_a = self.model(self.features, self.features_a, self.adj)
            self.loss_sl_1 = F.binary_cross_entropy(ret, self.label_CSL)
            self.loss_sl_2 = F.binary_cross_entropy(ret_a, self.label_CSL)
            self.loss_feat = F.mse_loss(self.features, self.emb)
            loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2)
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()        
        print("Optimization finished for ST data!")
        
        with torch.no_grad():
            self.model.eval()
            if self.datatype in ['Stereo', 'Slide']:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
            else:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1].detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec
            return self.adata
        
    def get_adj(self):
        return self.adj   
         
    def get_snet(self):
        return self.adata.uns['Spatial_Net']
    
class GARDEN_Batch():
    def __init__(self, 
        adata,
        adata_sc = None,
        device= torch.device('cpu'),
        learning_rate=0.001,
        weight_decay=0.00,
        epochs=800,
        dim_input=3000,
        dim_output=64,
        random_seed = 42,
        alpha = 10,
        beta = 4,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        deconvolution = False,
        datatype = '10X',
	    rad_cutoff = 150,
        k_dynamic = 3,
        radius = 50,
        k_cl = 4,
        model_select = 'Radius'
        ):
        '''\

        Parameters
        ----------
        adata : anndata
            AnnData object of spatial data.
        adata_sc : anndata, optional
            AnnData object of scRNA-seq data. adata_sc is needed for deconvolution. The default is None.
        device : string, optional
            Using GPU or CPU? The default is 'cpu'.
        learning_rate : float, optional
            Learning rate for ST representation learning. The default is 0.001.
        weight_decay : float, optional
            Weight factor to control the influence of weight parameters. The default is 0.00.
        epochs : int, optional
            Epoch for model training. The default is 600.
        dim_input : int, optional
            Dimension of input feature. The default is 3000.
        dim_output : int, optional
            Dimension of output representation. The default is 64.
        random_seed : int, optional
            Random seed to fix model initialization. The default is 41.
        alpha : float, optional
            Weight factor to control the influence of reconstruction loss in representation learning. 
            The default is 10.
        beta : float, optional
            Weight factor to control the influence of contrastive loss in representation learning. 
            The default is 1.
        lamda1 : float, optional
            Weight factor to control the influence of reconstruction loss in mapping matrix learning. 
            The default is 10.
        lamda2 : float, optional
            Weight factor to control the influence of contrastive loss in mapping matrix learning. 
            The default is 1.
        deconvolution : bool, optional
            Deconvolution task? The default is False.
        datatype : string, optional    
            Data type of input. Our model supports 10X Visium ('10X'), Stereo-seq ('Stereo'), and Slide-seq/Slide-seqV2 ('Slide') data. 
        Returns
        -------
        The learned representation 'self.emb_rec'.

        '''
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.weight_decay=weight_decay
        self.epochs=epochs
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.deconvolution = deconvolution
        self.datatype = datatype
        self.rad_cutoff=rad_cutoff
        self.k_dynamic=k_dynamic
        fix_seed(self.random_seed)
        self.radius=radius
        self.k_cl=k_cl
        self.model_select = model_select
        if 'highly_variable' not in adata.var.keys():
           preprocess(self.adata)

        if datatype not in ['Stereo','Slide','HD','Xenium']:
            if 'adj' not in adata.obsm.keys():
                construct_interaction(self.adata,radius=self.radius,neighborhood=self.k_cl)
                self.adj = self.adata.obsm['adj']
                self.graph_neigh = torch.FloatTensor(self.adata.obsm['graph_neigh'].copy() + np.eye(self.adj.shape[0])).to(self.device)
                
        if 'label_CSL' not in adata.obsm.keys():    
           add_contrastive_label(self.adata)
           
        if 'feat' not in adata.obsm.keys():
           get_feature(self.adata)

        Cal_Spatial_Net(self.adata, rad_cutoff=self.rad_cutoff, k_cutoff=self.k_dynamic, model = self.model_select)
        self.features = torch.FloatTensor(self.adata.obsm['feat'].copy()).to(self.device)
        self.features_a = torch.FloatTensor(self.adata.obsm['feat_a'].copy()).to(self.device)
        self.label_CSL = torch.FloatTensor(self.adata.obsm['label_CSL']).to(self.device)
    
        self.dim_input = self.features.shape[1]
        self.dim_output = dim_output
        
        if self.datatype in ['Stereo', 'Slide','HD','Xenium']:
           if self.datatype == 'Xenium':
               self.dim_output = 32
           print('No adj in high throughout')
        else: 
           # standard version
           self.adj = preprocess_adj(self.adj)
           self.adj = torch.FloatTensor(self.adj).to(self.device)
           
    def batch_train(self,batch_number=18,verbose=False): 

        if 'highly_variable' in self.adata.var.columns:
            adata_Vars = self.adata[:, self.adata.var['highly_variable']]
        else:
            adata_Vars = self.adata
        self.batch_number = batch_number
        self.adata.X = csr_matrix(self.adata.X)
        self.n_sample = self.features.shape[0]
        self.per_num_inbatch = int(math.ceil(1.0*self.n_sample/self.batch_number))

        # calculate_datalist
        data_list = []
        adj_list = []

        for batch_idx in range(self.batch_number):
            print('Batch : ' + str(batch_idx) + ' Preprocessing')
            idx_beg = batch_idx*self.per_num_inbatch
            idx_end = min((batch_idx+1)*self.per_num_inbatch, self.features.shape[0])
            adata_subset = adata_Vars[idx_beg:idx_end]
            
            Cal_Spatial_Net(adata_subset, rad_cutoff=self.rad_cutoff, k_cutoff=self.k_dynamic, model = self.model_select,verbose=False)
            data_in_batch = Transfer_pytorch_Data(adata_subset)
            data_list.append(data_in_batch)

            adj_batch = construct_interaction(adata_subset,radius=self.radius,neighborhood=self.k_cl)
            adj_batch = torch.FloatTensor(adj_batch)
            adj_list.append(adj_batch)
            print('Over!')

        graph_neigh_list = adj_list
        if 'Spatial_Net' not in self.adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

        self.model = Batch_Dynamic_Atten_Autoencoder(self.dim_input, self.dim_output).to(self.device)
        self.loss_CSL = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                         weight_decay=self.weight_decay)
        print('Begin to train ST data...')
        
        self.model.train()
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
            total_loss = 0  # Accumulate total loss over batches for logging
            for batch_idx in range(self.batch_number):
                # define begin and end index
                idx_beg = batch_idx*self.per_num_inbatch
                idx_end = min((batch_idx+1)*self.per_num_inbatch, self.features.shape[0])
                
                features_batch = self.features[idx_beg:idx_end]
                features_a_batch = self.features_a[idx_beg:idx_end]

                graph_neigh_batch = graph_neigh_list[batch_idx].to(self.device)
                data_batch = data_list[batch_idx].to(self.device)
                adj_batch = adj_list[batch_idx].to(self.device)

                # Forward pass for the batch
                hiden_feat, emb, ret, ret_a = self.model(features_batch, features_a_batch, graph_neigh_batch, data_batch, adj_batch)
                
                # Compute batch losses
                loss_sl_1 = self.loss_CSL(ret, self.label_CSL[idx_beg:idx_end])
                loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL[idx_beg:idx_end])
                loss_feat = F.mse_loss(features_batch, emb)
                
                # Combine losses
                batch_loss = self.alpha * loss_feat + self.beta * (loss_sl_1 + loss_sl_2)
                
                # Backward pass and parameter update
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                # Accumulate total loss for logging
                total_loss += batch_loss.item()
            
            # Log progress
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Overall Loss: {total_loss:.4f}')
                
        print("Optimization finished for ST data!")
        
        with torch.no_grad():
            self.model.eval()
            if self.datatype in ['Stereo', 'Slide']:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
                #self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().numpy()
            else:
                data = Transfer_pytorch_Data(adata_Vars) # include data.x, data.edge_index
                data = data.to(self.device)
                self.emb_rec = self.model.predict(self.features, self.features_a, data).detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec
            
            return self.adata
        
    def train_expand(self,batch_number=18,verbose=False): 

        if 'highly_variable' in self.adata.var.columns:
            adata_Vars = self.adata[:, self.adata.var['highly_variable']]
        else:
            adata_Vars = self.adata
        self.batch_number = batch_number
        self.adata.X = csr_matrix(self.adata.X)
        self.n_sample = self.features.shape[0]
        self.per_num_inbatch = int(math.ceil(1.0*self.n_sample/self.batch_number))
        # self.data = Transfer_pytorch_Data(adata_Vars).to(self.device) # include data.x, data.edge_index
        # data = self.data

        # calculate_datalist
        data_list = []
        adj_list = []

        for batch_idx in range(self.batch_number):
            print('Batch : ' + str(batch_idx) + ' Preprocessing')
            idx_beg = batch_idx*self.per_num_inbatch
            idx_end = min((batch_idx+1)*self.per_num_inbatch, self.features.shape[0])
            adata_subset = adata_Vars[idx_beg:idx_end]
            
            Cal_Spatial_Net(adata_subset, rad_cutoff=self.rad_cutoff, k_cutoff=self.k_dynamic, model = self.model_select,verbose=False)
            data_in_batch = Transfer_pytorch_Data(adata_subset).to(self.device)
            data_list.append(data_in_batch)

            adj_batch = construct_interaction(adata_subset,radius=self.radius,neighborhood=self.k_cl)
            adj_batch = torch.FloatTensor(adj_batch).to(self.device)
            adj_list.append(adj_batch)
            print('Over!')

        graph_neigh_list = adj_list
        if 'Spatial_Net' not in self.adata.uns.keys():
            raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

        self.data = Transfer_pytorch_Data(adata_Vars).to(self.device) # include data.x, data.edge_index
        self.model = Batch_Dynamic_Atten_Autoencoder(self.dim_input, self.dim_output).to(self.device)
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, 
                                         weight_decay=self.weight_decay)
        print('Begin to train ST data...')
        
        self.model.train()
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
            total_loss = 0  # Accumulate total loss over batches for logging
            for batch_idx in range(self.batch_number):
                # define begin and end index
                idx_beg = batch_idx*self.per_num_inbatch
                idx_end = min((batch_idx+1)*self.per_num_inbatch, self.features.shape[0])
                
                features_batch = self.features[idx_beg:idx_end]
                features_a_batch = self.features_a[idx_beg:idx_end]

                graph_neigh_batch = graph_neigh_list[batch_idx]
                data_batch = data_list[batch_idx]
                adj_batch = adj_list[batch_idx]

                # Forward pass for the batch
                hiden_feat, emb, ret, ret_a = self.model(features_batch, features_a_batch, graph_neigh_batch, data_batch, adj_batch)
                
                # Compute batch losses
                loss_sl_1 = self.loss_CSL(ret, self.label_CSL[idx_beg:idx_end])
                loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL[idx_beg:idx_end])
                loss_feat = F.mse_loss(features_batch, emb)
                
                # Combine losses
                batch_loss = self.alpha * loss_feat + self.beta * (loss_sl_1 + loss_sl_2)
                
                # Backward pass and parameter update
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()
                
                # Accumulate total loss for logging
                total_loss += batch_loss.item()
            
            # Log progress
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Overall Loss: {total_loss:.4f}')
                
        print("Optimization finished for ST data!")
        
        with torch.no_grad():
            self.model.eval()
            if self.datatype in ['Stereo', 'Slide']:
                self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
                self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
                #self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().numpy()
            else:
                data = Transfer_pytorch_Data(adata_Vars) # include data.x, data.edge_index
                data = data.to(self.device)
                self.emb_rec = self.model.predict(self.features, self.features_a, data).detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec
            return self.adata
        
    def transferHD(self,adata,verbose=False): 
        adata_transfer = adata.copy()
        Cal_Spatial_Net(adata_transfer, rad_cutoff=self.rad_cutoff, k_cutoff=self.k_dynamic, model = self.model_select,verbose=False)
        data = Transfer_pytorch_Data(adata_transfer).to(self.device)
        features = torch.FloatTensor(adata_transfer.obsm['feat']).to(self.device)
        features_a = torch.FloatTensor(adata_transfer.obsm['feat_a']).to(self.device)

        with torch.no_grad():
            self.model.eval()
            emb_rec = self.model.predict(features, features_a, data)

        adata_transfer.obsm['emb'] = emb_rec.detach().cpu().numpy()
        return adata_transfer

