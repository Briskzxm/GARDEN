from .preprocess import *
from .utils import build_nx_graph,build_tg_graph,get_rwr_matrix,generate_positive_pairs_spatial_and_feature
from .LGCN import run_LGCN
from .model import MLP,ReconDNN,FusedGWLoss,feature_reconstruct_loss
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
class GARDEN_Align():
    def __init__(self, 
        slice1,
        slice2,
        args,
        random_seed = 42
        ):
        self.slice1 = slice1
        self.slice2 = slice2
        self.ratio = args.ratio
        self.device = args.device
        self.K = args.K
        self.hidden_dim = args.hidden_dim
        self.out_dim = args.out_dim
        self.alpha = args.alpha
        self.gamma_p = args.gamma_p
        self.in_iter = args.in_iter
        self.out_iter = args.out_iter
        self.loss_weight = args.loss_weight
        self.lr = args.lr
        self.epochs = args.epochs
        self.mincells_ratio = args.mincells_ratio
        self.init_threshold_lambda = args.init_threshold_lambda
        self.seperate_process = False
        seed_everything(random_seed)

        # if 'Spatial_Net' not in self.slice1.uns:
        Cal_Spatial_Net(self.slice1, k_cutoff=min(self.slice1.shape[0],self.K), model='KNN')
        # if 'Spatial_Net' not in self.slice2.uns and not self.seperate_process:
        Cal_Spatial_Net(self.slice2, k_cutoff=min(self.slice2.shape[0],self.K), model='KNN')

        if self.seperate_process:
            print("Maybe you are doing sc and st integration")
            self.edges = []
            self.features = dual_svd([self.slice1,self.slice2], dim=10,device = args.device)
            print('Dual PCA over!')
            slice1.obsm['feat'] = self.features[0]
            slice2.obsm['feat'] = self.features[1]
            Cal_Feature_Net(self.slice2, k_cutoff = self.K)
            data_st = Transfer_pytorch_Data(self.slice1)
            data_sc = Transfer_pytorch_Data(self.slice2)  
            self.edges.append(data_st.edge_index)
            self.edges.append(data_sc.edge_index)
            adj1 = construct_interaction_KNN(self.slice1,n_neighbors=self.K)
            adj2 = construct_interaction_Feature(self.slice2,n_neighbors=self.K)
        else:
            if args.model=='RNA':
                self.edges, self.features = SVD_based_preprocess([slice1,slice2], dim= 15, mincells_ratio = self.mincells_ratio, device = args.device)
            elif args.model=='ATAC':
                self.edges, self.features = SVD_based_preprocess([slice1,slice2], dim= 300, mincells_ratio = self.mincells_ratio ,device = args.device)
            self.edge_index1 = self.edges[0].numpy().T
            self.edge_index2 = self.edges[1].numpy().T
            adj1 = construct_interaction_KNN(self.slice1,n_neighbors=self.K)
            adj2 = construct_interaction_KNN(self.slice2,n_neighbors=self.K)

        self.adj1 = torch.tensor(adj1, dtype=torch.float64).to(self.device)
        self.adj2 = torch.tensor(adj2, dtype=torch.float64).to(self.device)

        self.x1, self.x2 = self.features[0].cpu().numpy(),self.features[1].cpu().numpy()
        self.slice1.obsm['emb'], self.slice2.obsm['emb'] = self.features[0].cpu().numpy(),self.features[1].cpu().numpy()
        self.anchor_links = find_best_matching(slice1, slice2, k_list=[(2**i)*self.K for i in range(3)])

        self.anchor1_cross_slices, self.anchor2_cross_slices = self.anchor_links[:, 0], self.anchor_links[:, 1]
        self.G1,self.G2 = build_nx_graph(self.edge_index1, self.anchor1_cross_slices, self.x1), build_nx_graph(self.edge_index2, self.anchor2_cross_slices, self.x2)
        print('Random Walk with Restart')
        self.rwr1, self.rwr2 = get_rwr_matrix(self.G1, self.G2, self.anchor_links, device=args.device, dtype=torch.float64) #dtype=torch.float64
        self.x1 = torch.FloatTensor(np.concatenate([self.x1, self.rwr1], axis=1))
        self.x2 = torch.FloatTensor(np.concatenate([self.x2, self.rwr2], axis=1))
        # self.x1, self.x2, _ = run_LGCN([self.x1, self.x2],self.edges,LGCN_layer=1)

        self.edges[0] =  self.edges[0].to(self.device)
        self.edges[1] =  self.edges[1].to(self.device)

        self.anchor_list_slice1 = generate_positive_pairs_spatial_and_feature(slice1,k_spatial=6)
        self.anchor_list_slice2 = generate_positive_pairs_spatial_and_feature(slice2,k_spatial=6)
        self.anchor1_slice1, self.anchor2_slice1 = self.anchor_list_slice1[:, 0], self.anchor_list_slice1[:, 1]
        self.anchor1_slice2, self.anchor2_slice2 = self.anchor_list_slice2[:, 0], self.anchor_list_slice2[:, 1]

    def train(self, epochs = 50):
        torch.set_default_dtype(torch.float64)
        self.G1_tg = build_tg_graph(self.edge_index1, self.x1.cpu().numpy(), self.rwr1, dtype=torch.float64).to(self.device)
        self.G2_tg = build_tg_graph(self.edge_index2, self.x2.cpu().numpy(), self.rwr2, dtype=torch.float64).to(self.device)

        n1, n2 = self.G1_tg.x.shape[0], self.G2_tg.x.shape[0]
        self.gw_weight = self.alpha / (1 - self.alpha) * min(n1, n2) ** 0.5
        self.model = MLP(input_dim=self.G1_tg.x.shape[1],
            hidden_dim=self.hidden_dim,
            output_dim=self.out_dim).to(self.device)
        self.recon_model0 = ReconDNN(self.out_dim, self.G1_tg.x.shape[1]).to(self.device)
        self.recon_model1 = ReconDNN(self.out_dim, self.G2_tg.x.shape[1]).to(self.device)
        self.optimizer_mlp = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        self.optimizer_recon0 = torch.optim.Adam(self.recon_model0.parameters(), lr=0.001, weight_decay=5e-4)
        self.optimizer_recon1 = torch.optim.Adam(self.recon_model1.parameters(), lr=0.001, weight_decay=5e-4)
        self.criterion = FusedGWLoss(self.G1_tg, self.G2_tg, self.anchor1_cross_slices, self.anchor2_cross_slices,
                                gw_weight=self.gw_weight,
                                gamma_p=self.gamma_p,
                                init_threshold_lambda=self.init_threshold_lambda,
                                in_iter=self.in_iter,
                                out_iter=self.out_iter,
                                total_epochs=self.epochs).to(self.device)
        self.triplet_loss = torch.nn.TripletMarginLoss(margin=1, p=2, reduction='mean')

        
    # 初始化混合精度训练的 GradScaler
        scaler = GradScaler()

        # 训练循环
        for epoch in tqdm(range(epochs)):
            self.model.train()
            self.optimizer_mlp.zero_grad()
            self.optimizer_recon0.zero_grad()
            self.optimizer_recon1.zero_grad()

            # 使用混合精度训练
            with autocast():
                # out1 = self.model(self.G1_tg.x, self.edges[0])
                # out2 = self.model(self.G2_tg.x,self.edges[1])
                out1, out2 = self.model(self.G1_tg, self.G2_tg)
                rec_1, rec_2 = self.recon_model0(out1), self.recon_model1(out2)
                loss = self.compute_loss(out1, out2, self.anchor1_cross_slices, self.anchor2_cross_slices, self.G1_tg, self.G2_tg, self.recon_model0, self.recon_model1, self.adj1, self.adj2, self.criterion, self.triplet_loss)

            # 反向传播和优化
            scaler.scale(loss).backward()
            scaler.step(self.optimizer_mlp)
            scaler.step(self.optimizer_recon0)
            # scaler.step(self.optimizer_recon1)
            scaler.update()

        with torch.no_grad():
            self.model.eval()
            out1, out2 = self.model(self.G1_tg, self.G2_tg)
            loss, similarity, threshold_lambda = self.criterion(out1=out1, out2=out2)
            out1 = out1.cpu().detach().numpy()
            out2 = out2.cpu().detach().numpy()
            similarity = similarity.cpu().detach().numpy().T
            return out1,out2,similarity
        
    def compute_loss(self, out1, out2, anchor1, anchor2, G1_tg, G2_tg, recon_model0, recon_model1, adj1, adj2, criterion, triplet_loss):
        # 计算三元组损失
        self.anchor1_slice1, self.anchor2_slice1 = self.anchor_list_slice1[:, 0], self.anchor_list_slice1[:, 1]
        self.anchor1_slice2, self.anchor2_slice2 = self.anchor_list_slice2[:, 0], self.anchor_list_slice2[:, 1]

        # 跨切片 triplet loss
        anchor_embeddings = out1[self.anchor1_cross_slices]
        positive_embeddings = out2[self.anchor2_cross_slices]
        negative_indices = torch.randint(low=0, high=out2.shape[0], size=(self.anchor1_cross_slices.shape[0],)).to(self.device)
        negative_embeddings = out2[negative_indices]
        tri_output_cross = triplet_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

        # 同一切片 triplet loss（slice1）
        anchor_embeddings_slice1 = out1[self.anchor1_slice1]
        positive_embeddings_slice1 = out1[self.anchor2_slice1]
        negative_indices_slice1 = torch.randint(low=0, high=out1.shape[0], size=(self.anchor1_slice1.shape[0],)).to(self.device)
        negative_embeddings_slice1 = out1[negative_indices_slice1]
        tri_output_slice1 = triplet_loss(anchor_embeddings_slice1, positive_embeddings_slice1, negative_embeddings_slice1)

        # 同一切片 triplet loss（slice2）
        anchor_embeddings_slice2 = out2[self.anchor1_slice2]
        positive_embeddings_slice2 = out2[self.anchor2_slice2]
        negative_indices_slice2 = torch.randint(low=0, high=out2.shape[0], size=(self.anchor1_slice2.shape[0],)).to(self.device)
        negative_embeddings_slice2 = out2[negative_indices_slice2]
        tri_output_slice2 = triplet_loss(anchor_embeddings_slice2, positive_embeddings_slice2, negative_embeddings_slice2)

        # 总损失
        # tri_output_loss = 0.6*tri_output_cross + 0.2*(tri_output_slice1 + tri_output_slice2)
        tri_output_loss = tri_output_cross + tri_output_slice1 + tri_output_slice2

        # 计算主损失
        # loss_fgw, similarity, threshold_lambda = criterion(out1=out1, out2=out2)

        # 计算重构损失
        recon_adj1 = torch.sigmoid(torch.matmul(out1, out1.T))
        recon_adj2 = torch.sigmoid(torch.matmul(out2, out2.T))
        loss_ReconAdj = torch.sum(F.binary_cross_entropy_with_logits(recon_adj1, adj1)) +torch.sum( F.binary_cross_entropy_with_logits(recon_adj2, adj2))
        # loss_ReconAdj = 0
        # 计算特征重构损失
        loss_recon1 = feature_reconstruct_loss(out1, G1_tg.x, recon_model0)
        loss_recon2 = feature_reconstruct_loss(out2, G2_tg.x, recon_model0)

        loss_recon = loss_recon1 + loss_recon2
    
        total_loss = (
            1 * loss_recon +
            0 * loss_ReconAdj +
            0.1 * tri_output_loss
        )

        return total_loss







        
    

