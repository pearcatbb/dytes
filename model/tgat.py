import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as scatter

from modules.base_gnn import GCN, GAT
from modules.utils import MergeLayer_output, Feat_Process_Layer, drop_edge, MergeLayer_gnn
from modules.embedding_module import get_embedding_module
from modules.time_encoding import TimeEncode
from model.gdn import graph_deviation_network
from model.supconloss import SupConLoss


class TGAT(torch.nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.cfg = config

        self.nodes_dim = self.cfg.input_dim
        self.edge_dim = self.cfg.edge_dim
        self.dims = self.cfg.hidden_dim

        self.n_heads = self.cfg.n_heads
        self.dropout = self.cfg.drop_out
        self.n_layers = self.cfg.n_layer

        self.mode = self.cfg.mode

        self.time_encoder = TimeEncode(dimension=self.dims)
        self.embedding_module_type = self.cfg.module_type
        self.embedding_module = get_embedding_module(module_type=self.embedding_module_type,
                                                     time_encoder=self.time_encoder,
                                                     n_layers=self.n_layers,
                                                     node_features_dims=self.dims,
                                                     edge_features_dims=self.dims,
                                                     time_features_dim=self.dims,
                                                     hidden_dim=self.dims,
                                                     n_heads=self.n_heads, dropout=self.dropout)

        self.node_preocess_fn = Feat_Process_Layer(self.nodes_dim, self.dims)
        self.edge_preocess_fn = Feat_Process_Layer(self.edge_dim, self.dims)
        self.type_transfer = torch.nn.ModuleList([nn.LSTM(self.dims, int(self.dims / 2), 1, bidirectional = True) for _ in range(config.type_num)])

        self.affinity_score = MergeLayer_output(self.dims * 2, self.dims, drop_out=0.2)
        # self.affinity_score = MergeLayer_output(self.dims + 2, self.dims, drop_out=0.2)
        # self.affinity_score = MergeLayer_output(self.dims, self.dims // 2, drop_out=0.2)
        self.classify = nn.Linear(self.dims * 2, 2)
        # self.classify = nn.Linear(self.dims + 2, 2)
        # self.classify = nn.Linear(self.dims, 2)

        self.device = device

        self.gdn = graph_deviation_network(self.cfg, device)
        self.suploss = SupConLoss()

        self.all_nodes_features = nn.Parameter(torch.zeros([self.cfg.node_num, self.dims]), requires_grad=False)
        self.gnn_transfer = MergeLayer_gnn(self.dims, self.dims, drop_out=0.2)
        self.struct_embedding = GAT(self.dims, self.dims, self.dims)


    def forward(self, src_org_edge_feat, src_edge_to_time, src_center_node_idx, src_neigh_edge, src_node_features,
                current_time, src_center_node_idx_ori, src_neigh_edge_ori, label):
        # apply tgat

        src_node_type = torch.zeros(src_center_node_idx.shape[0])
        for i in range(self.cfg.type_num):
            idx = torch.where(src_center_node_idx >= self.cfg.range[i])[0]
            if len(idx) > 0:
                src_node_type[idx] = i + 1
        source_node_embedding = self.compute_temporal_embeddings(src_neigh_edge, src_edge_to_time,
                                                                                src_org_edge_feat, src_node_features, src_node_type)

        root_embedding = source_node_embedding[src_center_node_idx, :]
        gnn_transfer = self.gnn_transfer(root_embedding)
        self.all_nodes_features[src_center_node_idx_ori] = gnn_transfer.clone().detach()
        struct_embeddings = self.compute_struct_embeddings(src_center_node_idx_ori, src_neigh_edge_ori)
        root_embedding = torch.cat([root_embedding, struct_embeddings], dim=1)
        anom_score = self.gdn(root_embedding, current_time, label)  # 异常检测
        dev, group = self.gdn.dev_diff(torch.squeeze(anom_score), current_time, label)

        # 节点分类
        logits = self.affinity_score(root_embedding)
        clss = self.classify(root_embedding)
        # mapping = dict(zip(src_center_node_idx.tolist(), src_center_node_idx_ori.tolist()))
        # 样本增强
        aug_neigh_edge, aug_edge_to_time, aug_org_edge_feat, pickup_ids = drop_edge(src_neigh_edge,src_edge_to_time,src_org_edge_feat, 0.2)
        aug_node_embedding = self.compute_temporal_embeddings(aug_neigh_edge, aug_edge_to_time,
                                                                                aug_org_edge_feat, src_node_features, src_node_type)
        aug_node_embedding = aug_node_embedding[src_center_node_idx, :]

        aug_struct_embeddings = self.compute_struct_embeddings(src_center_node_idx_ori, src_neigh_edge_ori[pickup_ids])
        aug_node_embedding = torch.cat([aug_node_embedding, aug_struct_embeddings], dim=1)
        prediction_dict = {}
        prediction_dict['logits'] = torch.reshape(logits, [-1])
        prediction_dict['cls'] = clss.clone().detach().reshape((-1, 2))
        prediction_dict['anom_score'] = anom_score
        prediction_dict['time'] = current_time
        prediction_dict['root_embedding'] = torch.cat([root_embedding, aug_node_embedding], dim=0)
        prediction_dict['group'] = group.clone().detach().repeat(2, 1)
        prediction_dict['dev'] = dev.clone().detach().repeat(2, 1)
        
        return prediction_dict


    def compute_temporal_embeddings(self, neigh_edge, edge_to_time, edge_feat, node_feat, node_type):
        node_feat = self.node_preocess_fn(node_feat)

        ## 不同类型的节点处理
        ## 分别使用BiLSTM进行映射
        new_node_feat = torch.zeros([node_feat.shape[0], self.dims], requires_grad=True).to(self.device)
        max_type = int(torch.max(node_type))
        for i in range(max_type):
            idx = torch.where(node_type == i)[0].to(self.device)
            if(len(idx) > 0):
                new_node_feat[idx] = self.type_transfer[i](node_feat[idx])[0]
                # new_node_feat[idx] = node_feat[idx]

        edge_feat = self.edge_preocess_fn(edge_feat)

        node_embedding = self.embedding_module.compute_embedding(neigh_edge, edge_to_time,
                                                                 edge_feat, node_feat)

        return node_embedding

    def compute_struct_embeddings(self, src_center_node_idx, src_neigh_edge):
        adj = torch.zeros([self.cfg.node_num, self.cfg.node_num]).to(self.device)
        adj[src_neigh_edge[:, 0], src_neigh_edge[:, 1]] = 1

        # out = self.struct_embedding(adj, self.all_nodes_features)
        out = self.struct_embedding(self.all_nodes_features, adj)
        struct_embeddings = out[src_center_node_idx, :]
        return struct_embeddings
