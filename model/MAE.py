import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
import numpy as np
from model.loss import adj_loss, sce_loss
from functools import reduce
from operator import add
from utils import function as func
import torch_geometric
import metis

# GNN
class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv):
        super(Encoder, self).__init__()
        self.conv = [base_model(in_channels, 2 * out_channels)]
        # self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)
        self.activation = activation
        self.dropout = 0.2
        self.norm = nn.LayerNorm(out_channels)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self.activation(self.conv[0](x, edge_index))
        # x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.activation(self.conv[1](x1, edge_index))
        # x = self.norm(x)

        return x2

# Generation
class Genetation(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation):
        super(Genetation, self).__init__()
        self.conv = GATConv(in_channels, out_channels, add_self_loops=False)
        self.activation = activation

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        x = self.activation(self.conv(x, edge_index))
        return x


# Model
class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder, e2d, encoder_to_decoder, conv, mask_rate: float = 0.5,
                 loss_weight: float = 0.5, num_sub_g: int = 10, num_hop: int = 5):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.decoder = decoder
        self.encoder_to_decoder = encoder_to_decoder
        self.conv = conv

        self.e2d = e2d
        self._mask_rate: float = mask_rate
        self.num_sub_g = num_sub_g
        self.num_hop = num_hop

        self.fea_criterion = sce_loss
        self.adj_criterion = adj_loss
        self.loss_weight: float = loss_weight

    def forward(self, x, edge_index, edge_index2, adj, adj_lists):
        # the whole graph
        # use_x, use_adj = self.drop_node_fea(x, edge_index, self._mask_rate)

        #baseline
        use_x, use_adj = x, edge_index
        enc_rep = self.encoder(use_x, use_adj)
        if self.e2d:
            enc_rep = self.encoder_to_decoder(enc_rep)
        recon = self.decoder(enc_rep, edge_index2)

        graph_loss = self.graph_loss(recon, adj, x)

        # perform graph partitioning
        partitioned_x, partitioned_edge_index = self.metis_partition(use_x, use_adj, self.num_sub_g)

        # process subgraphs
        for i in range(self.num_sub_g):
            sub_x = partitioned_x[i]
            sub_edge_index = partitioned_edge_index[i]

            sub_enc_rep = self.encoder(sub_x, sub_edge_index)
            if self.e2d:
                sub_enc_rep = self.encoder_to_decoder(sub_enc_rep)
            sub_recon = self.decoder(sub_enc_rep, edge_index2)

            sub_adj = torch.zeros((sub_x.shape[0], sub_x.shape[0]))
            sub_adj[sub_edge_index[0], sub_edge_index[1]] = 1
            sub_adj = sub_adj.to(x.device)

            sub_graph_loss = self.graph_loss(sub_recon, sub_adj, sub_x)

            graph_loss += sub_graph_loss

        return graph_loss

    def metis_partition(self, x, edge_index, num_sub_g):
        data = torch_geometric.data.Data(x=x, edge_index=edge_index)
        data = torch_geometric.utils.to_networkx(data)

        part = metis.part_graph(data, num_sub_g)
        partitioned_x = []
        partitioned_edge_index = []

        for i in range(num_sub_g):
            idx = torch.where(part == i)[0]
            partitioned_x.append(x[idx])
            mask = torch.isin(edge_index[0], idx) & torch.isin(edge_index[1], idx)
            partitioned_edge_index.append(edge_index[:, mask])

        return partitioned_x, partitioned_edge_index

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x, edge_index)
        return z

    def graph_loss(self, recon, adj, x):
        adj_rec = torch.sigmoid(torch.matmul(recon, recon.t()))

        fea_loss = self.fea_criterion(recon, x)
        adj_loss = self.adj_criterion(adj_rec, adj)

        return fea_loss*self.loss_weight + adj_loss*(1-self.loss_weight)

    def drop_node_fea(self, x, edge_index, mask_rate):
        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        # node feature masking
        out_x = x.clone()
        out_x[mask_nodes] = 0.0

        # node edge masking
        src = edge_index[0, :]
        dst = edge_index[1, :]
        # mask out-degree of nodes
        keep_ind = [torch.nonzero(src==obj).tolist() for obj in keep_nodes]
        keep_ind = reduce(add, reduce(add, keep_ind))

        nsrc = src[keep_ind]
        ndst = dst[keep_ind]

        out_edge_index = torch.stack((nsrc.squeeze(), ndst.squeeze()), dim=0)

        return out_x, out_edge_index.squeeze()
