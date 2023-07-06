import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, GATConv
from functools import reduce
from operator import add


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
        # x = F.dropout(x, p=self.dropout, training=self.training)
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



class SRM(torch.nn.Module):
    def __init__(self, num_features, num_classes, activation=F.relu):
        super(SRM, self).__init__()
        self.encoder = Encoder(num_features, 64, activation)
        self.generation = Genetation(64, num_features, activation)
        self.classifier = torch.nn.Linear(num_features, num_classes)
        self._mask_rate = 0.15
        self.e2d = False
        self.encoder_to_decoder = nn.Linear(num_features, num_features, bias=False)

    def forward(self, x, edge_index):
        # Dropout 20% of the node features
        x, edge_index = self.drop_node_fea(x, edge_index, self._mask_rate)

        x = self.encoder(x, edge_index)
        if self.e2d:
            x = self.encoder_to_decoder(x)
        z = self.generation(x, edge_index)
        x_recon = torch.sigmoid(z)
        # x_recon = F.dropout(x_recon, p=0.2, training=self.training)
        x = self.classifier(x_recon)
        return F.log_softmax(x, dim=1)


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