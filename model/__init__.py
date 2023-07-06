from model.MAE import Encoder, Genetation, Model
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv


def build_model(num_features, config):

    num_hidden = config['num_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'gelu': nn.GELU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv, 'GATConv': GATConv})[config['base_model']]

    mask_rate = config['mask_rate']
    loss_weight = config['loss_weight']
    e2d = config['e2d']
    num_sub_g = config['num_sub_g']
    num_hop = config['num_hop']

    encoder = Encoder(num_features, num_hidden, activation, base_model=base_model)
    encoder_to_decoder = nn.Linear(num_hidden, num_hidden, bias=False)
    conv = nn.Linear(2*num_hidden, num_hidden, bias=False)
    decoder = Genetation(num_hidden, num_features, activation)
    model = Model(encoder, decoder, e2d, encoder_to_decoder, conv, mask_rate, loss_weight, num_sub_g, num_hop)

    return model