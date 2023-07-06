import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
from collections import defaultdict
from utils import function as func

import torch
from torch_geometric.datasets import Planetoid
from model import logreg as logreg
from model import build_model
from torch_geometric.utils import remove_self_loops, add_self_loops
import scipy.sparse as sp
import numpy as np
from test import test

def main(config):
    best_t = 0
    bestacc = 0

    # Data
    path = osp.join('./datasets', args.dataset)
    dataset = Planetoid(path, args.dataset)
    nb_classes = dataset.num_classes
    data = dataset[0]

    devices = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(devices)

    edge = data.edge_index.clone()
    adj = sp.coo_matrix((np.ones(edge.shape[1]), (edge.cpu()[0, :], edge.cpu()[1, :])),
                        shape=(data.y.shape[0], data.y.shape[0]), dtype=np.float32).toarray()
    adj = torch.from_numpy(adj).cuda()

    idx_train, idx_val, idx_test = data.train_mask, data.val_mask, data.test_mask
    data.edge_index, _ = add_self_loops(data.edge_index)
    adj_lists = defaultdict(set)
    for i in range(data.edge_index.size(1)):
        adj_lists[data.edge_index[0][i].item()].add(data.edge_index[1][i].item())
    edge_index2, _ = remove_self_loops(data.edge_index)

    # Sub_graphs
    nodes_batch = torch.randint(0, data.num_nodes, (config['num_sub_g'],))
    node_neighbor_cen = func.sub_sam(nodes_batch, adj_lists, config['num_hop'])

    # Model
    model = build_model(dataset.num_features, config)
    model = model.to(devices)
    logic = logreg.LogReg(config['num_hidden'], nb_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    scheduler = lambda epoch: (1 + np.cos((epoch) * np.pi / config['num_epochs'])) * 0.5
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    start = t()
    prev = start

    for epoch in range(1, config['num_epochs'] + 1):
        model.train()
        optimizer.zero_grad()

        loss = model(data.x, data.edge_index, edge_index2, adj, adj_lists)
        loss.backward()
        optimizer.step()
        if config['use_scheduler']:
            scheduler.step()

        if epoch % 10 == 0:
            acc = test(model, data, idx_train, idx_val, idx_test, 5, logic)
            if acc > bestacc:
                bestacc = acc
                best_t = epoch
                torch.save(model.state_dict(), args.savepath)
            print(f'acc={acc}, epoch={epoch}, bestacc={bestacc}, bestepoch={best_t}')

        now = t()

        learning_rate = config['learning_rate']
        num = config['num']

        print(f'(T)|E={epoch:03d},l={loss:.4f},t={now - prev:.2f},all={now - start:.2f},'
              f'{num},{args.dataset},{learning_rate}')
        prev = now

    print("=== Final ===")
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(args.savepath))
    acc = test(model, data, idx_train, idx_val, idx_test, 50, logic)
    print(f'bestacc={bestacc}, bestepoch={best_t}')
    print(f'acc={acc}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--savepath', type=str, default='save/Cora.pkl')

    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)
    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]
    torch.manual_seed(config['seed'])
    random.seed(12345)

    print(config)

    main(config)



