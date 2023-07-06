import torch
import torch_geometric
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import numpy as np
from pymetis import part_graph

def metis_partition(dataset, num_partitions):
    # 加载数据集
    dataset = dataset.transform(T.NormalizeFeatures())
    data = dataset[0]

    # 获取节点数和边数
    num_nodes = data.num_nodes
    num_edges = data.num_edges

    # 创建节点和边的索引
    edge_index = data.edge_index.cpu().numpy()
    x = data.x.cpu().numpy()

    # 使用pymetis库执行Metis分区算法
    _, part = part_graph(num_nodes, num_edges, edge_index[0], edge_index[1], np.arange(num_nodes), num_partitions)

    # 将分区结果转换为PyTorch张量
    part = torch.tensor(part)

    # 将分区结果作为节点特征添加到数据中
    data.x = torch.cat([data.x, part.unsqueeze(1).float()], dim=1)

    return data

# 使用示例
dataset = Planetoid(root='datasets/Cora', name='Cora')
num_partitions = 4

partitioned_data = metis_partition(dataset, num_partitions)
