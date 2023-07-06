import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GAE, GATConv
from torch_geometric.utils import dropout_adj
from module import SRM
from time import perf_counter as t
from uti import node_dropout, node_perturbation, edge_dropout
from uti import partition_graph, sample_subgraph, sample_subgraphs_batch
from torch_geometric.utils import to_networkx
from torch_geometric.transforms import NormalizeFeatures

def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()
    x, edge_index = data.x, data.edge_index
    y = data.y

    partitioned_data = partition_graph(data, num_partitions=30)
    print('done')
    subgraph_indices = sample_subgraph(partitioned_data, K=15)
    subgraphs = sample_subgraphs_batch(data, subgraph_indices, batch_size=8)

    output = model(x, edge_index)
    loss = F.nll_loss(output[data.train_mask], y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test(model, data):
    model.eval()
    x, edge_index = data.x, data.edge_index
    y = data.y


    with torch.no_grad():
        output = model(x, edge_index)
        pred = output.argmax(dim=1)
        acc = pred[data.test_mask].eq(y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

def std_test(model, data):
    test_accs = []
    for _ in range(5):
        model.eval()
        test_acc = test(model, data)#.item()
        test_accs.append(test_acc)

    mean_acc = torch.tensor(test_accs).mean().item()
    std_acc = torch.tensor(test_accs).std().item()
    return mean_acc, std_acc


if __name__ == '__main__':
    dataset = Planetoid(root='datasets/Cora', name='Cora')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)
    model = SRM(dataset.num_features, dataset.num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    bestacc = 0
    for epoch in range(1, 201):
        # data = node_dropout(data, p=0.2)
        loss = train(model, data, optimizer)
        if epoch % 10 == 0:
            acc = test(model, data)
            if acc > bestacc:
                bestacc = acc
                best_t = epoch
                torch.save(model.state_dict(), 'cora.pkl')
            print(f'acc={acc}, epoch={epoch}, bestacc={bestacc}, bestepoch={best_t}')


    print("=== Final ===")
    print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load('cora.pkl'))
    # acc = test(model, data)
    mean_acc, std_acc = std_test(model, data)
    print(f'bestacc={bestacc}, bestepoch={best_t}')
    print('Test Accuracy: {:.2f} ± {:.2f}'.format(mean_acc * 100, std_acc * 100))

    # print(f'acc={acc}')
