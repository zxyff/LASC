import torch
import numpy as np
from tqdm import tqdm
train_ratio = 0.6
valid_ratio = 0.2
test_ratio = 0.2

data = torch.load("/root/SIEG_OGB/data/pyg_graph.pt")

edge_index = data.edge_index
num_edges = edge_index.shape[1]

np.random.seed(42)

perm = torch.randperm(num_edges)

num_train = int(train_ratio * num_edges)
num_valid = int(valid_ratio * num_edges)
num_test = num_edges - num_train - num_valid

train_edges = edge_index[:, perm[:num_train]]
valid_edges = edge_index[:, perm[num_train:num_train + num_valid]]
test_edges = edge_index[:, perm[num_train + num_valid:]]

edge_set = set((edge_index[0, i].item(), edge_index[1, i].item()) for i in range(num_edges))

def negative_sampling(edge_index, source_nodes, num_neg_samples, num_nodes):
    neg_edges = []
    for src in tqdm(source_nodes):
        neg_samples = []
        while len(neg_samples) < num_neg_samples:
            dst_candidates = torch.randint(0, num_nodes, (num_neg_samples - len(neg_samples),))
            for dst in dst_candidates:
                if (src.item(), dst.item()) not in edge_set:
                    neg_samples.append(dst.item())
                    if len(neg_samples) == num_neg_samples:
                        break
        neg_edges.append(neg_samples)
    return torch.tensor(neg_edges)
num_nodes = data.num_nodes

valid_neg_edges = negative_sampling(edge_index, valid_edges[0], 500, num_nodes)
test_neg_edges = negative_sampling(edge_index, test_edges[0], 500, num_nodes)

split_edge = {
    'train': {'source_node': train_edges[0], 'target_node': train_edges[1]},
    'valid': {
        'source_node': valid_edges[0],
        'target_node': valid_edges[1],
        'target_node_neg': valid_neg_edges
    },
    'test': {
        'source_node': test_edges[0],
        'target_node': test_edges[1],
        'target_node_neg': test_neg_edges
    }
}

print(f"训练集： {split_edge['train']['source_node'].shape} {split_edge['train']['target_node'].shape}")
print(f"验证集： {split_edge['valid']['source_node'].shape} {split_edge['valid']['target_node'].shape} {split_edge['valid']['target_node_neg'].shape}")
print(f"测试集： {split_edge['test']['source_node'].shape} {split_edge['test']['target_node'].shape} {split_edge['test']['target_node_neg'].shape}")