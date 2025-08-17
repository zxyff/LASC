
import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data

# 读取特征
feature_df = pd.read_csv('dataset/GNNdataset/processed_feature.csv')

# 读取边数据及节点
train_pos_df = pd.read_csv('dataset/GNNdataset/label.csv')
all_edge_nodes = set(train_pos_df['node1']).union(set(train_pos_df['node2']))

existing_nodes = set(feature_df['node'])
missing_nodes = all_edge_nodes - existing_nodes

if missing_nodes:
    num_features = len(feature_df.columns) - 2
    missing_data = {
        'node': list(missing_nodes),
        'nodeid': [-1] * len(missing_nodes)
    }
    for i in range(num_features):
        missing_data[f'feature_{i}'] = [0.0] * len(missing_nodes)
    feature_df = pd.concat([feature_df, pd.DataFrame(missing_data)], ignore_index=True)

node_features = feature_df.drop(columns=['node', 'nodeid']).values
node_features = torch.tensor(node_features, dtype=torch.float32)
num_nodes = node_features.shape[0]

node_mapping = {node: idx for idx, node in enumerate(feature_df['node'])}

edge_index = []
for _, row in train_pos_df.iterrows():
    node1 = row['node1']
    node2 = row['node2']
    edge_index.append([node_mapping[node1], node_mapping[node2]])

edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

isolated_nodes = set(range(num_nodes)) - set(edge_index[0].tolist() + edge_index[1].tolist())
for node in isolated_nodes:
    edge_index = torch.cat([edge_index, torch.tensor([[node], [node]], dtype=torch.long)], dim=1)

data = Data(x=node_features, edge_index=edge_index, num_nodes=num_nodes)
torch.save(data, 'dataset/GNNdataset/pyg_graph.pt')

print(f"总节点数: {num_nodes} ")
print(f"边数: {len(train_pos_df)}")
print(f"最终边数: {data.edge_index.shape[1]}")