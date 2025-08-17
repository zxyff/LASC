import os
import pandas as pd
import torch
import dgl
from dgl.data.utils import save_graphs, load_graphs
from tqdm import tqdm
from numba import njit, prange
import numpy as np

class DglDataset:
    def __init__(self, name='TC-mixer', data_path='/root/SIEG_OGB/data/', graph_path='graph_data.dgl', split_path='edge_split.pt',
                  train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2):
        self.data_path = data_path
        self.graph_path = graph_path
        self.split_path = split_path
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

        # Load or build the graph
        if os.path.exists(self.graph_path):
            self.graph, _ = load_graphs(self.graph_path)
            self.graph = self.graph[0]
            print("Graph loaded from saved file.")
        else:
            print("Saved graph not found. Building graph from CSV files...")
            self.build_graph_from_csv()
            self.save_graph()

        # Load or create edge split
        if os.path.exists(self.split_path):
            self.split_edge = self.load_split_edge()
            print("Edge split loaded from saved file.")
        else:
            print("Saved edge split not found. Creating edge split...")
            self.split_edge = self.get_edge_split()
            self.save_split_edge()

    def build_graph_from_csv(self):
        # 读取特征数据
        feature_df = pd.read_csv(f'{self.data_path}processed_128_feature.csv')

        train_pos_df = pd.read_csv(f'{self.data_path}label+1.csv')
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
                col_name = feature_df.columns[2+i]
                missing_data[col_name] = [0.0] * len(missing_nodes)
            feature_df = pd.concat([feature_df, pd.DataFrame(missing_data)], ignore_index=True)

        self.node_features = torch.tensor(
            feature_df.drop(columns=['node', 'nodeid']).values,
            dtype=torch.float32
        )

        # 构建节点映射
        self.node_mapping = {node: idx for idx, node in enumerate(feature_df['node'])}

        # 构建边列表
        src, dst = [], []
        missing_node_count = 0
        for _, row in train_pos_df.iterrows():
            node1 = row['node1']
            node2 = row['node2']

            if node1 not in self.node_mapping: 
                missing_node_count += 1
                continue
            if node2 not in self.node_mapping:
                missing_node_count += 1
                continue
                
            src.append(self.node_mapping[node1])
            dst.append(self.node_mapping[node2])

        if missing_node_count > 0:
            print(f"警告：仍有 {missing_node_count} 次节点查询失败，请检查合并逻辑")

        # 创建DGL图
        num_nodes = len(feature_df)
        self.graph = dgl.graph((torch.tensor(src), torch.tensor(dst)), num_nodes=num_nodes)
        self.graph.ndata['feat'] = self.node_features
        self.add_isolated_selfloops()

    def add_isolated_selfloops(self):
        num_nodes = self.graph.num_nodes()
        in_deg = self.graph.in_degrees()
        out_deg = self.graph.out_degrees()
        isolated_nodes = torch.where((in_deg == 0) & (out_deg == 0))[0]
        
        if len(isolated_nodes) > 0:
            self.graph.add_edges(isolated_nodes, isolated_nodes)

    def save_graph(self):
        save_graphs(self.graph_path, [self.graph])
        print(f"Graph saved to {self.graph_path}")

    def save_split_edge(self):
        torch.save(self.split_edge, self.split_path)
        print(f"Edge split saved to {self.split_path}")

    def load_split_edge(self):
        return torch.load(self.split_path)

    def get_edge_split(self):
        src, dst = self.graph.edges()
        self_loop_mask = src == dst
        normal_mask = ~self_loop_mask

        self_loop_edges = (src[self_loop_mask], dst[self_loop_mask])
        normal_edges = (src[normal_mask], dst[normal_mask])

        num_self_loops = len(self_loop_edges[0])
        num_normal_edges = len(normal_edges[0])

        num_train_loops = int(self.train_ratio * num_self_loops)
        num_valid_loops = int(self.valid_ratio * num_self_loops)
        num_test_loops = num_self_loops - num_train_loops - num_valid_loops

        num_train_normals = int(self.train_ratio * num_normal_edges)
        num_valid_normals = int(self.valid_ratio * num_normal_edges)
        num_test_normals = num_normal_edges - num_train_normals - num_valid_normals

        self_loop_perm = torch.randperm(num_self_loops)
        normal_perm = torch.randperm(num_normal_edges)

        train_self_loops = self_loop_perm[:num_train_loops]
        valid_self_loops = self_loop_perm[num_train_loops:num_train_loops + num_valid_loops]
        test_self_loops = self_loop_perm[num_train_loops + num_valid_loops:]

        train_normals = normal_perm[:num_train_normals]
        valid_normals = normal_perm[num_train_normals:num_train_normals + num_valid_normals]
        test_normals = normal_perm[num_train_normals + num_valid_normals:]

        train_src = torch.cat([self_loop_edges[0][train_self_loops], normal_edges[0][train_normals]])
        train_dst = torch.cat([self_loop_edges[1][train_self_loops], normal_edges[1][train_normals]])

        valid_src = torch.cat([self_loop_edges[0][valid_self_loops], normal_edges[0][valid_normals]])
        valid_dst = torch.cat([self_loop_edges[1][valid_self_loops], normal_edges[1][valid_normals]])

        test_src = torch.cat([self_loop_edges[0][test_self_loops], normal_edges[0][test_normals]])
        test_dst = torch.cat([self_loop_edges[1][test_self_loops], normal_edges[1][test_normals]])

        valid_neg_dst = self.numba_negative_sampling(valid_src.numpy(), num_samples=50, num_nodes=self.graph.num_nodes())
        test_neg_dst = self.numba_negative_sampling(test_src.numpy(), num_samples=50, num_nodes=self.graph.num_nodes())

        valid_neg_dst = torch.tensor(valid_neg_dst, dtype=torch.long)
        test_neg_dst = torch.tensor(test_neg_dst, dtype=torch.long)

        split_edge = {
            'train': {'source_node': train_src, 'target_node': train_dst},
            'valid': {'source_node': valid_src, 'target_node': valid_dst, 'target_node_neg': valid_neg_dst},
            'test': {'source_node': test_src, 'target_node': test_dst, 'target_node_neg': test_neg_dst}
        }

        return split_edge

    @staticmethod
    @njit(parallel=True)
    def numba_negative_sampling(src, num_samples, num_nodes):
        neg_dst = np.empty((len(src), num_samples), dtype=np.int64)
        for i in prange(len(src)):
            samples = set()
            while len(samples) < num_samples:
                candidate = np.random.randint(0, num_nodes, num_samples)
                for c in candidate:
                    if c != src[i] and c not in samples:
                        samples.add(c)
                    if len(samples) >= num_samples:
                        break
            neg_dst[i, :] = np.array(list(samples))
        return neg_dst

    def __getitem__(self, idx):
        assert idx == 0, "Only a single graph is available in this dataset."
        return self.graph

# Usage example
if __name__ == '__main__':
    dgl_dataset = DglDataset(name='TC-mixer')
    split_edge = dgl_dataset.split_edge
    print(dgl_dataset[0])

    print(split_edge['train'].keys())
    print(split_edge['valid'].keys())
    print(split_edge['test'].keys())
    print(split_edge['train']['source_node'].shape, split_edge['train']['target_node'].shape)
    print(split_edge['valid']['source_node'].shape, split_edge['valid']['target_node'].shape, split_edge['valid']['target_node_neg'].shape)
    print(split_edge['test']['source_node'].shape, split_edge['test']['target_node'].shape, split_edge['test']['target_node_neg'].shape)
