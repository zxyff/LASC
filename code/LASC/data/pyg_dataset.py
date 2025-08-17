import torch
import os
import numpy as np
from tqdm import tqdm

class pyg_dataset:
    def __init__(self, name='TC-mixer', root='data', train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2):
        # 加载图数据
        self.data = torch.load(f"/root/SIEG_OGB/data/pyg_graph.pt")
        self.edge_index = self.data.edge_index
        self.num_nodes = self.data.num_nodes
        self.num_edges = self.edge_index.shape[1]

        # 设定数据划分比例
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio

        # 边划分文件路径
        self.edge_split_file = f"/root/SIEG_OGB/data/edge_split_pyg.pt"

        # 加载或生成边划分
        self.split_edge = self.load_or_create_edge_split()

    def load_or_create_edge_split(self):
        if os.path.exists(self.edge_split_file):
            print(f"Loading edge split from {self.edge_split_file}")
            split_edge = torch.load(self.edge_split_file)
        else:
            print(f"Edge split file not found, generating split...")
            split_edge = self.get_edge_split()
            torch.save(split_edge, self.edge_split_file)
        return split_edge

    def get_edge_split(self):

        self_loop_mask = self.edge_index[0] == self.edge_index[1]
        self.self_loop_edges = self.edge_index[:, self_loop_mask]
        self.normal_edges = self.edge_index[:, ~self_loop_mask]

        perm_normal = torch.randperm(self.normal_edges.shape[1])
        num_train_normal = int(self.train_ratio * self.normal_edges.shape[1])
        num_valid_normal = int(self.valid_ratio * self.normal_edges.shape[1])
        num_test_normal = self.normal_edges.shape[1] - num_train_normal - num_valid_normal

        train_normal = self.normal_edges[:, perm_normal[:num_train_normal]]
        valid_normal = self.normal_edges[:, perm_normal[num_train_normal:num_train_normal + num_valid_normal]]
        test_normal = self.normal_edges[:, perm_normal[num_train_normal + num_valid_normal:]]

        perm_self_loop = torch.randperm(self.self_loop_edges.shape[1])
        num_train_self_loop = int(self.train_ratio * self.self_loop_edges.shape[1])
        num_valid_self_loop = int(self.valid_ratio * self.self_loop_edges.shape[1])
        num_test_self_loop = self.self_loop_edges.shape[1] - num_train_self_loop - num_valid_self_loop

        train_self_loop = self.self_loop_edges[:, perm_self_loop[:num_train_self_loop]]
        valid_self_loop = self.self_loop_edges[:, perm_self_loop[num_train_self_loop:num_train_self_loop + num_valid_self_loop]]
        test_self_loop = self.self_loop_edges[:, perm_self_loop[num_train_self_loop + num_valid_self_loop:]]

        train_edges = torch.cat([train_normal, train_self_loop], dim=1)
        valid_edges = torch.cat([valid_normal, valid_self_loop], dim=1)
        test_edges = torch.cat([test_normal, test_self_loop], dim=1)

        self.edge_set = set((self.edge_index[0, i].item(), self.edge_index[1, i].item()) for i in range(self.num_edges))

        # 生成验证和测试集的负样本
        valid_neg_edges = self._negative_sampling(valid_edges[0], 50)
        test_neg_edges = self._negative_sampling(test_edges[0], 50)

        # 按照原格式保存分割结果
        split_edge = {
            'train': {'source_node': train_edges[0], 'target_node': train_edges[1]},
            'valid': {'source_node': valid_edges[0], 'target_node': valid_edges[1], 'target_node_neg': valid_neg_edges},
            'test': {'source_node': test_edges[0], 'target_node': test_edges[1], 'target_node_neg': test_neg_edges}
        }
        return split_edge

    def _negative_sampling(self, source_nodes, num_neg_samples):
        """生成负样本"""
        neg_edges = []
        for src in tqdm(source_nodes, desc="Generating negative samples"):
            neg_samples = []
            while len(neg_samples) < num_neg_samples:
                dst_candidates = torch.randint(0, self.num_nodes, (num_neg_samples - len(neg_samples),))
                for dst in dst_candidates:
                    if (src.item(), dst.item()) not in self.edge_set:
                        neg_samples.append(dst.item())
                        if len(neg_samples) == num_neg_samples:
                            break
            neg_edges.append(neg_samples)
        return torch.tensor(neg_edges)

    def __getitem__(self, idx):
        assert idx == 0, "Only a single graph is available in this dataset."
        return self.data

# 使用示例
if __name__ == '__main__':
    pyg_dataset = pyg_dataset(name='TC-mixer', root='data')
    
    # 打印
    print(pyg_dataset[0])
    print(pyg_dataset.split_edge['train'].keys())
    print(pyg_dataset.split_edge['valid'].keys())
    print(pyg_dataset.split_edge['test'].keys())
    print(pyg_dataset.split_edge['train']['source_node'].shape, pyg_dataset.split_edge['train']['target_node'].shape)
    print(pyg_dataset.split_edge['valid']['source_node'].shape, pyg_dataset.split_edge['valid']['target_node'].shape, pyg_dataset.split_edge['valid']['target_node_neg'].shape)
    print(pyg_dataset.split_edge['test']['source_node'].shape, pyg_dataset.split_edge['test']['target_node'].shape, pyg_dataset.split_edge['test']['target_node_neg'].shape)
