from torch.utils.data import Dataset
import ngnn_utils
import numpy as np
import torch
import os
import dgl
from dgl.data.utils import load_graphs, save_graphs
from tqdm import tqdm


model='NGNNDGCNNGraphormer_noNeigFeat'
# ngnn dataset
class SEALOGBLDataset(Dataset):
    def __init__(
        self,
        data_pyg,
        preprocess_fn,
        root,
        graph,
        split_edge,
        percent=100,
        split="train",
        ratio_per_hop=1.0,
        directed=False,
        dynamic=True,
    ) -> None:
        super().__init__()
        self.data_pyg = data_pyg
        self.preprocess_fn = preprocess_fn
        self.root = root
        self.graph = graph
        self.split = split
        self.split_edge = split_edge
        self.percent = percent
        self.ratio_per_hop = ratio_per_hop
        self.directed = directed
        self.dynamic = dynamic
        # import pdb; pdb.set_trace()
        if "weights" in self.graph.edata:
            self.edge_weights = self.graph.edata["weights"]
        else:
            self.edge_weights = None
        if "feat" in self.graph.ndata:
            self.node_features = self.graph.ndata["feat"]
        else:
            self.node_features = None

        pos_edge, neg_edge = ngnn_utils.get_pos_neg_edges(
            self.split, self.split_edge, self.graph, self.percent
        )
        self.links = torch.cat([pos_edge, neg_edge], 0)  # [Np + Nn, 2] [1215518, 2]
        self.labels = np.array([1] * len(pos_edge) + [0] * len(neg_edge))  # [1215518]

        if not self.dynamic:
            self.g_list, tensor_dict = self.load_cached()
            self.labels = tensor_dict["y"]

        # import pdb; pdb.set_trace()
        # compute degree from dataset_pyg
        if 'Graphormer' in model:
            if 'edge_weight' in data_pyg:
                edge_weight = data_pyg.edge_weight.view(-1)
            else:
                edge_weight = torch.ones(data_pyg.edge_index.size(1), dtype=int)
            import scipy.sparse as ssp
            A = ssp.csr_matrix(
                (edge_weight, (data_pyg.edge_index[0], data_pyg.edge_index[1])), 
                shape=(data_pyg.num_nodes, data_pyg.num_nodes))
            if directed:
                A_undirected = ssp.csr_matrix((np.concatenate([edge_weight, edge_weight]), (np.concatenate([data_pyg.edge_index[0], data_pyg.edge_index[1]]), np.concatenate([data_pyg.edge_index[1], data_pyg.edge_index[0]]))), shape=(data_pyg.num_nodes, data_pyg.num_nodes))
                degree_undirected = A_undirected.sum(axis=0).flatten().tolist()[0]
                degree_in = A.sum(axis=0).flatten().tolist()[0]
                degree_out = A.sum(axis=1).flatten().tolist()[0]
                self.degree = torch.Tensor([degree_undirected, degree_in, degree_out]).long()
            else:
                degree_undirected = A.sum(axis=0).flatten().tolist()[0]
                self.degree = torch.Tensor([degree_undirected]).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.dynamic:
            g, y = self.g_list[idx], self.labels[idx]
            x = None if "x" not in g.ndata else g.ndata["x"]
            w = None if "w" not in g.edata else g.eata["w"]
            return g, g.ndata["z"], x, w, y

        src, dst = self.links[idx][0].item(), self.links[idx][1].item()
        y = self.labels[idx]  # 1
        subg = ngnn_utils.k_hop_subgraph(
            src, dst, 1, self.graph, self.ratio_per_hop, self.directed
        )

        # import pdb; pdb.set_trace()
        # Remove the link between src and dst.
        direct_links = [[], []]
        for s, t in [(0, 1), (1, 0)]:
            if subg.has_edges_between(s, t):
                direct_links[0].append(s)
                direct_links[1].append(t)
        if len(direct_links[0]):
            subg.remove_edges(subg.edge_ids(*direct_links))

        NIDs, EIDs = subg.ndata[dgl.NID], subg.edata[dgl.EID]  # [32] [72]

        z = ngnn_utils.drnl_node_labeling(subg.adj(scipy_fmt="csr"), 0, 1)  # [32]
        edge_weights = (
            self.edge_weights[EIDs] if self.edge_weights is not None else None
        )
        x = self.node_features[NIDs] if self.node_features is not None else None  # [32, 128]

        subg_aug = subg.add_self_loop()
        if edge_weights is not None:  # False
            edge_weights = torch.cat(
                [
                    edge_weights,
                    torch.ones(subg_aug.num_edges() - subg.num_edges()),
                ]
            )

        # compute structure from pyg data
        if 'Graphormer' in model:
            subg.x = x
            subg.z = z
            subg.node_id = NIDs
            subg.edge_index = torch.cat([subg.edges()[0].unsqueeze(0), subg.edges()[1].unsqueeze(0)], 0)
            if self.preprocess_fn is not None:
                self.preprocess_fn(subg, directed=self.directed, degree=self.degree)

        # import pdb; pdb.set_trace()
        return subg_aug, z, x, edge_weights, y, subg

    @property
    def cached_name(self):
        return f"SEAL_{self.split}_{self.percent}%.pt"

    def process(self):
        g_list, labels = [], []
        self.dynamic = True
        for i in tqdm(range(len(self))):
            g, z, x, weights, y = self[i]
            g.ndata["z"] = z
            if x is not None:
                g.ndata["x"] = x
            if weights is not None:
                g.edata["w"] = weights
            g_list.append(g)
            labels.append(y)
        self.dynamic = False
        return g_list, {"y": torch.tensor(labels)}

    def load_cached(self):
        path = os.path.join(self.root, self.cached_name)
        if os.path.exists(path):
            return load_graphs(path)

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        g_list, labels = self.process()
        save_graphs(path, g_list, labels)
        return g_list, labels