import numpy as np
import torch
from torch_geometric.data import HeteroData
import faiss
from itertools import combinations
from typing import Dict, List, Any


class MultiModalMultiplexGraphBuilder:

    def __init__(self, k_size: int = 3, similarity_method: str = "IndexFlatL2",
                 use_intra_edges: bool = True, use_inter_edges: bool = True):
        self.k_size = k_size
        self.similarity_method = similarity_method
        self.use_intra_edges = use_intra_edges
        self.use_inter_edges = use_inter_edges

    def build_graph(self,
                   modality_embeddings: Dict[str, np.ndarray],
                   labels: List[int],
                   use_fusion_node: bool = False,
                   fusion_init: str = 'zero') -> HeteroData:
        num_pairs = len(labels)
        modality_names = list(modality_embeddings.keys())

        data = HeteroData()

        for mod_name in modality_names:
            data[mod_name].x = torch.tensor(modality_embeddings[mod_name], dtype=torch.float)
            data[mod_name].y = torch.tensor(labels, dtype=torch.float)

        edges = self._build_edges(modality_embeddings, modality_names, num_pairs)

        if self.use_intra_edges:
            for mod_name in modality_names:
                intra_key = f'{mod_name}_{mod_name}'
                if intra_key in edges and edges[intra_key].size(1) > 0:
                    data[mod_name, f'{mod_name}TO{mod_name}', mod_name].edge_index = edges[intra_key]

        if self.use_inter_edges:
            for mod_a, mod_b in combinations(modality_names, 2):
                inter_key = f'{mod_a}_{mod_b}'
                if inter_key in edges and edges[inter_key].size(1) > 0:
                    data[mod_a, f'{mod_a}TO{mod_b}', mod_b].edge_index = edges[inter_key]
                    data[mod_b, f'{mod_b}TO{mod_a}', mod_a].edge_index = edges[f'{mod_b}_{mod_a}']

        if use_fusion_node:
            if fusion_init == 'mean':
                fused_features = np.mean(
                    [modality_embeddings[m] for m in modality_names], axis=0
                )
                data['fused'].x = torch.tensor(fused_features, dtype=torch.float)
            else:
                emb_dim = next(iter(modality_embeddings.values())).shape[1]
                data['fused'].x = torch.zeros(num_pairs, emb_dim, dtype=torch.float)
            data['fused'].y = torch.tensor(labels, dtype=torch.float)

            identity_edges = torch.arange(num_pairs, dtype=torch.long).unsqueeze(0).repeat(2, 1)
            for mod_name in modality_names:
                data[mod_name, f'{mod_name}TOfused', 'fused'].edge_index = identity_edges

        return data

    def _build_edges(self,
                    modality_embeddings: Dict[str, np.ndarray],
                    modality_names: List[str],
                    num_pairs: int) -> Dict[str, torch.Tensor]:

        neighbors_dict = {}
        if self.use_intra_edges and self.k_size > 0:
            for mod_name in modality_names:
                neighbors_dict[mod_name] = self._compute_knn(modality_embeddings[mod_name], self.k_size)

        edges_dict = {}
        edges_dict = self._create_intra_edges(neighbors_dict, edges_dict)
        edges_dict = self._create_inter_edges(edges_dict, modality_names, num_pairs)

        return edges_dict

    def _compute_knn(self, embeddings: np.ndarray, k: int) -> Dict[int, List[int]]:
        d = embeddings.shape[1]
        embeddings = np.array(embeddings, dtype="float32")

        if self.similarity_method == "IndexFlatL2":
            index = faiss.IndexFlatL2(d)
        elif self.similarity_method == "IndexFlatIP":
            index = faiss.IndexFlatIP(d)
        else:
            raise ValueError(f"Unknown similarity method: {self.similarity_method}")

        index.add(embeddings)

        neighbors_dict = {}
        for i, embedding in enumerate(embeddings):
            query_embedding = np.expand_dims(embedding, axis=0)
            _, neighbors = index.search(query_embedding, min(k + 1, len(embeddings)))

            neighbor_list = [n for n in neighbors[0] if n != i][:k]
            neighbors_dict[i] = neighbor_list

        return neighbors_dict

    def _create_intra_edges(self, neighbors_dict: Dict[str, Dict[int, List[int]]], edges_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create intra-modality edges"""
        if self.k_size > 0:
            for modality, neighbors_dict_modality in neighbors_dict.items():
                edges_list = []
                for node_id, neighbors in neighbors_dict_modality.items():
                    edges_list.extend([[neighbor, node_id] for neighbor in neighbors[:self.k_size]])
                edges_dict[f'{modality}_{modality}'] = torch.tensor(edges_list, dtype=torch.long).t()
        return edges_dict

    def _create_inter_edges(self, edges_dict: Dict[str, torch.Tensor], modality_names: List[str], total_nodes: int) -> Dict[str, torch.Tensor]:
        """Create inter-modality edges"""
        inter_edges_list = [[node_id, node_id] for node_id in range(total_nodes)]
        inter_edges_tensor = torch.tensor(inter_edges_list, dtype=torch.long).t()

        for mod_a, mod_b in combinations(modality_names, 2):
            edges_dict[f'{mod_a}_{mod_b}'] = inter_edges_tensor
            edges_dict[f'{mod_b}_{mod_a}'] = inter_edges_tensor

        return edges_dict

    def save_graph(self, graph: HeteroData, save_path: str):
        torch.save(graph, save_path)

    def load_graph(self, load_path: str) -> HeteroData:
        return torch.load(load_path)

    def get_graph_statistics(self, graph: HeteroData) -> Dict[str, Any]:
        node_types = graph.node_types
        stats = {}

        for nt in node_types:
            stats[f'num_{nt}_nodes'] = graph[nt].x.size(0)
            stats[f'{nt}_feature_dim'] = graph[nt].x.size(1)

        edge_types = []
        edge_counts = {}

        for edge_type in graph.edge_types:
            edge_types.append(edge_type)
            edge_key = '__'.join(edge_type)
            edge_counts[edge_key] = graph[edge_type].edge_index.size(1)

        stats['edge_types'] = edge_types
        stats['edge_counts'] = edge_counts
        stats['total_edges'] = sum(edge_counts.values())

        return stats
