import torch
import torch.nn.functional as F
from torch_geometric.nn import HANConv


class HAN2(torch.nn.Module):
    def __init__(self, hidden_channels, seed, metadata, heads=8, activation='relu', residual=True):
        super().__init__()
        torch.manual_seed(seed)
        self.activation = activation
        self.residual = residual
        self.hidden_channels = hidden_channels
        self.node_types = metadata[0]

        self.conv1 = HANConv(-1, hidden_channels, metadata, heads=heads, dropout=0.2)
        self.conv2 = HANConv(hidden_channels, 2, metadata, heads=1, dropout=0.2)

        if residual:
            self.res_proj1 = torch.nn.ModuleDict({
                node_type: torch.nn.LazyLinear(hidden_channels) for node_type in self.node_types
            })

    def _activation(self, x):
        return F.relu(x) if self.activation == 'relu' else F.elu(x)

    def forward(self, x_dict, edge_index_dict):
        if self.residual:
            identity1 = {key: self.res_proj1[key](x) for key, x in x_dict.items()}
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {key: self._activation(x + identity1[key]) for key, x in x_dict.items()}
        else:
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {key: self._activation(x) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=0.2, training=self.training) for key, x in x_dict.items()}

        x_dict = self.conv2(x_dict, edge_index_dict)
        return {key: F.log_softmax(x, dim=1) for key, x in x_dict.items()}


def get_model(gnn_model: str, hidden_channels: int, seed: int, gnn_layers: int,
              metadata=None, heads=2, activation='relu', residual=True):
    if gnn_model != "HAN":
        raise ValueError(f"This release only includes the HAN model; got '{gnn_model}'.")
    if metadata is None:
        raise ValueError("HAN model requires metadata (node_types, edge_types)")
    if gnn_layers != 2:
        raise ValueError("This release only includes the 2-layer HAN used in the paper (gnn_layers=2).")
    return HAN2(hidden_channels, seed, metadata, heads=heads, activation=activation, residual=residual), False
