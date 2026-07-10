#!/usr/bin/env python3

import os
# Standard fix for PyTorch + FAISS OpenMP conflict on macOS
# Both libraries bundle their own libomp.dylib, which is a known issue
# This is the recommended solution by PyTorch and FAISS communities
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import argparse
import copy
import json
import logging
import pickle
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from gnn_models import get_model
from datetime import datetime
from graph_builder import MultiModalMultiplexGraphBuilder
from metrics import calc_precision, calc_recall, calc_F1
from utils import setup_experiment_dir

def configure_device():
    # Use an accelerator if available (Apple MPS or NVIDIA CUDA), else CPU.
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        try:
            test_tensor = torch.randn(10, 10).to(device)
            _ = test_tensor @ test_tensor.T
            print("Using MPS (Apple Silicon)")
            return device
        except Exception as e:
            print(f"MPS available but not functional ({e}); trying other devices.")

    if torch.cuda.is_available():
        print(f"Using CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")

    print("Using CPU")
    return torch.device("cpu")

device = configure_device()


@dataclass
class MultiModalERConfig:
    """Configuration for multimodal entity resolution with GNNs"""
    embeddings_path: str  # Path to intermediate_embeddings.pkl
    data_root: str = "."  # Root directory for data
    cache_dir: str = "cache"  # Directory for caching graphs
    k_size: int = 3  # Number of nearest neighbors for similarity edges
    similarity_method: str = "IndexFlatL2"  # FAISS similarity method
    
    seed: int = 0
    hidden_channels: int = 250
    epochs_num: int = 150
    gnn_layers: int = 3
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    heads: int = 2
    activation: str = 'relu'
    residual: bool = True
    use_intra_edges: bool = True
    use_inter_edges: bool = True
    modalities: List[str] = None  # None = use all detected
    use_fusion_node: bool = False
    fusion_init: str = 'zero'
    auxiliary_loss: bool = False


EMBEDDINGS_SUFFIX = '_hidden_embeddings'


def detect_modalities(split_data: Dict[str, Any]) -> List[str]:
    return [
        k.replace(EMBEDDINGS_SUFFIX, '')
        for k in split_data.keys()
        if k.endswith(EMBEDDINGS_SUFFIX)
    ]


class MultiModalDataLoader:
    """Loads and processes intermediate embeddings from TensorFlow fusion model"""
    
    def __init__(self, config: MultiModalERConfig):
        self.config = config
        self.embeddings_data = None
        self.modality_names = None

    def load_intermediate_embeddings(self) -> Dict[str, Any]:
        logging.info(f"Loading intermediate embeddings from: {self.config.embeddings_path}")

        with open(self.config.embeddings_path, 'rb') as f:
            self.embeddings_data = pickle.load(f)

        logging.info("Loaded embeddings with keys: " + ", ".join(self.embeddings_data.keys()))

        first_split = self.embeddings_data[next(k for k in ['train', 'val', 'test'])]
        all_modalities = detect_modalities(first_split)
        logging.info(f"Detected modalities: {all_modalities}")

        if self.config.modalities:
            unknown = [m for m in self.config.modalities if m not in all_modalities]
            if unknown:
                raise ValueError(f"Requested modalities not found in embeddings: {unknown}. "
                                 f"Available: {all_modalities}")
            self.modality_names = self.config.modalities
            logging.info(f"Using modality subset: {self.modality_names}")
        else:
            self.modality_names = all_modalities

        for split_name in ['train', 'val', 'test']:
            if split_name in self.embeddings_data:
                metadata = self.embeddings_data[split_name].get('metadata', {})
                if metadata:
                    logging.info(f"{split_name.upper()} metadata: {metadata}")

        return self.embeddings_data

    def prepare_embeddings_for_graph(self) -> Dict[str, Dict[str, Any]]:
        splits = {}

        for split_name in ['train', 'val', 'test']:
            split_data = self.embeddings_data[split_name]

            modality_embeddings = {}
            for mod_name in self.modality_names:
                modality_embeddings[mod_name] = split_data[f'{mod_name}{EMBEDDINGS_SUFFIX}']

            splits[split_name] = {
                'modality_embeddings': modality_embeddings,
                'labels': split_data['labels']
            }

            n_samples = len(split_data['labels'])
            n_positive = np.sum(split_data['labels'])
            logging.info(f"{split_name.upper()} split: {n_samples} samples, {n_positive} positive ({n_positive/n_samples:.2%})")
            for mod_name, emb in modality_embeddings.items():
                logging.info(f"  {mod_name} pair embeddings: {emb.shape}")

        return splits


def build_graphs(config: MultiModalERConfig, embeddings: Dict[str, Any]) -> Dict[str, Any]:
    graph_builder = MultiModalMultiplexGraphBuilder(
        k_size=config.k_size,
        similarity_method=config.similarity_method,
        use_intra_edges=config.use_intra_edges,
        use_inter_edges=config.use_inter_edges,
    )

    graphs = {}

    for split in ['train', 'val', 'test']:
        logging.info(f"Building graph for {split.upper()} split...")

        graph = graph_builder.build_graph(
            modality_embeddings=embeddings[split]['modality_embeddings'],
            labels=embeddings[split]['labels'],
            use_fusion_node=config.use_fusion_node,
            fusion_init=config.fusion_init,
        )

        graphs[split] = graph

        stats = graph_builder.get_graph_statistics(graph)
        logging.info(f"  Total edges: {stats['total_edges']}")
        for nt in graph.node_types:
            if nt != 'fused':
                logging.info(f"  {nt} feature dim: {stats[f'{nt}_feature_dim']}")
        for edge_type, count in stats['edge_counts'].items():
            logging.info(f"  {edge_type}: {count} edges")

    return graphs


def log_prediction_distribution(preds, labels, modality_name: str, split_name: str):
    total = len(preds)
    pred_positives = (preds == 1).sum().item()
    pred_negatives = (preds == 0).sum().item()
    actual_positives = (labels == 1).sum().item()
    actual_negatives = (labels == 0).sum().item()

    true_positives = ((preds == 1) & (labels == 1)).sum().item()
    true_negatives = ((preds == 0) & (labels == 0)).sum().item()

    logging.info(f"    [{split_name}] {modality_name}: "
                f"Pred 0/1: {pred_negatives}/{pred_positives} | "
                f"Actual 0/1: {actual_negatives}/{actual_positives} | "
                f"TP: {true_positives}, TN: {true_negatives}")


def train_model_gnn_all_modalities(model, input_data, optimizer, criterion, seed: int, modality_names: List[str]):
    """Train model for one epoch on ALL modalities at once (per-modality predictions)."""
    torch.manual_seed(seed)
    model.train()

    optimizer.zero_grad()
    out = model(input_data.x_dict, input_data.edge_index_dict)

    total_loss = torch.tensor(0.0, device=device)
    f1_dict = {}
    preds_dict = {}

    for mod_name in modality_names:
        mod_labels = input_data[mod_name].y.type(torch.LongTensor).to(device)
        mod_out = out[mod_name]
        total_loss = total_loss + criterion(mod_out, mod_labels)
        mod_preds = mod_out.argmax(dim=1)
        f1_dict[mod_name] = calc_F1(mod_preds, mod_labels)
        preds_dict[mod_name] = (mod_preds, mod_labels)

    total_loss.backward()
    optimizer.step()

    return f1_dict, total_loss, preds_dict


def eval_model_gnn(model, input_data, modality_key: str, seed: int):
    torch.manual_seed(seed)
    model.eval()

    labels = input_data[modality_key].y.type(torch.LongTensor).to(device)

    with torch.no_grad():
        out = model(input_data.x_dict, input_data.edge_index_dict)
        preds = out[modality_key].argmax(dim=1)

    f1 = calc_F1(preds, labels)
    precision = calc_precision(preds, labels)
    recall = calc_recall(preds, labels)
    return f1, precision, recall, preds, labels


def train_model_gnn_fusion(model, input_data, optimizer, criterion, seed: int,
                           modality_names: List[str], auxiliary_loss: bool = False):
    torch.manual_seed(seed)
    model.train()

    optimizer.zero_grad()
    out = model(input_data.x_dict, input_data.edge_index_dict)

    fused_labels = input_data["fused"].y.type(torch.LongTensor).to(device)
    fused_out = out["fused"]
    loss = criterion(fused_out, fused_labels)

    if auxiliary_loss:
        for mod_name in modality_names:
            mod_labels = input_data[mod_name].y.type(torch.LongTensor).to(device)
            loss = loss + criterion(out[mod_name], mod_labels)

    loss.backward()
    optimizer.step()

    preds = fused_out.argmax(dim=1)
    f1 = calc_F1(preds, fused_labels)
    return f1, loss, (preds, fused_labels)


def eval_model_gnn_fusion(model, input_data, seed: int):
    torch.manual_seed(seed)
    model.eval()

    labels = input_data["fused"].y.type(torch.LongTensor).to(device)

    with torch.no_grad():
        out = model(input_data.x_dict, input_data.edge_index_dict)
        preds = out["fused"].argmax(dim=1)

    f1 = calc_F1(preds, labels)
    precision = calc_precision(preds, labels)
    recall = calc_recall(preds, labels)
    return f1, precision, recall, preds, labels

def train_and_evaluate_gnn(graphs: Dict[str, HeteroData], config: MultiModalERConfig, modality_names: List[str], experiment_dir: str, gnn_model_type: str, prebuilt_model: torch.nn.Module = None):
    use_fusion_node = config.use_fusion_node

    metadata = graphs['train'].metadata()

    logging.info(f"GNN Training Configuration:")
    logging.info(f"  Modalities: {modality_names}")
    logging.info(f"  Hidden channels: {config.hidden_channels}")
    logging.info(f"  Epochs: {config.epochs_num}")
    logging.info(f"  Model: {gnn_model_type}")
    logging.info(f"  Layers: {config.gnn_layers}")
    logging.info(f"  Seed: {config.seed}")
    logging.info(f"  Learning rate: {config.learning_rate}")
    logging.info(f"  Weight decay: {config.weight_decay}")
    if use_fusion_node:
        logging.info(f"  Fusion node mode: Predictions from fused node type")
        logging.info(f"  Fusion init: {config.fusion_init}")
        logging.info(f"  Auxiliary loss: {config.auxiliary_loss}")

    if prebuilt_model is not None:
        logging.info(f"  Using prebuilt model (sequential Phase 2)")
        model = prebuilt_model
    else:
        model, _ = get_model(
            gnn_model_type,
            config.hidden_channels,
            config.seed,
            config.gnn_layers,
            metadata=metadata,
            heads=config.heads,
            activation=config.activation,
            residual=config.residual,
        )
        logging.info(f"  Model is natively heterogeneous (HAN)")

    model.to(device)

    train_graph = graphs['train'].to(device)
    val_graph = graphs['val'].to(device)
    test_graph = graphs['test'].to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    logging.info(f"  Loss: CrossEntropyLoss (no class weights)")
    
    scheduler = None
    if gnn_model_type == "HAN":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=30, T_mult=2, eta_min=1e-6
        )
        logging.info(f"  Scheduler: CosineAnnealingWarmRestarts (T_0=30, T_mult=2, eta_min=1e-6)")
        logging.info(f"  Restart epochs: 30, 90, 210, 450, ...")
    else:
        logging.info(f"  Scheduler: None")

    results_dict = {}
    
    logging.info("=" * 50)
    logging.info("Training All Modalities Together:")
    logging.info("=" * 50)

    best_combined_val_f1 = 0.0
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, config.epochs_num + 1):
        if use_fusion_node:
            F1_train_fused, train_loss, (fused_preds_train, fused_labels_train) = train_model_gnn_fusion(
                model, train_graph, optimizer, criterion, config.seed,
                modality_names=modality_names, auxiliary_loss=config.auxiliary_loss
            )
        else:
            train_f1_dict, train_loss, train_preds_dict = train_model_gnn_all_modalities(
                model, train_graph, optimizer, criterion, config.seed,
                modality_names=modality_names
            )

        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

        if use_fusion_node:
            F1_val_fused, _, _, val_fused_preds, val_fused_labels = eval_model_gnn_fusion(model, val_graph, config.seed)
            combined_val_f1 = F1_val_fused.item()
        else:
            val_f1_dict = {}
            val_preds_dict = {}
            combined_val_f1 = 0.0
            for mod_name in modality_names:
                f1_val, _, _, val_preds, val_labels = eval_model_gnn(model, val_graph, mod_name, config.seed)
                val_f1_dict[mod_name] = f1_val
                val_preds_dict[mod_name] = (val_preds, val_labels)
                combined_val_f1 += f1_val.item()

        if epoch % 5 == 0 or epoch < 5:
            if use_fusion_node:
                logging.info(f'Epoch {epoch:3d} Results (Training Loss: {train_loss.item():.4f}):')
                logging.info(f'  Fused  - Train F1: {F1_train_fused.item():.4f}, Val F1: {F1_val_fused.item():.4f}')
            else:
                logging.info(f'Epoch {epoch:3d} Results (Combined Training Loss: {train_loss.item():.4f}):')
                for mod_name in modality_names:
                    train_f1_val = train_f1_dict[mod_name].item()
                    val_f1_val = val_f1_dict[mod_name].item()
                    logging.info(f'  {mod_name:15s} - Train F1: {train_f1_val:.4f}, Val F1: {val_f1_val:.4f}')

        if scheduler is not None and epoch in [30, 90, 210, 450]:
            logging.info(f'  WARM RESTART at epoch {epoch} - LR reset to {current_lr:.2e}')

        if epoch <= 10 or epoch % 10 == 0 or (epoch > 55 and epoch <= 70):
            logging.info(f'  Epoch {epoch} Prediction Distribution:')
            if use_fusion_node:
                log_prediction_distribution(fused_preds_train, fused_labels_train, "Fused", "Train")
                log_prediction_distribution(val_fused_preds, val_fused_labels, "Fused", "Val")
            else:
                for mod_name in modality_names:
                    t_preds, t_labels = train_preds_dict[mod_name]
                    v_preds, v_labels = val_preds_dict[mod_name]
                    log_prediction_distribution(t_preds, t_labels, mod_name, "Train")
                    log_prediction_distribution(v_preds, v_labels, mod_name, "Val")

        if combined_val_f1 > best_combined_val_f1:
            best_combined_val_f1 = combined_val_f1
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            logging.info(f'  New best Val F1: {combined_val_f1:.4f} at epoch {epoch}')

    logging.info("=" * 50)
    if best_model_state is not None:
        logging.info(f"Restoring best model weights from epoch {best_epoch}")
        logging.info(f"   Best Val F1: {best_combined_val_f1:.4f}")
        model.load_state_dict(best_model_state)
    else:
        logging.warning("No epoch improved validation F1 beyond 0.0 — using final model weights")
    logging.info("=" * 50)

    logging.info("Final evaluation on test set with best model...")
    if use_fusion_node:
        F1_test_fused, prec_test_fused, rec_test_fused, _, _ = eval_model_gnn_fusion(model, test_graph, config.seed)
        results_dict['fused'] = {
            'f1_score': float(F1_test_fused.item()),
            'precision': float(prec_test_fused.item()),
            'recall': float(rec_test_fused.item())
        }
        results_dict['best_epoch'] = best_epoch
        results_dict['best_val_f1'] = float(best_combined_val_f1)
        logging.info("=" * 50)
        logging.info(f"Best Model from Epoch {best_epoch} - Test Set Results:")
        logging.info(f"   Fused  - F1: {F1_test_fused.item():.4f}, Precision: {prec_test_fused.item():.4f}, Recall: {rec_test_fused.item():.4f}")
        logging.info("=" * 50)
    else:
        for mod_name in modality_names:
            f1_t, prec_t, rec_t, _, _ = eval_model_gnn(model, test_graph, mod_name, config.seed)
            results_dict[mod_name] = {
                'f1_score': float(f1_t.item()),
                'precision': float(prec_t.item()),
                'recall': float(rec_t.item())
            }

        results_dict['best_epoch'] = best_epoch
        results_dict['best_combined_val_f1'] = float(best_combined_val_f1)

        logging.info("=" * 50)
        logging.info(f"Best Model from Epoch {best_epoch} - Test Set Results:")
        combined_test_f1 = 0.0
        for mod_name in modality_names:
            r = results_dict[mod_name]
            logging.info(f"   {mod_name:15s} - F1: {r['f1_score']:.4f}, Precision: {r['precision']:.4f}, Recall: {r['recall']:.4f}")
            combined_test_f1 += r['f1_score']
        logging.info(f"   Combined Test F1: {combined_test_f1:.4f}")
        logging.info("=" * 50)
    logging.info("GNN training completed!")

    if experiment_dir:
        logging.info("=" * 50)
        logging.info("Saving GNN model and embeddings...")
        logging.info("=" * 50)

        model_path = os.path.join(experiment_dir, "gnn_model.pt")
        torch.save({
            'model_state_dict': model.state_dict(),
            'best_epoch': best_epoch,
            'best_combined_val_f1': best_combined_val_f1,
            'config': {
                'hidden_channels': config.hidden_channels,
                'gnn_layers': config.gnn_layers,
                'learning_rate': config.learning_rate,
                'weight_decay': config.weight_decay,
                'epochs_num': config.epochs_num,
                'seed': config.seed
            }
        }, model_path)
        logging.info(f"Model saved to: {model_path}")

        logging.info("=" * 50)

    return results_dict, model


def _is_semantic_attention_param(name: str) -> bool:
    return name.endswith('.q') or '.k_lin.' in name


def freeze_transferred_weights(model: torch.nn.Module,
                               prev_state_dict: Dict[str, torch.Tensor]) -> int:
    """Transfer weights from the previous phase wherever name and shape match."""
    n_frozen = 0
    n_warm_trainable = 0
    for name, param in model.named_parameters():
        if name in prev_state_dict and prev_state_dict[name].shape == param.data.shape:
            param.data.copy_(prev_state_dict[name])
            if _is_semantic_attention_param(name):
                n_warm_trainable += 1
            else:
                param.requires_grad = False
                n_frozen += 1
    logging.info(f"  Transferred {n_frozen + n_warm_trainable} tensors from prev phase "
                 f"(froze {n_frozen}, kept {n_warm_trainable} semantic-attention trainable)")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    logging.info(f"  Trainable params this phase: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
    return n_frozen


def train_sequential_gnn(all_embeddings: Dict[str, Any],
                          config: MultiModalERConfig,
                          stages: List[List[str]],
                          experiment_dir: str,
                          gnn_model_type: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Chained sequential training over an ordered list of cumulative modality sets.

      stages[0]        — base set (>=2 modalities), trained from scratch.
      stages[i] (i>=1) — expand the graph to stages[i], transfer + freeze the
                          parameters carried over from the previous (trained) stage,
                          and train only the parameters introduced by the newly added
                          modality and its new links (plus shared semantic-attention).

    Each stage is cumulative: stages[i] must be a superset of stages[i-1].
    Returns (final_stage_metrics, per_stage_records).
    """
    from gnn_models import get_model

    stage_records = []
    prev_state_dict = None
    final_metrics = None

    for i, mods in enumerate(stages):
        phase_num = i + 1
        phase_dir = os.path.join(experiment_dir, f'phase{phase_num}_{len(mods)}mod')
        os.makedirs(phase_dir, exist_ok=True)

        sub_embeddings = {
            split: {
                'modality_embeddings': {m: data['modality_embeddings'][m] for m in mods},
                'labels': data['labels'],
            }
            for split, data in all_embeddings.items()
        }
        graphs = build_graphs(config, sub_embeddings)

        if i == 0:
            logging.info(f"── Phase {phase_num}: train base {mods} from scratch ──")
            metrics, model = train_and_evaluate_gnn(
                graphs, config, mods, phase_dir, gnn_model_type,
            )
        else:
            added = [m for m in mods if m not in stages[i - 1]]
            logging.info(f"── Phase {phase_num}: expand to {mods} (+adding {added}), freeze prior weights ──")
            metadata = graphs['train'].metadata()
            model, _ = get_model(
                gnn_model_type, config.hidden_channels, config.seed,
                config.gnn_layers, metadata=metadata,
                heads=config.heads, activation=config.activation, residual=config.residual,
            )
            model.to(device)

            dummy_graph = graphs['train'].to(device)
            with torch.no_grad():
                model(dummy_graph.x_dict, dummy_graph.edge_index_dict)

            freeze_transferred_weights(model, prev_state_dict)

            metrics, model = train_and_evaluate_gnn(
                graphs, config, mods, phase_dir, gnn_model_type,
                prebuilt_model=model,
            )

        prev_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
        stage_records.append({
            'phase': phase_num,
            'n_modalities': len(mods),
            'modalities': list(mods),
            'added': [] if i == 0 else [m for m in mods if m not in stages[i - 1]],
            'test_results': metrics,
        })
        final_metrics = metrics

    return final_metrics, stage_records


DATASET_HPARAMS = {
    "WDC Shoes": {"epochs": 150, "learning_rate": 0.005, "heads": 2, "activation": "relu"},
    "Zalando":   {"epochs": 150, "learning_rate": 0.001, "heads": 8, "activation": "elu"},
    "Hague":     {"epochs": 500, "learning_rate": 0.005, "heads": 8, "activation": "relu"},
}


def main():
    parser = argparse.ArgumentParser(description='Run GNN training with embeddings from trained embedding branches')
    parser.add_argument('--embedding-branches-dir', type=str, required=True,
                       help='Directory containing precomputed embeddings and config_and_results.json')
    parser.add_argument('--k-size', type=int, default=3,
                       help='Number of nearest neighbors for intra-modality similarity edges')
    parser.add_argument('--edge-types', type=str, default='both',
                       choices=['both', 'intra_only', 'inter_only'],
                       help='Which edge types to include in the graph')
    parser.add_argument('--modalities', type=str, nargs='+', default=None,
                       help='Subset of modalities to use (default: all detected). '
                            'Example: --modalities image feature_vector')
    parser.add_argument('--sequential', action='store_true',
                       help='Enable chained sequential training: train the base set, then '
                            'add modalities one at a time, freezing prior weights at each step.')
    parser.add_argument('--base-modalities', type=str, nargs='+', default=None,
                       help='Starting modalities for sequential training (>=2). Required with --sequential.')
    parser.add_argument('--add-modalities', type=str, nargs='+', default=None,
                       help='Modalities to add one at a time (in this order) after the base, '
                            'for --sequential. e.g. --base-modalities image feature_vector '
                            '--add-modalities point_cloud normals  →  2→3→4 chain.')

    args = parser.parse_args()

    sequential_stages = None
    if args.sequential:
        if not args.base_modalities or len(args.base_modalities) < 2:
            raise ValueError("--sequential requires --base-modalities with at least 2 modalities")
        if not args.add_modalities:
            raise ValueError("--sequential requires --add-modalities (modalities to add one at a time)")
        if set(args.base_modalities) & set(args.add_modalities):
            raise ValueError("--base-modalities and --add-modalities must be disjoint")
        if len(set(args.add_modalities)) != len(args.add_modalities):
            raise ValueError("--add-modalities must not contain duplicates")
        sequential_stages = [list(args.base_modalities)]
        for m in args.add_modalities:
            sequential_stages.append(sequential_stages[-1] + [m])
        args.modalities = sequential_stages[-1]

    experiment_dir = setup_experiment_dir("gnn")
    embedding_config_path = os.path.join(args.embedding_branches_dir, "config_and_results.json")
    with open(embedding_config_path, 'r') as f:
        embedding_config = json.load(f)
    embeddings_path = embedding_config['embeddings']['embeddings_path']

    dataset_name = embedding_config['dataset']
    hp = DATASET_HPARAMS[dataset_name]

    config = MultiModalERConfig(
        embeddings_path=embeddings_path,
        k_size=args.k_size,
        similarity_method="IndexFlatL2",
        use_intra_edges=(args.edge_types in ('both', 'intra_only')),
        use_inter_edges=(args.edge_types in ('both', 'inter_only')),
        modalities=args.modalities,
        cache_dir=os.path.join(experiment_dir, "graph_cache"),
        seed=42,
        hidden_channels=256,
        epochs_num=hp["epochs"],
        gnn_layers=2,
        learning_rate=hp["learning_rate"],
        weight_decay=5e-4,
        heads=hp["heads"],
        activation=hp["activation"],
        residual=True,
        use_fusion_node=True,
        fusion_init="zero",
        auxiliary_loss=True,
    )

    logging.info("=== Multimodal Entity Resolution GNN Pipeline ===")
    logging.info("GNN Model: HAN")
    logging.info(f"Loading embeddings from: {args.embedding_branches_dir}")
    logging.info(f"GNN results will be saved to: {experiment_dir}")
    logging.info(f"Dataset: {embedding_config['dataset']}")
    logging.info(f"Embeddings path: {config.embeddings_path}")
    logging.info(f"K-size for similarity: {config.k_size}")
    logging.info(f"Edge types: {args.edge_types} (intra={config.use_intra_edges}, inter={config.use_inter_edges})")
    if args.modalities:
        logging.info(f"Modality subset: {args.modalities}")
    if args.sequential:
        chain = ' → '.join('+'.join(s) for s in sequential_stages)
        logging.info(f"Sequential training chain: {chain}")
    logging.info(f"Similarity method: {config.similarity_method}")

    try:
        logging.info("Step 1: Loading intermediate embeddings...")
        data_loader = MultiModalDataLoader(config)
        data_loader.load_intermediate_embeddings()
        modality_names = data_loader.modality_names
        embeddings = data_loader.prepare_embeddings_for_graph()

        for split, data in embeddings.items():
            parts = [f"{m} {data['modality_embeddings'][m].shape}" for m in modality_names]
            logging.info(f"{split.upper()}: {', '.join(parts)}, labels {len(data['labels'])}")

        if args.sequential:
            unknown = [m for m in sequential_stages[-1] if m not in modality_names]
            if unknown:
                raise ValueError(f"Sequential modalities not found in embeddings: {unknown}. "
                                 f"Available: {modality_names}")

            logging.info(f"\nStep 2+3: Chained sequential GNN training...")
            final_metrics, stage_records = train_sequential_gnn(
                embeddings, config,
                stages=sequential_stages,
                experiment_dir=experiment_dir,
                gnn_model_type="HAN",
            )
        else:
            stage_records = None
            logging.info("Step 2: Building multiplex graphs...")
            graphs = build_graphs(config, embeddings)

            logging.info("\nStep 3: GNN training with HAN (multimodal entity resolution)...")
            final_metrics, _ = train_and_evaluate_gnn(graphs, config, modality_names, experiment_dir, gnn_model_type="HAN")

        modality_config = embedding_config.get('modalities')
        if not modality_config:
            modality_config = {}
            for m in modality_names:
                emb_key = f'{m}_embedding'
                if emb_key in embedding_config:
                    modality_config[m] = embedding_config[emb_key]

        results_summary = {
            'experiment_type': 'gnn',
            'embedding_branches_dir': args.embedding_branches_dir,
            'dataset': embedding_config['dataset'],
            'modalities': modality_names,
            'modality_config': modality_config,
            'gnn_config': {
                'model_type': "HAN",
                'k_size': config.k_size,
                'edge_types': args.edge_types,
                'use_intra_edges': config.use_intra_edges,
                'use_inter_edges': config.use_inter_edges,
                'modality_subset': args.modalities,
                'hidden_channels': config.hidden_channels,
                'epochs': config.epochs_num,
                'gnn_layers': config.gnn_layers,
                'learning_rate': config.learning_rate,
                'weight_decay': config.weight_decay,
                'heads': config.heads,
                'activation': config.activation,
                'residual': config.residual,
                'use_fusion_node': config.use_fusion_node,
                'fusion_init': config.fusion_init,
                'auxiliary_loss': config.auxiliary_loss,
                'sequential_training': args.sequential,
                'base_modalities': args.base_modalities if args.sequential else None,
                'add_modalities': args.add_modalities if args.sequential else None,
                'sequential_stages': sequential_stages if args.sequential else None,
            },
            'test_results': final_metrics,
            'sequential_stage_results': stage_records,
            'saved_artifacts': {
                'model_path': os.path.join(experiment_dir, "gnn_model.pt")
            }
        }

        summary_path = os.path.join(experiment_dir, "gnn_results_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        logging.info(f"\nResults summary saved to: {summary_path}")

        logging.info("\nMultimodal GNN pipeline completed successfully!")

        return final_metrics

    except Exception as e:
        logging.error(f"Error in GNN pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
