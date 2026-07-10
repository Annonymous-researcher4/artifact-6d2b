#!/usr/bin/env python3
"""

Modalities:
  - Image: contrastive fine-tuning of CLIP ViT-B/32 on rendered 3D mesh pairs
  - Feature vector: 3-layer MLP with BatchNorm on 25 geometric property ratios
  - Point cloud: Siamese PointNet on raw 3D vertices
  - Normals: MLP on area-weighted surface normal histograms
"""

import os
import json
import pickle
import logging
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset

import clip
from PIL import Image

from unimodal_pair_encoders.utils import setup_experiment_dir
from unimodal_pair_encoders.hague.utils import (
    generate_figs_dir, get_existing_images_dict, generate_png_figs_wrapper,
    load_dataset_partition_dict, load_object_dict
)
from unimodal_pair_encoders.hague.object_properties import ObjectPropertiesProcessor
from unimodal_pair_encoders.hague.process_pairs import PairProcessor
from unimodal_pair_encoders.hague import config as hague_config

CLIP_EMBEDDING_DIMS = {'ViT-B_32': 512, 'ViT-L_14': 768}
HIDDEN_DIM = 256
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

from unimodal_pair_encoders.hague import utils as hague_utils
hague_utils.device = str(DEVICE)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ContrastiveBinaryClassifier(nn.Module):

    def __init__(self, vision_encoder, emb_dim, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.encoder = vision_encoder
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, img1, img2):
        emb1 = self.encoder(img1)
        emb2 = self.encoder(img2)
        diff = torch.abs(emb1 - emb2)
        return self.classifier(diff).squeeze(1)

    def extract_hidden(self, img1, img2):
        with torch.no_grad():
            emb1 = self.encoder(img1)
            emb2 = self.encoder(img2)
            diff = torch.abs(emb1 - emb2)
            hidden = self.classifier[0](diff)
            return self.classifier[1](hidden)


class FeatureVectorMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))
        return self.classifier(x).squeeze(-1)

    def extract_hidden(self, x):
        with torch.no_grad():
            x = self.relu(self.bn1(self.fc1(x)))
            x = self.relu(self.bn2(self.fc2(x)))
            return self.relu(self.bn3(self.fc3(x)))


class PointNetEncoder(nn.Module):

    def __init__(self, out_dim=HIDDEN_DIM):
        super().__init__()
        self.mlp1 = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.BatchNorm1d(64))
        self.mlp2 = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.BatchNorm1d(128))
        self.mlp3 = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.BatchNorm1d(256))
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x):
        B, N, _ = x.shape
        x = x.reshape(B * N, 3)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = x.reshape(B, N, 256)
        x = x.max(dim=1)[0]
        return self.fc(x)


class SiamesePointNet(nn.Module):

    def __init__(self, n_points, out_dim=HIDDEN_DIM):
        super().__init__()
        self.n_points = n_points
        self.encoder = PointNetEncoder(out_dim)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, 1)
        )

    def forward(self, cloud1, cloud2):
        emb1 = self.encoder(cloud1)
        emb2 = self.encoder(cloud2)
        diff = torch.abs(emb1 - emb2)
        return self.classifier(diff).squeeze(-1)

    def extract_hidden(self, cloud1, cloud2):
        with torch.no_grad():
            emb1 = self.encoder(cloud1)
            emb2 = self.encoder(cloud2)
            diff = torch.abs(emb1 - emb2)
            return self.classifier[1](self.classifier[0](diff))


class NormalsMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = self.relu(self.hidden(x))
        return self.classifier(h).squeeze(-1)

    def extract_hidden(self, x):
        with torch.no_grad():
            return self.relu(self.hidden(x))


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class ContrastivePairDataset(Dataset):

    def __init__(self, pairs: List[Tuple], cands_figs_path: str,
                 index_figs_path: str, transform):
        self.pairs = pairs
        self.cands_figs_path = cands_figs_path
        self.index_figs_path = index_figs_path
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cand_id, index_id = self.pairs[idx]
        label = 1.0 if cand_id == index_id else 0.0
        cand_img = self.transform(
            Image.open(os.path.join(self.cands_figs_path, f"{cand_id}.png")).convert("RGB")
        )
        index_img = self.transform(
            Image.open(os.path.join(self.index_figs_path, f"{index_id}.png")).convert("RGB")
        )
        return cand_img, index_img, torch.tensor(label, dtype=torch.float32)


class PointCloudPairDataset(Dataset):

    def __init__(self, pairs, all_cands, all_index, n_points):
        self.pairs = pairs
        self.all_cands = all_cands
        self.all_index = all_index
        self.n_points = n_points

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        cand_id, index_id = self.pairs[idx]
        label = 1.0 if cand_id == index_id else 0.0
        cand_verts = _get_vertices(self.all_cands[cand_id])
        index_verts = _get_vertices(self.all_index[index_id])
        cand_cloud = _pad_or_sample_points(_normalize_point_cloud(cand_verts), self.n_points)
        index_cloud = _pad_or_sample_points(_normalize_point_cloud(index_verts), self.n_points)
        return (
            torch.tensor(cand_cloud, dtype=torch.float32),
            torch.tensor(index_cloud, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_hague_data(dataset_name: str, seed: int, dataset_size: str, neg_samples: int,
                      matching_cands_generation: str):
    logger = logging.getLogger('hague_loader')
    dataset_partition_dict = load_dataset_partition_dict(dataset_name, logger, seed)
    train_pairs = dataset_partition_dict['train'][matching_cands_generation][dataset_size][neg_samples]
    test_pairs = dataset_partition_dict['test']['matching'][matching_cands_generation][dataset_size][neg_samples]
    return train_pairs, test_pairs


def load_object_dicts(dataset_name: str, seed: int, dataset_size: str, neg_samples: int,
                      matching_cands_generation: str, evaluation_mode: str = 'matching'):
    object_dict_path = f"{hague_config.FilePaths.object_dict_path}{dataset_name}/"
    train_path = (f"{object_dict_path}train_{evaluation_mode}_{dataset_size}_"
                  f"neg_samples_num={neg_samples}_seed_{seed}.joblib")
    test_path = (f"{object_dict_path}test_{evaluation_mode}_{matching_cands_generation}"
                 f"_{dataset_size}_neg_samples_num={neg_samples}_seed_{seed}.joblib")
    logger = logging.getLogger('hague_loader')
    train_object_dict = load_object_dict(logger, train_path, 'train_object_dict')
    test_object_dict = load_object_dict(logger, test_path, 'test_object_dict')
    return train_object_dict, test_object_dict


def generate_png_images(object_dict: Dict, dataset_name: str):
    """Render 3D meshes to PNGs."""
    index_figs_path, cands_figs_path = generate_figs_dir(dataset_name)
    existing_images_dict = get_existing_images_dict(index_figs_path, cands_figs_path)
    generate_png_figs_wrapper(object_dict, existing_images_dict, index_figs_path, cands_figs_path)
    cands_figs_path = os.path.abspath(cands_figs_path)
    index_figs_path = os.path.abspath(index_figs_path)
    return cands_figs_path, index_figs_path


def compute_property_dict(object_dict: Dict, normalization: str = 'log_transform') -> Dict:
    processor = ObjectPropertiesProcessor(object_dict, normalization)
    prop_dict = processor.prop_vals_dict
    return prop_dict


def build_feature_vector_pairs(pairs: List[Tuple], property_dict: Dict) -> Dict[str, np.ndarray]:
    """Build feature-vector pair features from geometric property ratios."""
    pair_processor = PairProcessor(property_dict, pairs)
    fv_vecs = pair_processor.feature_vec
    labels = np.array([1 if p[0] == p[1] else 0 for p in pairs], dtype=np.int64)
    return {
        'feature_vector_pair_features': np.array(fv_vecs, dtype=np.float32),
        'labels': labels
    }


def filter_pairs_with_images(pairs: List[Tuple], cands_figs_path: str,
                             index_figs_path: str) -> List[Tuple]:
    valid = []
    for cand_id, index_id in pairs:
        if (os.path.exists(os.path.join(cands_figs_path, f"{cand_id}.png")) and
                os.path.exists(os.path.join(index_figs_path, f"{index_id}.png"))):
            valid.append((cand_id, index_id))
    return valid


# ---------------------------------------------------------------------------
# Point cloud helpers
# ---------------------------------------------------------------------------

def _get_vertices(obj):
    pts = [pt for surface in obj['polygon_mesh'] for pt in surface]
    return np.unique(pts, axis=0)


def _normalize_point_cloud(vertices):
    """Center to origin and scale to unit sphere."""
    centered = vertices - vertices.mean(axis=0)
    max_dist = np.abs(centered).max()
    if max_dist > 1e-10:
        centered = centered / max_dist
    return centered


def _pad_or_sample_points(vertices, n_points):
    """Pad (repeat) or randomly subsample to exactly n_points."""
    n = len(vertices)
    if n >= n_points:
        idx = np.random.choice(n, n_points, replace=False)
        return vertices[idx]
    pad_idx = np.random.choice(n, n_points - n, replace=True)
    return np.concatenate([vertices, vertices[pad_idx]], axis=0)


# ---------------------------------------------------------------------------
# Surface normal histogram helpers
# ---------------------------------------------------------------------------

def _compute_face_normals(polygon_mesh):
    """Compute unit normals and areas for each triangulated face."""
    normals, areas = [], []
    for surface in polygon_mesh:
        pts = np.array(surface)
        if len(pts) < 3:
            continue
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[0]
            v2 = pts[i + 1] - pts[0]
            cross = np.cross(v1, v2)
            area = np.linalg.norm(cross)
            if area > 1e-10:
                normals.append(cross / area)
                areas.append(area)
    if not normals:
        return np.zeros((1, 3)), np.array([1.0])
    return np.array(normals), np.array(areas)


def _normal_histogram(polygon_mesh, n_elev_bins=18, n_azim_bins=36):
    """Area-weighted histogram of face normal orientations."""
    normals, areas = _compute_face_normals(polygon_mesh)
    elevations = np.degrees(np.arcsin(np.clip(normals[:, 2], -1, 1)))
    azimuths = np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
    elev_hist, _ = np.histogram(elevations, bins=n_elev_bins, range=(-90, 90), weights=areas)
    azim_hist, _ = np.histogram(azimuths, bins=n_azim_bins, range=(-180, 180), weights=areas)
    elev_sum = elev_hist.sum()
    azim_sum = azim_hist.sum()
    if elev_sum > 0:
        elev_hist = elev_hist / elev_sum
    if azim_sum > 0:
        azim_hist = azim_hist / azim_sum
    return np.concatenate([elev_hist, azim_hist]).astype(np.float32)


def build_normal_pair_features(pairs, all_cands, all_index):
    """Build |hist_cand - hist_index| pair features for all pairs."""
    features, labels = [], []
    for cand_id, index_id in pairs:
        cand_hist = _normal_histogram(all_cands[cand_id]['polygon_mesh'])
        index_hist = _normal_histogram(all_index[index_id]['polygon_mesh'])
        features.append(np.abs(cand_hist - index_hist))
        labels.append(1 if cand_id == index_id else 0)
    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int64)


# ----------------------------------------------
# Image branch: contrastive fine-tuning of CLIP 
# ----------------------------------------------

def train_contrastive_image_model(
    vit_model_name: str, clip_dim: int,
    train_pairs: List[Tuple], val_pairs: List[Tuple],
    cands_figs_path: str, index_figs_path: str,
    epochs: int = 8, batch_size: int = 4, lr: float = 1e-5
) -> Tuple[ContrastiveBinaryClassifier, any]:
    """Fine-tune CLIP vision encoder on image pairs with BCEWithLogitsLoss."""

    clip_model_name = vit_model_name.replace('_', '/')
    model, preprocess = clip.load(clip_model_name, device=DEVICE)
    vision_encoder = model.visual

    contrastive_model = ContrastiveBinaryClassifier(
        vision_encoder, clip_dim, HIDDEN_DIM
    ).to(DEVICE).float()

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(contrastive_model.parameters(), lr=lr)

    train_dataset = ContrastivePairDataset(train_pairs, cands_figs_path, index_figs_path, preprocess)
    val_dataset = ContrastivePairDataset(val_pairs, cands_figs_path, index_figs_path, preprocess)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_f1 = -1
    best_state = None
    patience_counter = 0
    patience = 3

    for epoch in range(1, epochs + 1):
        contrastive_model.train()
        train_loss = 0.0
        for img1, img2, label in train_loader:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            output = contrastive_model(img1, img2)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(label)
        train_loss /= len(train_dataset)

        val_metrics = _evaluate_contrastive(contrastive_model, val_loader)
        logging.info(
            f"  [image] Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['acc']:.4f}, "
            f"val_f1={val_metrics['f1']:.4f}, val_pr={val_metrics['precision']:.4f}, "
            f"val_re={val_metrics['recall']:.4f}"
        )

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = {k: v.clone() for k, v in contrastive_model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"  [image] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        contrastive_model.load_state_dict(best_state)
    logging.info(f"  [image] Best val F1: {best_val_f1:.4f}")
    return contrastive_model, preprocess


def _evaluate_contrastive(model, dataloader):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for img1, img2, label in dataloader:
            img1, img2, label = img1.to(DEVICE), img2.to(DEVICE), label.to(DEVICE)
            output = model(img1, img2)
            total_loss += loss_fn(output, label).item() * len(label)
            preds = (torch.sigmoid(output) > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(label.cpu().numpy().astype(int))

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    n = len(all_labels)
    return {
        'loss': total_loss / n,
        'acc': (all_preds == all_labels).mean(),
        'f1': f1_score(all_labels, all_preds, zero_division=0),
        'precision': precision_score(all_labels, all_preds, zero_division=0),
        'recall': recall_score(all_labels, all_preds, zero_division=0),
    }


def extract_contrastive_embeddings(model: ContrastiveBinaryClassifier, pairs: List[Tuple],
                                   cands_figs_path: str, index_figs_path: str,
                                   preprocess, batch_size: int = 32) -> np.ndarray:
    """Extract embeddings from the contrastive model for all pairs."""
    model.eval()
    dataset = ContrastivePairDataset(pairs, cands_figs_path, index_figs_path, preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_hidden = []
    with torch.no_grad():
        for img1, img2, _ in loader:
            img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
            hidden = model.extract_hidden(img1, img2)
            all_hidden.append(hidden.cpu().numpy())

    return np.concatenate(all_hidden, axis=0)


# ---------------------------------------------------------------------------
# Feature vector branch: MLP on geometric property ratios
# ---------------------------------------------------------------------------

def train_fv_mlp(model: FeatureVectorMLP, train_x: np.ndarray, train_y: np.ndarray,
                 val_x: np.ndarray, val_y: np.ndarray,
                 epochs: int = 500, batch_size: int = 64, lr: float = 1e-4,
                 modality_name: str = 'feature_vector') -> FeatureVectorMLP:

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(val_x), torch.tensor(val_y, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 25

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                val_loss += criterion(out, yb).item() * len(xb)
                preds = (torch.sigmoid(out) > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.long().cpu().numpy())
        val_loss /= len(val_ds)
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        val_acc = (all_preds == all_labels).mean()
        val_f1 = f1_score(all_labels, all_preds)

        if epoch % 10 == 0 or epoch <= 5:
            logging.info(f"  [{modality_name}] Epoch {epoch:3d}: "
                        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                        f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"  [{modality_name}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# PointNet branch: Siamese PointNet on raw 3D vertices
# ---------------------------------------------------------------------------

def _evaluate_siamese(model, dataloader):
    """Evaluate a Siamese model (PointNet) on a dataloader."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            *inputs, labels = [b.to(DEVICE) for b in batch]
            out = model(*inputs)
            total_loss += loss_fn(out, labels).item() * len(labels)
            preds = (torch.sigmoid(out) > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy().astype(int))

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    n = len(all_labels)
    return {
        'loss': total_loss / n,
        'acc': float((all_preds == all_labels).mean()),
        'f1': float(f1_score(all_labels, all_preds, zero_division=0)),
        'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, zero_division=0)),
    }


def train_pointnet(model, train_loader, val_loader, train_dataset,
                   epochs=50, lr=5e-4, patience=15):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_f1 = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for cloud1, cloud2, labels in train_loader:
            cloud1, cloud2, labels = cloud1.to(DEVICE), cloud2.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(cloud1, cloud2)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(labels)
        train_loss /= len(train_dataset)

        val_metrics = _evaluate_siamese(model, val_loader)

        if epoch % 5 == 0 or epoch <= 3:
            logging.info(
                f"  [point_cloud] Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, val_f1={val_metrics['f1']:.4f}, "
                f"val_acc={val_metrics['acc']:.4f}"
            )

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"  [point_cloud] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logging.info(f"  [point_cloud] Best val F1: {best_val_f1:.4f}")
    return model


def extract_pointnet_embeddings(model: SiamesePointNet, pairs: List[Tuple],
                                all_cands: Dict, all_index: Dict,
                                n_points: int, batch_size: int = 32) -> np.ndarray:
    """Extract embeddings from the Siamese PointNet."""
    model.eval()
    dataset = PointCloudPairDataset(pairs, all_cands, all_index, n_points)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_hidden = []
    with torch.no_grad():
        for cloud1, cloud2, _ in loader:
            cloud1, cloud2 = cloud1.to(DEVICE), cloud2.to(DEVICE)
            hidden = model.extract_hidden(cloud1, cloud2)
            all_hidden.append(hidden.cpu().numpy())

    return np.concatenate(all_hidden, axis=0)


# ---------------------------------------------------------------------------
# Normals branch: MLP on surface normal histograms
# ---------------------------------------------------------------------------

def train_normals_mlp(model: NormalsMLP, train_x: np.ndarray, train_y: np.ndarray,
                      val_x: np.ndarray, val_y: np.ndarray,
                      epochs: int = 200, batch_size: int = 64, lr: float = 1e-3,
                      modality_name: str = 'normals') -> NormalsMLP:

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(torch.tensor(train_x), torch.tensor(train_y, dtype=torch.float32))
    val_ds = TensorDataset(torch.tensor(val_x), torch.tensor(val_y, dtype=torch.float32))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 10

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                val_loss += criterion(out, yb).item() * len(xb)
                preds = (torch.sigmoid(out) > 0.5).long()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.long().cpu().numpy())
        val_loss /= len(val_ds)
        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        val_acc = (all_preds == all_labels).mean()
        val_f1 = f1_score(all_labels, all_preds)

        if epoch % 10 == 0 or epoch <= 5:
            logging.info(f"  [{modality_name}] Epoch {epoch:3d}: "
                        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                        f"val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"  [{modality_name}] Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Shared evaluation & embedding extraction for MLP-based branches
# ---------------------------------------------------------------------------

def evaluate_mlp(model, features: np.ndarray, labels: np.ndarray,
                 modality_name: str, split_name: str = 'test'):
    model.eval()
    model.to(DEVICE)
    x = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    y = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        out = model(x)
        loss = criterion(out, y).item()
        preds = (torch.sigmoid(out) > 0.5).long().cpu().numpy()

    y_np = labels.astype(int)
    acc = (preds == y_np).mean()
    f1 = f1_score(y_np, preds)
    pr = precision_score(y_np, preds, zero_division=0)
    re = recall_score(y_np, preds, zero_division=0)
    logging.info(f"  [{modality_name}] {split_name.upper()}: "
                 f"loss={loss:.4f}, acc={acc:.4f}, f1={f1:.4f}, pr={pr:.4f}, re={re:.4f}")
    return {'loss': loss, 'acc': acc, 'f1': f1, 'precision': pr, 'recall': re}


def extract_mlp_embeddings(model, features: np.ndarray) -> np.ndarray:
    model.eval()
    model.to(DEVICE)
    x = torch.tensor(features, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        hidden = model.extract_hidden(x)
    return hidden.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dataset_name = 'Hague'
    dataset_size = 'small'
    neg_samples = 2
    matching_cands_generation = 'blocking-based'
    normalization = hague_config.Features.normalization  # "log_transform"
    seed = 1
    vit_model_name = 'ViT-B_32'
    clip_dim = CLIP_EMBEDDING_DIMS[vit_model_name]
    val_ratio = 0.2
    image_epochs = 8
    image_batch_size = 4
    image_lr = 1e-5
    pointnet_epochs = 50
    pointnet_lr = 5e-4
    pointnet_patience = 15
    pointnet_batch_size = 32
    normals_epochs = 200
    normals_lr = 1e-3
    normals_batch_size = 64

    experiment_dir = os.path.abspath(setup_experiment_dir("hague_embedding_branches"))
    logging.info("=== Hague Embedding Branches (4 modalities) ===")
    logging.info(f"Dataset: {dataset_name}, Size: {dataset_size}, Seed: {seed}")
    logging.info(f"Modalities: image, feature_vector, point_cloud, normals")
    logging.info(f"Device: {DEVICE}")
    logging.info(f"Results dir: {experiment_dir}")

    logging.info("=" * 50)
    logging.info("STEP 1: Loading Hague data")
    logging.info("=" * 50)

    train_pairs, test_pairs = load_hague_data(
        dataset_name, seed, dataset_size, neg_samples, matching_cands_generation
    )
    logging.info(f"Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")

    train_object_dict, test_object_dict = load_object_dicts(
        dataset_name, seed, dataset_size, neg_samples, matching_cands_generation
    )

    all_cands = {**train_object_dict['cands'], **test_object_dict['cands']}
    all_index = {**train_object_dict['index'], **test_object_dict['index']}

    logging.info("=" * 50)
    logging.info("STEP 2: Rendering 3D meshes to PNG images")
    logging.info("=" * 50)

    cands_figs_path, index_figs_path = generate_png_images(
        {**{'cands': all_cands, 'index': all_index}}, dataset_name
    )
    logging.info(f"Cands images: {cands_figs_path}")
    logging.info(f"Index images: {index_figs_path}")

    train_pairs = filter_pairs_with_images(train_pairs, cands_figs_path, index_figs_path)
    test_pairs = filter_pairs_with_images(test_pairs, cands_figs_path, index_figs_path)
    logging.info(f"Pairs with valid images - Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    logging.info("=" * 50)
    logging.info("STEP 3: Computing geometric properties")
    logging.info("=" * 50)

    train_property_dict = compute_property_dict(train_object_dict, normalization)
    test_property_dict = compute_property_dict(test_object_dict, normalization)

    combined_property_dict = {}
    for prop_name in train_property_dict:
        combined_property_dict[prop_name] = {'cands': {}, 'index': {}}
        for obj_type in ['cands', 'index']:
            combined_property_dict[prop_name][obj_type].update(train_property_dict[prop_name][obj_type])
            combined_property_dict[prop_name][obj_type].update(test_property_dict[prop_name][obj_type])

    logging.info(f"Properties computed: {len(combined_property_dict)} features")

    logging.info("=" * 50)
    logging.info("STEP 4: Building pair features and splitting data")
    logging.info("=" * 50)

    train_fv_data = build_feature_vector_pairs(train_pairs, combined_property_dict)
    test_fv_data = build_feature_vector_pairs(test_pairs, combined_property_dict)
    fv_input_dim = train_fv_data['feature_vector_pair_features'].shape[1]
    logging.info(f"Feature vector dim: {fv_input_dim}")

    n_train_orig = len(train_fv_data['labels'])
    indices = np.arange(n_train_orig)
    train_idx, val_idx = train_test_split(
        indices, test_size=val_ratio, random_state=seed, stratify=train_fv_data['labels']
    )

    train_pairs_arr = np.array(train_pairs, dtype=object)
    split_train_pairs = [tuple(p) for p in train_pairs_arr[train_idx]]
    split_val_pairs = [tuple(p) for p in train_pairs_arr[val_idx]]

    fv_splits = {
        'train': {
            'features': train_fv_data['feature_vector_pair_features'][train_idx],
            'labels': train_fv_data['labels'][train_idx],
        },
        'val': {
            'features': train_fv_data['feature_vector_pair_features'][val_idx],
            'labels': train_fv_data['labels'][val_idx],
        },
        'test': {
            'features': test_fv_data['feature_vector_pair_features'],
            'labels': test_fv_data['labels'],
        }
    }

    logging.info("Building normal histogram pair features...")
    train_norm_x, train_norm_y = build_normal_pair_features(train_pairs, all_cands, all_index)
    test_norm_x, test_norm_y = build_normal_pair_features(test_pairs, all_cands, all_index)
    norm_input_dim = train_norm_x.shape[1]
    logging.info(f"Normal histogram dim: {norm_input_dim}")

    norm_splits = {
        'train': {'features': train_norm_x[train_idx], 'labels': train_norm_y[train_idx]},
        'val':   {'features': train_norm_x[val_idx],   'labels': train_norm_y[val_idx]},
        'test':  {'features': test_norm_x,             'labels': test_norm_y},
    }

    logging.info("Computing point cloud statistics...")
    vert_counts = []
    for obj_dict in [all_cands, all_index]:
        for obj in obj_dict.values():
            vert_counts.append(len(_get_vertices(obj)))
    vert_counts = np.array(vert_counts)
    n_points = int(np.percentile(vert_counts, 90))
    logging.info(
        f"  Vertex counts: min={vert_counts.min()}, median={int(np.median(vert_counts))}, "
        f"p90={n_points}, max={vert_counts.max()}"
    )

    for s_name, s_data in fv_splits.items():
        n = len(s_data['labels'])
        n_pos = s_data['labels'].sum()
        logging.info(f"  {s_name.upper()}: {n} pairs ({n_pos} positive, {n_pos/n:.2%})")

    logging.info("=" * 50)
    logging.info("STEP 5a: Training image branch (contrastive CLIP fine-tuning)")
    logging.info(f"  Architecture: CLIP {vit_model_name} -> |emb1-emb2| -> Linear({clip_dim},{HIDDEN_DIM}) -> ReLU -> Linear({HIDDEN_DIM},1)")
    logging.info(f"  Epochs: {image_epochs}, Batch size: {image_batch_size}, LR: {image_lr}")
    logging.info("=" * 50)

    contrastive_model, preprocess = train_contrastive_image_model(
        vit_model_name, clip_dim,
        split_train_pairs, split_val_pairs,
        cands_figs_path, index_figs_path,
        epochs=image_epochs, batch_size=image_batch_size, lr=image_lr
    )

    logging.info("Evaluating image branch on test set...")
    test_img_dataset = ContrastivePairDataset(test_pairs, cands_figs_path, index_figs_path, preprocess)
    test_img_loader = DataLoader(test_img_dataset, batch_size=image_batch_size, shuffle=False, num_workers=0)
    image_test_metrics = _evaluate_contrastive(contrastive_model, test_img_loader)
    logging.info(
        f"  [image] TEST: loss={image_test_metrics['loss']:.4f}, acc={image_test_metrics['acc']:.4f}, "
        f"f1={image_test_metrics['f1']:.4f}, pr={image_test_metrics['precision']:.4f}, re={image_test_metrics['recall']:.4f}"
    )

    image_model_path = os.path.join(experiment_dir, "image_contrastive_model.pt")
    torch.save(contrastive_model.state_dict(), image_model_path)
    logging.info(f"  Image model saved to: {image_model_path}")
    logging.info("=" * 50)
    logging.info("STEP 5b: Training feature_vector branch (MLP with BatchNorm)")
    logging.info(f"  Architecture: Linear({fv_input_dim},512) -> BN -> ReLU -> Linear(512,128) -> BN -> ReLU -> Linear(128,{HIDDEN_DIM}) -> BN -> ReLU -> Linear({HIDDEN_DIM},1)")
    logging.info("=" * 50)

    fv_model = FeatureVectorMLP(fv_input_dim, HIDDEN_DIM)
    fv_model = train_fv_mlp(
        fv_model,
        fv_splits['train']['features'], fv_splits['train']['labels'],
        fv_splits['val']['features'], fv_splits['val']['labels'],
        epochs=500, batch_size=64, lr=1e-4,
        modality_name='feature_vector'
    )

    evaluate_mlp(fv_model, fv_splits['val']['features'], fv_splits['val']['labels'], 'feature_vector', 'val')
    fv_test_metrics = evaluate_mlp(fv_model, fv_splits['test']['features'], fv_splits['test']['labels'], 'feature_vector', 'test')

    fv_model_path = os.path.join(experiment_dir, "feature_vector_mlp.pt")
    torch.save(fv_model.state_dict(), fv_model_path)
    logging.info(f"  Feature vector model saved to: {fv_model_path}")

    logging.info("=" * 50)
    logging.info("STEP 5c: Training point_cloud branch (Siamese PointNet)")
    logging.info(f"  Architecture: PointNet({n_points} pts) -> {HIDDEN_DIM}-dim -> |emb1-emb2| -> classifier")
    logging.info(f"  Epochs: {pointnet_epochs}, LR: {pointnet_lr}, Patience: {pointnet_patience}")
    logging.info("=" * 50)

    train_pc_ds = PointCloudPairDataset(split_train_pairs, all_cands, all_index, n_points)
    val_pc_ds = PointCloudPairDataset(split_val_pairs, all_cands, all_index, n_points)
    test_pc_ds = PointCloudPairDataset(test_pairs, all_cands, all_index, n_points)

    train_pc_loader = DataLoader(train_pc_ds, batch_size=pointnet_batch_size, shuffle=True, num_workers=0)
    val_pc_loader = DataLoader(val_pc_ds, batch_size=pointnet_batch_size, shuffle=False, num_workers=0)
    test_pc_loader = DataLoader(test_pc_ds, batch_size=pointnet_batch_size, shuffle=False, num_workers=0)

    pointnet_model = SiamesePointNet(n_points, HIDDEN_DIM)
    logging.info(f"  PointNet params: {sum(p.numel() for p in pointnet_model.parameters()):,}")

    pointnet_model = train_pointnet(
        pointnet_model, train_pc_loader, val_pc_loader, train_pc_ds,
        epochs=pointnet_epochs, lr=pointnet_lr, patience=pointnet_patience
    )

    logging.info("Evaluating point_cloud branch on test set...")
    pc_test_metrics = _evaluate_siamese(pointnet_model, test_pc_loader)
    logging.info(
        f"  [point_cloud] TEST: loss={pc_test_metrics['loss']:.4f}, acc={pc_test_metrics['acc']:.4f}, "
        f"f1={pc_test_metrics['f1']:.4f}, pr={pc_test_metrics['precision']:.4f}, re={pc_test_metrics['recall']:.4f}"
    )

    pc_model_path = os.path.join(experiment_dir, "pointnet_model.pt")
    torch.save(pointnet_model.state_dict(), pc_model_path)
    logging.info(f"  PointNet model saved to: {pc_model_path}")

    logging.info("=" * 50)
    logging.info("STEP 5d: Training normals branch (MLP on normal histograms)")
    logging.info(f"  Architecture: |h1-h2| -> Linear({norm_input_dim},{HIDDEN_DIM}) -> ReLU -> Linear({HIDDEN_DIM},1)")
    logging.info(f"  Epochs: {normals_epochs}, LR: {normals_lr}")
    logging.info("=" * 50)

    normals_model = NormalsMLP(norm_input_dim, HIDDEN_DIM)
    logging.info(f"  Normals MLP params: {sum(p.numel() for p in normals_model.parameters()):,}")

    normals_model = train_normals_mlp(
        normals_model,
        norm_splits['train']['features'], norm_splits['train']['labels'],
        norm_splits['val']['features'], norm_splits['val']['labels'],
        epochs=normals_epochs, batch_size=normals_batch_size, lr=normals_lr,
        modality_name='normals'
    )

    evaluate_mlp(normals_model, norm_splits['val']['features'], norm_splits['val']['labels'], 'normals', 'val')
    normals_test_metrics = evaluate_mlp(normals_model, norm_splits['test']['features'], norm_splits['test']['labels'], 'normals', 'test')

    normals_model_path = os.path.join(experiment_dir, "normals_mlp_model.pt")
    torch.save(normals_model.state_dict(), normals_model_path)
    logging.info(f"  Normals model saved to: {normals_model_path}")

    logging.info("=" * 50)
    logging.info("STEP 6: Extracting 256-dim hidden embeddings (4 modalities)")
    logging.info("=" * 50)

    all_embeddings_data = {}
    for split_name, pairs, fv_data, norm_data in [
        ('train', split_train_pairs, fv_splits['train'], norm_splits['train']),
        ('val', split_val_pairs, fv_splits['val'], norm_splits['val']),
        ('test', test_pairs, fv_splits['test'], norm_splits['test']),
    ]:
        image_embs = extract_contrastive_embeddings(
            contrastive_model, pairs, cands_figs_path, index_figs_path, preprocess
        )
        fv_embs = extract_mlp_embeddings(fv_model, fv_data['features'])
        pc_embs = extract_pointnet_embeddings(
            pointnet_model, pairs, all_cands, all_index, n_points
        )
        norm_embs = extract_mlp_embeddings(normals_model, norm_data['features'])

        all_embeddings_data[split_name] = {
            'labels': fv_data['labels'],
            'image_hidden_embeddings': image_embs,
            'feature_vector_hidden_embeddings': fv_embs,
            'point_cloud_hidden_embeddings': pc_embs,
            'normals_hidden_embeddings': norm_embs,
        }
        logging.info(
            f"  {split_name.upper()}: image={image_embs.shape}, feature_vector={fv_embs.shape}, "
            f"point_cloud={pc_embs.shape}, normals={norm_embs.shape}"
        )

    logging.info("=" * 50)
    logging.info("STEP 7: Saving results")
    logging.info("=" * 50)

    embeddings_path = os.path.join(experiment_dir, "intermediate_embeddings.pkl")
    with open(embeddings_path, 'wb') as f:
        pickle.dump(all_embeddings_data, f)
    logging.info(f"Intermediate embeddings saved to: {embeddings_path}")

    total_samples = sum(len(s['labels']) for s in all_embeddings_data.values())

    config_data = {
        'experiment_type': 'hague_embedding_branches',
        'dataset': dataset_name,
        'modalities': ['image', 'feature_vector', 'point_cloud', 'normals'],
        'image_branch': {
            'type': 'contrastive_clip_finetuning',
            'clip_model': vit_model_name,
            'clip_dim': clip_dim,
            'hidden_dim': HIDDEN_DIM,
            'pair_method': 'absolute_difference',
            'epochs': image_epochs,
            'batch_size': image_batch_size,
            'lr': image_lr,
            'test_metrics': {k: float(v) for k, v in image_test_metrics.items()},
        },
        'feature_vector_branch': {
            'type': 'mlp_3layer_batchnorm',
            'architecture': f'{fv_input_dim} -> 512 -> BN -> 128 -> BN -> {HIDDEN_DIM} -> BN -> 1',
            'properties': hague_config.Features.object_properties,
            'input_dim': fv_input_dim,
            'hidden_dim': HIDDEN_DIM,
            'lr': 1e-4,
            'test_metrics': {k: float(v) for k, v in fv_test_metrics.items()},
        },
        'point_cloud_branch': {
            'type': 'siamese_pointnet',
            'n_points': n_points,
            'hidden_dim': HIDDEN_DIM,
            'pair_method': 'absolute_difference',
            'epochs': pointnet_epochs,
            'lr': pointnet_lr,
            'patience': pointnet_patience,
            'batch_size': pointnet_batch_size,
            'test_metrics': {k: float(v) for k, v in pc_test_metrics.items()},
        },
        'normals_branch': {
            'type': 'mlp_on_normal_histograms',
            'architecture': f'{norm_input_dim} -> {HIDDEN_DIM} -> 1',
            'pair_method': 'absolute_difference_of_histograms',
            'input_dim': norm_input_dim,
            'hidden_dim': HIDDEN_DIM,
            'epochs': normals_epochs,
            'lr': normals_lr,
            'batch_size': normals_batch_size,
            'test_metrics': {k: float(v) for k, v in normals_test_metrics.items()},
        },
        'seed': seed,
        'dataset_size': dataset_size,
        'neg_samples': neg_samples,
        'embeddings': {
            'embeddings_path': embeddings_path,
            'train_samples': len(all_embeddings_data['train']['labels']),
            'val_samples': len(all_embeddings_data['val']['labels']),
            'test_samples': len(all_embeddings_data['test']['labels']),
            'total_samples': total_samples,
            'modality_shapes': {
                mod: str(all_embeddings_data['test'][f'{mod}_hidden_embeddings'].shape)
                for mod in ['image', 'feature_vector', 'point_cloud', 'normals']
            },
        }
    }

    config_path = os.path.join(experiment_dir, "config_and_results.json")
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    logging.info(f"Config saved to: {config_path}")

    logging.info("=" * 50)
    logging.info("Hague embedding branches completed! (4 modalities)")
    logging.info(f"Next step: python run_gnn.py --embedding-branches-dir {experiment_dir}")
    logging.info("=" * 50)


if __name__ == "__main__":
    main()
