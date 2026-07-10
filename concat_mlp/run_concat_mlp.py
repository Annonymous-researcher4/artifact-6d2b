#!/usr/bin/env python3

import os
import argparse
import pickle
import logging
import json
import numpy as np
from unimodal_pair_encoders.utils import setup_experiment_dir

from sklearn.metrics import f1_score, precision_score, recall_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class IntermediateFusionMLP(nn.Module):

    def __init__(self, n_modalities, emb_dim=256, dropout=0.3):
        super().__init__()
        input_dim = n_modalities * emb_dim
        hidden_dim = input_dim
        bottleneck_dim = input_dim // 2

        self.net = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_embeddings(embedding_branches_dir):
    config_path = os.path.join(embedding_branches_dir, "config_and_results.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    embeddings_path = os.path.join(embedding_branches_dir, "intermediate_embeddings.pkl")
    with open(embeddings_path, 'rb') as f:
        all_embeddings = pickle.load(f)

    modalities = config['modalities']
    return all_embeddings, modalities, config


def build_split_tensors(all_embeddings, modalities, split_name):
    emb_list = [all_embeddings[split_name][f'{mod}_hidden_embeddings'] for mod in modalities]
    x = np.concatenate(emb_list, axis=1)
    y = all_embeddings[split_name]['labels']
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def evaluate(model, dataloader):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    loss_fn = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for xb, yb in dataloader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            out = model(xb)
            total_loss += loss_fn(out, yb).item() * len(yb)
            preds = (torch.sigmoid(out) > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.cpu().numpy().astype(int))

    all_preds, all_labels = np.array(all_preds), np.array(all_labels)
    n = len(all_labels)
    return {
        'loss': total_loss / n,
        'acc': float((all_preds == all_labels).mean()),
        'f1': float(f1_score(all_labels, all_preds, zero_division=0)),
        'precision': float(precision_score(all_labels, all_preds, zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, zero_division=0)),
    }


def train_fusion(model, train_loader, val_loader, train_size,
                 epochs, lr, patience):
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_f1 = -1
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= train_size

        val_metrics = evaluate(model, val_loader)

        if epoch % 5 == 0 or epoch <= 3:
            logging.info(
                f"  Epoch {epoch:3d}: train_loss={train_loss:.4f}, "
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
                logging.info(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logging.info(f"  Best val F1: {best_val_f1:.4f}")
    return model


def run_pytorch_fusion(args):
    experiment_dir = setup_experiment_dir("intermediate_fusion")
    logging.info("=== Intermediate Fusion (PyTorch) ===")
    logging.info(f"Results dir: {experiment_dir}")
    logging.info(f"Loading from: {args.embedding_branches_dir}")

    all_embeddings, modalities, base_config = load_embeddings(args.embedding_branches_dir)
    n_modalities = len(modalities)
    input_dim = n_modalities * 256

    logging.info(f"Dataset: {base_config['dataset']}")
    logging.info(f"Modalities ({n_modalities}): {modalities}")
    logging.info(f"Input dim: {input_dim} ({n_modalities} × 256)")
    logging.info(f"Architecture: {input_dim} → BN → {input_dim} → {input_dim} → {input_dim // 2} → 1")
    logging.info(f"Device: {DEVICE}")

    train_x, train_y = build_split_tensors(all_embeddings, modalities, 'train')
    val_x, val_y = build_split_tensors(all_embeddings, modalities, 'val')
    test_x, test_y = build_split_tensors(all_embeddings, modalities, 'test')

    batch_size = 32
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=batch_size)

    for name, y in [('train', train_y), ('val', val_y), ('test', test_y)]:
        n_pos = int(y.sum())
        logging.info(f"  {name.upper()}: {len(y)} pairs ({n_pos} positive, {n_pos/len(y):.2%})")

    model = IntermediateFusionMLP(n_modalities)
    logging.info(f"Fusion model params: {sum(p.numel() for p in model.parameters()):,}")

    logging.info("=" * 50)
    logging.info("Training Intermediate Fusion")
    logging.info("=" * 50)

    model = train_fusion(
        model, train_loader, val_loader, len(train_x),
        epochs=args.fusion_epochs, lr=1e-3, patience=10
    )

    logging.info("=" * 50)
    logging.info("Final Evaluation (TEST)")
    logging.info("=" * 50)

    test_metrics = evaluate(model, test_loader)
    logging.info(
        f"  TEST: loss={test_metrics['loss']:.4f}, acc={test_metrics['acc']:.4f}, "
        f"f1={test_metrics['f1']:.4f}, pr={test_metrics['precision']:.4f}, "
        f"re={test_metrics['recall']:.4f}"
    )

    model_path = os.path.join(experiment_dir, "intermediate_fusion_model.pt")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model saved to: {model_path}")

    results_summary = {
        'experiment_type': 'intermediate_fusion',
        'base_experiment_dir': args.embedding_branches_dir,
        'dataset': base_config['dataset'],
        'modalities': modalities,
        'n_modalities': n_modalities,
        'architecture': {
            'input_dim': input_dim,
            'layers': [input_dim, input_dim, input_dim // 2, 1],
            'dropout': 0.3,
        },
        'training': {
            'epochs': args.fusion_epochs,
            'lr': 1e-3,
            'batch_size': 32,
            'patience': 10,
        },
        'test_metrics': test_metrics,
        'model_path': model_path,
    }

    summary_path = os.path.join(experiment_dir, "fusion_results_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    logging.info(f"Results saved to: {summary_path}")

    logging.info("=" * 50)
    logging.info("INTERMEDIATE FUSION COMPLETED!")
    logging.info("=" * 50)


def run_tensorflow_fusion(args):
    import tensorflow as tf
    from transformers import TFCLIPVisionModel
    from unimodal_pair_encoders.bimodal import data_preparation
    from unimodal_pair_encoders import models_mps as models
    from unimodal_pair_encoders.tf_utils import LoggingCallback, format_results

    def train_intermediate_fusion_with_logging(model, data_id, train_batches, val_batches, epochs):
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss="BinaryCrossentropy", metrics=models.metrics)
        callbacks = [models.callback, LoggingCallback()]
        model.fit(
            train_batches,
            validation_data=val_batches,
            epochs=epochs,
            use_multiprocessing=True,
            workers=-1,
            class_weight=data_preparation.get_class_weights(data_id),
            callbacks=callbacks,
            verbose=0
        )
        return model

    def load_pretrained_models(experiment_dir):
        config_path = os.path.join(experiment_dir, "config_and_results.json")
        with open(config_path, 'r') as f:
            config = json.load(f)

        logging.info(f"Dataset: {config['dataset']}")
        logging.info(f"Text Embedding: {config['text_embedding']}")
        logging.info(f"Image Embedding: {config['image_embedding']}")

        fine_tuned_path = config['model_paths']['fine_tuned_model']
        siamese_path = config['model_paths']['siamese_model']

        if not os.path.exists(fine_tuned_path):
            raise FileNotFoundError(f"Fine-tuned model not found: {fine_tuned_path}")
        if not os.path.exists(siamese_path):
            raise FileNotFoundError(f"Siamese model not found: {siamese_path}")

        logging.info(f"Loading fine-tuned model from: {fine_tuned_path}")
        fine_tuned_model = tf.keras.models.load_model(fine_tuned_path, compile=False)

        logging.info(f"Loading siamese model from: {siamese_path}")
        use_clip = config.get('use_clip', False)
        if use_clip:
            siamese_model = tf.keras.models.load_model(
                siamese_path,
                custom_objects={"TFCLIPVisionModel": TFCLIPVisionModel},
                compile=False
            )
        else:
            siamese_model = tf.keras.models.load_model(siamese_path, compile=False)

        for layer in fine_tuned_model.layers:
            layer._name = 'text_' + layer.name
        for layer in siamese_model.layers:
            layer._name = 'image_' + layer.name

        return fine_tuned_model, siamese_model, config

    experiment_dir = setup_experiment_dir("intermediate_fusion")
    logging.info("=== Intermediate Fusion Model Training (TensorFlow) ===")
    logging.info(f"Results will be saved to: {experiment_dir}")

    fine_tuned_model, siamese_model, config = load_pretrained_models(args.embedding_branches_dir)

    data_id = config['dataset']
    text_embedding_id = config['text_embedding']
    fusion_epochs = args.fusion_epochs

    train_batches_fusion, val_batches_fusion, test_batches_fusion = data_preparation.get_batches(
        data_id, "Intermediate Fusion", text_embedding_id
    )

    intermediate_fusion_model = models.build_intermediate_fusion_model(
        fine_tuned_model, siamese_model
    )

    intermediate_fusion_model = train_intermediate_fusion_with_logging(
        intermediate_fusion_model, data_id, train_batches_fusion, val_batches_fusion, fusion_epochs
    )

    fusion_results = intermediate_fusion_model.evaluate(test_batches_fusion, verbose=1)

    logging.info("=== Intermediate Fusion Model Test Results ===")
    if isinstance(fusion_results, list):
        for name, value in zip(intermediate_fusion_model.metrics_names, fusion_results):
            value_float = float(value.numpy()) if hasattr(value, 'numpy') else float(value)
            logging.info(f"  {name}: {value_float:.4f}")

    fusion_path = os.path.join(experiment_dir, f"intermediate_fusion_model_{data_id.lower().replace(' ', '_')}.h5")
    intermediate_fusion_model.save(fusion_path)
    logging.info(f"Fusion model saved to: {fusion_path}")

    results_summary = {
        'experiment_type': 'intermediate_fusion',
        'base_experiment_dir': args.embedding_branches_dir,
        'dataset': data_id,
        'text_embedding': text_embedding_id,
        'image_embedding': config['image_embedding'],
        'fusion_epochs': fusion_epochs,
        'pretrained_models': {
            'fine_tuned_model': config['model_paths']['fine_tuned_model'],
            'siamese_model': config['model_paths']['siamese_model']
        },
        'pretrained_test_results': config['test_results'],
        'fusion_test_results': format_results(fusion_results, intermediate_fusion_model),
        'fusion_model_path': fusion_path
    }

    summary_path = os.path.join(experiment_dir, "fusion_results_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    logging.info(f"Results saved to: {summary_path}")

    logging.info("=" * 50)
    logging.info("INTERMEDIATE FUSION COMPLETED!")
    logging.info("=" * 50)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding-branches-dir', type=str, required=True)
    parser.add_argument('--fusion-epochs', type=int, default=40)
    args = parser.parse_args()

    config_path = os.path.join(args.embedding_branches_dir, "config_and_results.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    experiment_type = config.get('experiment_type', '')

    if experiment_type == 'hague_embedding_branches':
        run_pytorch_fusion(args)
    else:
        run_tensorflow_fusion(args)


if __name__ == "__main__":
    main()
