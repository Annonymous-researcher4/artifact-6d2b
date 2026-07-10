#!/usr/bin/env python3
"""
Train Embedding Branches - Part 1 of Intermediate Fusion
Trains and saves the Fine-tuned (text) and Private Siamese (image) models
"""

import os
import argparse
import logging
import numpy as np
import json
import tensorflow as tf
from datetime import datetime
from unimodal_pair_encoders.bimodal import data_preparation
from unimodal_pair_encoders import models_mps as models  # MPS-enabled variant
from unimodal_pair_encoders.tf_utils import (LoggingCallback, setup_experiment_dir,
                      extract_intermediate_embeddings, format_results)


def train_fine_tuned_with_logging(model, data_id, train_batches, val_batches, epochs, unfreeze=False):
    if not unfreeze:
        learning_rate = 1e-3
    else:
        model.trainable = True
        learning_rate = 1e-5

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss="BinaryCrossentropy", metrics=models.metrics)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 1: train the text (RoBERTa) and image (CLIP ViT-B/32) embedding "
                    "branches for WDC Shoes / Zalando and export intermediate embeddings.")
    parser.add_argument('--dataset', type=str, required=True, choices=['WDC Shoes', 'Zalando'],
                        help='Dataset to train on.')
    parser.add_argument('--text-epochs', type=int, default=40,
                        help='Epochs per phase for the text branch (frozen + unfrozen).')
    parser.add_argument('--contrastive-epochs', type=int, default=12,
                        help='Epochs for CLIP contrastive fine-tuning (image branch, stage 1).')
    parser.add_argument('--classifier-epochs', type=int, default=60,
                        help='Epochs for the image classifier head (image branch, stage 2).')
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_dir = setup_experiment_dir("embedding_branches")
    logging.info("=== Training Embedding Branches ===")
    logging.info(f"Results will be saved to: {experiment_dir}")

    data_id = args.dataset
    text_embedding_id = "RoBERTa"

    # Image branch: CLIP two-stage (contrastive fine-tuning + classifier head).
    image_embedding_id = "ViT-B/32"
    use_clip = True

    text_epochs = args.text_epochs
    contrastive_epochs = args.contrastive_epochs
    classifier_epochs = args.classifier_epochs

    logging.info(f"Dataset: {data_id}")
    logging.info(f"Text Embedding: {text_embedding_id}")
    logging.info(f"Image Embedding: {image_embedding_id}")

    if use_clip:
        logging.info(f"Image Branch Architecture: CLIP (Stage 1: contrastive fine-tuning, Stage 2: classifier)")
    else:
        logging.info(f"Image Branch Architecture: Traditional ({image_embedding_id})")

    logging.info(f"Text Model Epochs: {text_epochs}")
    logging.info(f"Contrastive Epochs: {contrastive_epochs}")
    logging.info(f"Classifier Epochs: {classifier_epochs}")

    try:
        logging.info("=" * 50)
        logging.info("STEP 1: Training Text Branch (Fine-tuned Model)")
        logging.info("=" * 50)

        logging.info("Loading text data batches...")
        train_batches_text, val_batches_text, test_batches_text = data_preparation.get_batches(
            data_id, "Fine-tuned", text_embedding_id
        )
        logging.info(f"Text data loaded")

        logging.info(f"Building Fine-tuned model with {text_embedding_id}...")
        fine_tuned_model = models.build_fine_tuned_model(text_embedding_id)

        logging.info(f"Phase 1: Training text model with frozen embeddings...")
        fine_tuned_model = train_fine_tuned_with_logging(
            fine_tuned_model, data_id, train_batches_text, val_batches_text, text_epochs, unfreeze=False
        )

        logging.info(f"Phase 2: Fine-tuning text model with unfrozen embeddings...")
        fine_tuned_model = train_fine_tuned_with_logging(
            fine_tuned_model, data_id, train_batches_text, val_batches_text, text_epochs, unfreeze=True
        )
        logging.info("Text branch training completed")

        logging.info("Evaluating Fine-tuned model on test set...")
        fine_tuned_results = fine_tuned_model.evaluate(test_batches_text, verbose=1)
        logging.info("Fine-tuned model evaluation completed")
        logging.info("=== Fine-tuned Model Test Results ===")
        if isinstance(fine_tuned_results, list):
            for name, value in zip(fine_tuned_model.metrics_names, fine_tuned_results):
                value_float = float(value.numpy()) if hasattr(value, 'numpy') else float(value)
                logging.info(f"  {name}: {value_float:.4f}")

        fine_tuned_path = os.path.join(experiment_dir, f"fine_tuned_model_{data_id.lower().replace(' ', '_')}")
        fine_tuned_model.save(fine_tuned_path, save_format='tf')
        logging.info(f"Fine-tuned model saved to: {fine_tuned_path}")

        logging.info("=" * 50)
        logging.info("STEP 2: Training Image Branch (Private Siamese Model)")
        logging.info("=" * 50)

        logging.info("Loading image data batches...")
        train_batches_image, val_batches_image, test_batches_image = data_preparation.get_batches(
            data_id, "Siamese"
        )
        logging.info(f"Image data loaded")

        if use_clip:
            logging.info(f"Building and training CLIP model with {image_embedding_id}...")
            logging.info(f"    Stage 1: Fine-tune CLIP with contrastive loss ({contrastive_epochs} epochs)")
            logging.info(f"    Stage 2: Train classifier with frozen CLIP ({classifier_epochs} epochs)")
        else:
            logging.info(f"Building and training Private Siamese model with {image_embedding_id}...")

        siamese_model = models.build_and_train_private_siamese_model(
            image_embedding_id, data_id, train_batches_image, val_batches_image,
            contrastive_epochs, classifier_epochs, use_clip
        )
        logging.info("Image branch training completed")

        logging.info("Evaluating Private Siamese model on test set...")
        siamese_results = siamese_model.evaluate(test_batches_image, verbose=1)
        logging.info("=== Private Siamese Model Test Results ===")
        if isinstance(siamese_results, list):
            for name, value in zip(siamese_model.metrics_names, siamese_results):
                value_float = float(value.numpy()) if hasattr(value, 'numpy') else float(value)
                logging.info(f"  {name}: {value_float:.4f}")

        siamese_path = os.path.join(experiment_dir, f"siamese_model_{data_id.lower().replace(' ', '_')}")
        siamese_model.save(siamese_path, save_format='tf')
        logging.info(f"Siamese model saved to: {siamese_path}")

        logging.info("=" * 50)
        logging.info("STEP 3: Extracting Intermediate Embeddings from Trained Branches")
        logging.info("=" * 50)

        logging.info("Loading fusion data batches for embedding extraction...")
        train_batches_fusion, val_batches_fusion, test_batches_fusion = data_preparation.get_batches(
            data_id, "Intermediate Fusion", text_embedding_id
        )
        logging.info(f"Fusion data loaded")

        all_embeddings_data = {}

        for split_name, split_batches in [
            ('train', train_batches_fusion),
            ('val', val_batches_fusion),
            ('test', test_batches_fusion)
        ]:
            logging.info(f"Extracting embeddings for {split_name.upper()} split...")
            split_embeddings = extract_intermediate_embeddings(
                fine_tuned_model, siamese_model, split_batches
            )
            all_embeddings_data[split_name] = split_embeddings

        import pickle
        embeddings_path = os.path.join(experiment_dir, "intermediate_embeddings.pkl")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(all_embeddings_data, f)
        logging.info(f"Intermediate embeddings saved to: {embeddings_path}")

        total_samples = sum(len(split_data['labels']) for split_data in all_embeddings_data.values())
        logging.info(f"Total embeddings extracted: {total_samples} samples across all splits")

        if use_clip:
            image_branch_type = 'clip'
            encoder_details = f'CLIP {image_embedding_id}'
        else:
            image_branch_type = 'traditional'
            encoder_details = f'{image_embedding_id}'

        training_approach = f'Stage 1: Fine-tune {encoder_details} with contrastive loss, Stage 2: Train classifier with frozen encoder'

        config_data = {
            'experiment_type': 'embedding_branches_training',
            'dataset': data_id,
            'text_embedding': text_embedding_id,
            'image_embedding': image_embedding_id,
            'image_branch_type': image_branch_type,
            'training_approach': training_approach,
            'use_clip': use_clip,
            'text_epochs': text_epochs,
            'contrastive_epochs': contrastive_epochs,
            'classifier_epochs': classifier_epochs,
            'model_paths': {
                'fine_tuned_model': fine_tuned_path,
                'siamese_model': siamese_path
            },
            'test_results': {
                'fine_tuned_model': format_results(fine_tuned_results, fine_tuned_model),
                'siamese_model': format_results(siamese_results, siamese_model)
            },
            'embeddings': {
                'embeddings_path': embeddings_path,
                'train_samples': len(all_embeddings_data['train']['labels']),
                'val_samples': len(all_embeddings_data['val']['labels']),
                'test_samples': len(all_embeddings_data['test']['labels']),
                'total_samples': total_samples,
                'text_hidden_shape': str(all_embeddings_data['test']['text_hidden_embeddings'].shape),
                'image_hidden_shape': str(all_embeddings_data['test']['image_hidden_embeddings'].shape)
            },
            'training_notes': {
                'text_model': 'Trained for 2x epochs (frozen + unfrozen phases)',
                'siamese_model': training_approach,
                'loss_functions': 'Stage 1: Cosine Embedding Loss, Stage 2: BCEWithLogitsLoss',
                'classifier_input': 'Absolute difference between embeddings'
            }
        }

        config_path = os.path.join(experiment_dir, "config_and_results.json")
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        logging.info(f"Configuration and results saved to: {config_path}")

        return {
            'fine_tuned_model': fine_tuned_model,
            'siamese_model': siamese_model,
            'experiment_dir': experiment_dir
        }

    except Exception as e:
        logging.error(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
