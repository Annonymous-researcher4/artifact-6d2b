import logging
import numpy as np
import tensorflow as tf
from unimodal_pair_encoders.utils import setup_experiment_dir


class LoggingCallback(tf.keras.callbacks.Callback):

    def on_epoch_begin(self, epoch, logs=None):
        logging.info(f"Epoch {epoch + 1}/{self.params['epochs']} - Starting...")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        log_msg = f"Epoch {epoch + 1}/{self.params['epochs']} - "

        def to_float(value):
            if hasattr(value, 'numpy'):
                return float(value.numpy())
            elif isinstance(value, np.ndarray):
                return float(value)
            else:
                return float(value)

        metric_parts = []
        if 'loss' in logs:
            metric_parts.append(f"loss: {to_float(logs['loss']):.4f}")
        if 'accuracy' in logs:
            metric_parts.append(f"accuracy: {to_float(logs['accuracy']):.4f}")
        if 'precision' in logs:
            metric_parts.append(f"precision: {to_float(logs['precision']):.4f}")
        if 'recall' in logs:
            metric_parts.append(f"recall: {to_float(logs['recall']):.4f}")
        if 'f1_score' in logs:
            metric_parts.append(f"f1_score: {to_float(logs['f1_score']):.4f}")

        val_parts = []
        if 'val_loss' in logs:
            val_parts.append(f"val_loss: {to_float(logs['val_loss']):.4f}")
        if 'val_accuracy' in logs:
            val_parts.append(f"val_accuracy: {to_float(logs['val_accuracy']):.4f}")
        if 'val_precision' in logs:
            val_parts.append(f"val_precision: {to_float(logs['val_precision']):.4f}")
        if 'val_recall' in logs:
            val_parts.append(f"val_recall: {to_float(logs['val_recall']):.4f}")
        if 'val_f1_score' in logs:
            val_parts.append(f"val_f1_score: {to_float(logs['val_f1_score']):.4f}")

        log_msg += " - ".join(metric_parts)
        if val_parts:
            log_msg += " | " + " - ".join(val_parts)

        logging.info(log_msg)

    def on_train_begin(self, logs=None):
        logging.info("Training started...")

    def on_train_end(self, logs=None):
        logging.info("Training completed!")


def extract_intermediate_embeddings(fine_tuned_model, siamese_model, fusion_batches):
    logging.info("Extracting intermediate embeddings before fusion...")

    fine_tuned_model.trainable = False
    siamese_model.trainable = False

    text_hidden_layer = fine_tuned_model.get_layer("text_hidden_layer")
    image_hidden_layer = siamese_model.get_layer("image_hidden_layer")

    logging.info(f"Found text hidden layer with {text_hidden_layer.units} units: {text_hidden_layer.name}")
    logging.info(f"Found image hidden layer with {image_hidden_layer.units} units: {image_hidden_layer.name}")

    text_hidden_features = text_hidden_layer.output
    image_hidden_features = image_hidden_layer.output

    text_embedding_extractor = tf.keras.Model(
        inputs=fine_tuned_model.input,
        outputs=text_hidden_features
    )

    image_embedding_extractor = tf.keras.Model(
        inputs=siamese_model.input,
        outputs=image_hidden_features
    )

    all_text_hidden_embeddings = []
    all_image_hidden_embeddings = []
    all_labels = []

    logging.info("Extracting embeddings from batches...")
    for i, batch in enumerate(fusion_batches):
        if i % 10 == 0:
            logging.info(f"Processing batch {i+1}/{len(fusion_batches)}")

        inputs, labels = batch
        input_ids, attention_masks, token_type_ids, image_1, image_2 = inputs

        text_hidden = text_embedding_extractor([input_ids, attention_masks, token_type_ids])
        text_hidden_np = text_hidden.numpy() if hasattr(text_hidden, 'numpy') else text_hidden
        all_text_hidden_embeddings.append(text_hidden_np)

        img_hidden = image_embedding_extractor([image_1, image_2])
        img_hidden_np = img_hidden.numpy() if hasattr(img_hidden, 'numpy') else img_hidden
        all_image_hidden_embeddings.append(img_hidden_np)

        labels_np = labels.numpy() if hasattr(labels, 'numpy') else labels
        all_labels.append(labels_np)

    text_hidden_embeddings = np.concatenate(all_text_hidden_embeddings, axis=0)
    image_hidden_embeddings = np.concatenate(all_image_hidden_embeddings, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    embeddings_data = {
        'text_hidden_embeddings': text_hidden_embeddings,
        'image_hidden_embeddings': image_hidden_embeddings,
        'labels': labels,
        'metadata': {
            'text_hidden_shape': text_hidden_embeddings.shape,
            'image_hidden_shape': image_hidden_embeddings.shape,
            'total_samples': len(labels)
        }
    }

    logging.info(f"Intermediate embeddings extracted successfully")
    logging.info(f"  Text hidden embeddings: {text_hidden_embeddings.shape}")
    logging.info(f"  Image hidden embeddings: {image_hidden_embeddings.shape}")
    logging.info(f"  Total samples: {len(labels)}")

    return embeddings_data


def format_results(results, model):
    if isinstance(results, list):
        results_dict = {}
        for name, value in zip(model.metrics_names, results):
            if isinstance(value, np.ndarray):
                results_dict[name] = float(value)
            elif hasattr(value, 'numpy'):
                results_dict[name] = float(value.numpy())
            else:
                results_dict[name] = float(value)
        return results_dict
    else:
        if isinstance(results, np.ndarray):
            return float(results)
        elif hasattr(results, 'numpy'):
            return float(results.numpy())
        else:
            return float(results)
