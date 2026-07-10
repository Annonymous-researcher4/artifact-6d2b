import pickle
import tensorflow as tf
import transformers
import numpy as np
from PIL import Image

import os

# Raw bimodal data (data files + *_images/) lives under the top-level data/ dir.
# Resolve it from this file so paths work regardless of the current directory;
# the image paths stored inside the data files are relative to this directory.
_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'raw', 'bimodal')
)


def load_wdc_shoes_data():
    train_data, val_data, test_data = pickle.load(open(os.path.join(_DATA_DIR, "wdc_shoes_data.pickle"), "rb"))

    return train_data, val_data, test_data


def load_zalando_data():
    train_data, val_data, test_data = pickle.load(open(os.path.join(_DATA_DIR, "zalando_data.pickle"), "rb"))

    return train_data, val_data, test_data


def get_class_weights(data_id):
    if data_id == "WDC Shoes":
        train_data, val_data, test_data = load_wdc_shoes_data()
    elif data_id == "Zalando":
        train_data, val_data, test_data = load_zalando_data()

    neg = train_data["labels"].count(0)
    pos = train_data["labels"].count(1)
    total = neg + pos

    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    return {0: weight_for_0, 1: weight_for_1}


def preprocess_image(filename):
    image = np.array(Image.open(os.path.join(_DATA_DIR, filename)))
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize_with_pad(image, 224, 224)
    return image


def encode_texts(tokenizer, text_pairs):
    encoded = tokenizer.batch_encode_plus(
        text_pairs.tolist(),
        add_special_tokens=True,
        max_length=128,
        return_attention_mask=True,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_tensors="tf"
    )

    input_ids = np.array(encoded["input_ids"], dtype="int32")
    attention_masks = np.array(encoded["attention_mask"], dtype="int32")
    token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

    return input_ids, attention_masks, token_type_ids


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(
            self,
            model_id,
            text_embedding_id,
            text_pairs,
            image_pairs,
            labels,
            shuffle=True,
            include_targets=True,
    ):
        """
        Generate batches for training, validation or testing
        :param model_id: model to generate inputs for
        :param text_embedding_id: text embedding network to tokenize texts for
        :param text_pairs: text pairs to encode
        :param image_pairs: image pairs to load and preprocess
        :param labels: labels to use for validation and testing
        :param shuffle: specifies whether to shuffle pairs and labels
        :param include_targets: specifies whether to include labels
        """
        self.model_id = model_id
        self.text_pairs = text_pairs
        self.image_pairs = image_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = 32
        self.include_targets = include_targets
        if text_embedding_id == "RoBERTa":
            self.tokenizer = transformers.RobertaTokenizer.from_pretrained("roberta-base")
        elif text_embedding_id == "BERT":
            self.tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-cased")
        self.indexes = np.arange(len(self.text_pairs))
        self.on_epoch_end()

    def __len__(self):
        return len(self.text_pairs) // self.batch_size

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]
        text_pairs = self.text_pairs[indexes]
        image_pairs = self.image_pairs[indexes]

        inputs = []
        if self.model_id == "Fine-tuned" or self.model_id == "Intermediate Fusion":
            input_ids, attention_masks, token_type_ids = encode_texts(self.tokenizer, text_pairs)

            inputs = [input_ids, attention_masks, token_type_ids]
        if self.model_id == "Siamese" or self.model_id == "Intermediate Fusion":
            images_1 = [preprocess_image(image_pair[0]) for image_pair in image_pairs]
            images_2 = [preprocess_image(image_pair[1]) for image_pair in image_pairs]

            inputs += [np.array(images_1, dtype="float32"), np.array(images_2, dtype="float32")]

        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="float32")
            return inputs, labels
        else:
            return inputs

    def on_epoch_end(self):
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)


def get_batches(data_id, model_id, text_embedding_id=None):
    """
    Generate batches for training, validation and testing
    :param data_id: dataset to use
    :param model_id: model to generate batches for
    :param text_embedding_id: text embedding network to use for text encoding
    :return: batches for training, validation and testing
    """
    if data_id == "WDC Shoes":
        train_data, val_data, test_data = load_wdc_shoes_data()
    elif data_id == "Zalando":
        train_data, val_data, test_data = load_zalando_data()

    train_batches = BatchGenerator(
        model_id,
        text_embedding_id,
        np.array(train_data["text_pairs"]),
        np.array(train_data["image_pairs"]),
        np.array(train_data["labels"]),
        shuffle=True,
    )
    val_batches = BatchGenerator(
        model_id,
        text_embedding_id,
        np.array(val_data["text_pairs"]),
        np.array(val_data["image_pairs"]),
        np.array(val_data["labels"]),
        shuffle=False,
    )
    test_batches = BatchGenerator(
        model_id,
        text_embedding_id,
        np.array(test_data["text_pairs"]),
        np.array(test_data["image_pairs"]),
        np.array(test_data["labels"]),
        shuffle=False,
    )

    return train_batches, val_batches, test_batches
