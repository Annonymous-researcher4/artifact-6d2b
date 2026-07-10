from unimodal_pair_encoders.bimodal import data_preparation
import tensorflow as tf
import tensorflow_addons as tfa
import transformers
from transformers import TFCLIPVisionModel

#   https://tfhub.dev/sayakpaul/swin_base_patch4_window7_224_in22k_fe/1
path_to_swin_model = "swin-tensorflow2-base-patch4-window7-224-in22k-fe-v1"

CLIP_MODEL_MAPPING = {
    'ViT-B/32': 'openai/clip-vit-base-patch32',
    'ViT-L/14': 'openai/clip-vit-large-patch14'
}

CLIP_EMBEDDING_DIMS = {
    'ViT-B/32': 768,
    'ViT-L/14': 1024
}

metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tfa.metrics.F1Score(num_classes=1, threshold=0.5, name="f1_score")
]

callback = tf.keras.callbacks.EarlyStopping(monitor="val_f1_score", mode="max", patience=10, restore_best_weights=True)

def build_fine_tuned_model(embedding_id):
    """
    Build model with Fine-tuned architecture
    :param embedding_id: embedding network to use
    :return: uncompiled model
    """

    input_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)
    attention_masks = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)
    token_type_ids = tf.keras.layers.Input(shape=(128,), dtype=tf.int32)

    if embedding_id == "RoBERTa":
        embedding_network = transformers.TFRobertaModel.from_pretrained("roberta-base")
        embedding_network.trainable = False

        embedding_network_output = embedding_network.roberta(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

    elif embedding_id == "BERT":
        embedding_network = transformers.TFBertModel.from_pretrained("bert-base-cased")
        embedding_network.trainable = False

        embedding_network_output = embedding_network.bert(
            input_ids,
            attention_mask=attention_masks,
            token_type_ids=token_type_ids
        )

    sequence_output = embedding_network_output.last_hidden_state
    bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(sequence_output)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    batch_norm = tf.keras.layers.BatchNormalization()(concat)
    dropout_1 = tf.keras.layers.Dropout(0.3)(batch_norm)
    dense_1 = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(0.3)(dense_1)
    dense_2 = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal", name="text_hidden_layer")(dropout_2)
    dropout_3 = tf.keras.layers.Dropout(0.3)(dense_2)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout_3)

    model = tf.keras.Model(inputs=[input_ids, attention_masks, token_type_ids], outputs=output)

    return model


def build_contrastive_siamese_model(embedding_id):
    """
    Build Contrastive Siamese model for STAGE 1: Contrastive learning with cosine embedding loss
    Outputs normalized embeddings for computing cosine similarity
    
    :param embedding_id: embedding network to use ("Swin-Transformer" or "ResNet")
    :return: uncompiled contrastive model
    """
    if embedding_id == "Swin-Transformer":
        embedding_network = tf.keras.models.load_model(path_to_swin_model)

        for layer in embedding_network.layers:
            if layer.name != "basic_layer_3":
                layer.trainable = False

        for layer in embedding_network.get_layer("basic_layer_3").layers:
            if layer.name != "swin_transformer_block_1":
                layer.trainable = False

        for layer in embedding_network.get_layer("basic_layer_3").get_layer("swin_transformer_block_1").layers:
            if layer.name != "mlp":
                layer.trainable = False

        for layer in embedding_network.get_layer("basic_layer_3").get_layer("swin_transformer_block_1").get_layer(
                "mlp").layers:
            if layer.name not in ["dense_49", "dense_50"]:
                layer.trainable = False
            else:
                layer.trainable = True

    elif embedding_id == "ResNet":
        embedding_network = tf.keras.applications.resnet.ResNet50(
            weights="imagenet",
            input_shape=(224, 224) + (3,),
            include_top=False
        )

        trainable = False
        for layer in embedding_network.layers:
            if layer.name == "conv5_block1_out":
                trainable = True
            layer.trainable = trainable

        pool = tf.keras.layers.GlobalAveragePooling2D()(embedding_network.output)
        embedding_network = tf.keras.Model(embedding_network.input, pool)

    image = tf.keras.layers.Input((224, 224) + (3,))
    embedding = embedding_network(image, training=False)
    dropout = tf.keras.layers.Dropout(0.3)(embedding)
    dense_1 = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")(dropout)
    batch_norm_1 = tf.keras.layers.BatchNormalization()(dense_1)
    dense_2 = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(batch_norm_1)
    batch_norm_2 = tf.keras.layers.BatchNormalization()(dense_2)
    sister_output = tf.keras.layers.Dense(256)(batch_norm_2)
    sister_model = tf.keras.Model(image, sister_output, name="sister_network")
    
    image_1 = tf.keras.layers.Input((224, 224) + (3,))
    embedding_1 = sister_model(image_1)

    image_2 = tf.keras.layers.Input((224, 224) + (3,))
    embedding_2 = sister_model(image_2)

    normalized_emb_1 = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1)
    )(embedding_1)
    normalized_emb_2 = tf.keras.layers.Lambda(
        lambda x: tf.math.l2_normalize(x, axis=1)
    )(embedding_2)

    embeddings_concat = tf.keras.layers.Concatenate()([normalized_emb_1, normalized_emb_2])
    
    contrastive_model = tf.keras.Model(
        inputs=[image_1, image_2], 
        outputs=embeddings_concat,
    )
    
    return contrastive_model


def build_clip_contrastive_stage1(clip_model_name):
    """
    Build CLIP contrastive model for STAGE 1
    Fine-tunes CLIP with contrastive loss to learn better embeddings
    
    :param clip_model_name: CLIP model to use ('ViT-B/32' or 'ViT-L/14')
    :return: uncompiled contrastive model (outputs normalized embeddings)
    """
    model_id = CLIP_MODEL_MAPPING[clip_model_name]
    embedding_dim = CLIP_EMBEDDING_DIMS[clip_model_name]
    
    clip_vision_model = TFCLIPVisionModel.from_pretrained(model_id)
    clip_vision_model.trainable = True
    
    def preprocess_for_clip(images):
        """Preprocess images for CLIP: normalize and transpose to channels-first"""
        mean = tf.constant([0.48145466, 0.45782750, 0.40821073], shape=[1, 1, 1, 3])
        std = tf.constant([0.26862954, 0.26130258, 0.27577711], shape=[1, 1, 1, 3])
        normalized = (images - mean) / std
        transposed = tf.transpose(normalized, perm=[0, 3, 1, 2])
        return transposed
    
    image = tf.keras.layers.Input((224, 224, 3), name="image_input")
    preprocessed = tf.keras.layers.Lambda(preprocess_for_clip)(image)
    clip_outputs = clip_vision_model(pixel_values=preprocessed, training=True)
    clip_embedding = clip_outputs.pooler_output
    encoder = tf.keras.Model(image, clip_embedding, name="clip_encoder")
    
    image_1 = tf.keras.layers.Input((224, 224, 3), name="image_1")
    image_2 = tf.keras.layers.Input((224, 224, 3), name="image_2")
    
    emb_1 = encoder(image_1)
    emb_2 = encoder(image_2)
    
    norm_emb_1 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(emb_1)
    norm_emb_2 = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(emb_2)
    
    embeddings_concat = tf.keras.layers.Concatenate()([norm_emb_1, norm_emb_2])
    
    model = tf.keras.Model(
        inputs=[image_1, image_2], 
        outputs=embeddings_concat,
        name="clip_contrastive_stage1"
    )
    
    return model


def build_clip_classifier_stage2(trained_contrastive_model):
    """
    Build classifier for STAGE 2
    Uses frozen CLIP encoder from Stage 1 + tiny classifier on absolute difference
    
    :param trained_contrastive_model: trained contrastive model from Stage 1
    :return: uncompiled classifier model
    """
    clip_encoder = trained_contrastive_model.get_layer("clip_encoder")
    clip_encoder.trainable = False
    
    image_1 = tf.keras.layers.Input((224, 224, 3), name="image_1")
    image_2 = tf.keras.layers.Input((224, 224, 3), name="image_2")
    
    emb_1 = clip_encoder(image_1)
    emb_2 = clip_encoder(image_2)
    
    abs_diff = tf.keras.layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([emb_1, emb_2])
    
    batch_norm = tf.keras.layers.BatchNormalization()(abs_diff)
    hidden = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal",
                                   name="image_hidden_layer")(batch_norm)
    dropout = tf.keras.layers.Dropout(0.3)(hidden)
    
    output = tf.keras.layers.Dense(1, activation=None, kernel_initializer="glorot_uniform")(dropout)
    
    model = tf.keras.Model(
        inputs=[image_1, image_2],
        outputs=output,
        name="clip_classifier_stage2"
    )
    
    return model


def build_classifier_model(trained_contrastive_model):
    """
    Build classifier model for STAGE 2: Uses trained embeddings from contrastive model
    Extracts the sister network, freezes it, and adds a 2-layer classifier on top
    
    :param trained_contrastive_model: trained contrastive siamese model from stage 1
    :return: uncompiled classifier model with frozen embeddings
    """
    sister_model = trained_contrastive_model.get_layer("sister_network")
    sister_model.trainable = False
    
    image_1 = tf.keras.layers.Input((224, 224) + (3,))
    embedding_1 = sister_model(image_1)

    image_2 = tf.keras.layers.Input((224, 224) + (3,))
    embedding_2 = sister_model(image_2)

    absolute_diff = tf.keras.layers.Lambda(
        lambda x: tf.abs(x[0] - x[1])
    )([embedding_1, embedding_2])
    
    batch_norm = tf.keras.layers.BatchNormalization()(absolute_diff)
    
    classifier_hidden = tf.keras.layers.Dense(
        256, 
        activation="relu", 
        kernel_initializer="he_normal",
        name="image_hidden_layer"
    )(batch_norm)
    classifier_dropout = tf.keras.layers.Dropout(0.3)(classifier_hidden)
    
    output = tf.keras.layers.Dense(
        1, 
        activation=None,
        kernel_initializer="glorot_uniform"
    )(classifier_dropout)
    
    classifier_model = tf.keras.Model(
        inputs=[image_1, image_2], 
        outputs=output
    )
    
    return classifier_model 


def build_intermediate_fusion_model(fine_tuned_model, siamese_model):
    """
    Build model with Intermediate Fusion architecture
    Uses the hidden layer representations (256-dim) from both pre-trained models
    :param fine_tuned_model: pre-trained Text Branch
    :param siamese_model: pre-trained Image Branch (Private Siamese classifier)
    :return: uncompiled model
    """
    fine_tuned_model.trainable = False
    siamese_model.trainable = False

    text_hidden = fine_tuned_model.get_layer("text_text_hidden_layer").output
    image_hidden = siamese_model.get_layer("image_image_hidden_layer").output

    concat = tf.keras.layers.concatenate([text_hidden, image_hidden])
    batch_norm = tf.keras.layers.BatchNormalization()(concat)
    dropout_1 = tf.keras.layers.Dropout(0.3)(batch_norm)
    dense_1 = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")(dropout_1)
    dropout_2 = tf.keras.layers.Dropout(0.3)(dense_1)
    dense_2 = tf.keras.layers.Dense(512, activation="relu", kernel_initializer="he_normal")(dropout_2)
    dropout_3 = tf.keras.layers.Dropout(0.3)(dense_2)
    dense_3 = tf.keras.layers.Dense(256, activation="relu", kernel_initializer="he_normal")(dropout_3)
    dropout_4 = tf.keras.layers.Dropout(0.3)(dense_3)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(dropout_4)

    model = tf.keras.models.Model(inputs=fine_tuned_model.input + siamese_model.input, outputs=output)

    return model


def fit(model, data_id, train_batches, val_batches, epochs):
    model.fit(
        train_batches,
        validation_data=val_batches,
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
        class_weight=data_preparation.get_class_weights(data_id),
        callbacks=[callback]
    )

    return model


def cosine_embedding_loss(margin=0.5):
    """
    Cosine embedding loss for contrastive learning
    
    Formula:
        loss(x, y) = {
            1 - cos(x1, x2),                    if y = 1 (similar)
            max(0, cos(x1, x2) - margin),       if y = 0 (dissimilar)
        }
    
    For similar pairs (y=1): minimize cosine distance (maximize similarity)
    For dissimilar pairs (y=0): push cosine similarity below margin (make dissimilar)
    
    :param margin: margin for dissimilar pairs (typically 0.0 to 1.0)
    :return: loss function
    """
    def loss_fn(y_true, embeddings_concat):
        embedding_dim = tf.shape(embeddings_concat)[1] // 2
        emb_1 = embeddings_concat[:, :embedding_dim]
        emb_2 = embeddings_concat[:, embedding_dim:]
        
        cosine_sim = tf.reduce_sum(emb_1 * emb_2, axis=1, keepdims=True)
        loss_similar = y_true * (1.0 - cosine_sim)
        loss_dissimilar = (1 - y_true) * tf.maximum(0.0, cosine_sim - margin)
        
        return tf.reduce_mean(loss_similar + loss_dissimilar)
    
    return loss_fn


def train_clip_contrastive_stage1(contrastive_model, data_id, train_batches, val_batches, epochs):
    """
    STAGE 1: Train CLIP with contrastive loss
    Fine-tunes CLIP encoder to learn better embeddings
    
    :param contrastive_model: CLIP contrastive model
    :param data_id: dataset to use for class weight calculation
    :param train_batches: batches for training
    :param val_batches: batches for validation
    :param epochs: number of epochs
    :return: trained contrastive model
    """
    contrastive_model.compile(
        loss=cosine_embedding_loss(margin=0.5),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=[]
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    contrastive_model.fit(
        train_batches,
        validation_data=val_batches,
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
        class_weight=data_preparation.get_class_weights(data_id),
        callbacks=[early_stopping]
    )
    
    return contrastive_model


def train_clip_classifier_stage2(classifier_model, data_id, train_batches, val_batches, epochs):
    """
    STAGE 2: Train classifier with frozen CLIP
    Trains lightweight classifier on absolute difference of embeddings
    
    :param classifier_model: classifier model with frozen CLIP encoder
    :param data_id: dataset to use for class weight calculation
    :param train_batches: batches for training
    :param val_batches: batches for validation
    :param epochs: number of epochs
    :return: trained classifier model
    """
    classifier_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        metrics=metrics
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_f1_score",
        mode="max",
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    classifier_model.fit(
        train_batches,
        validation_data=val_batches,
        epochs=epochs,
        use_multiprocessing=True,
        workers=-1,
        class_weight=data_preparation.get_class_weights(data_id),
        callbacks=[early_stopping]
    )
    
    return classifier_model


def build_and_train_private_siamese_model(embedding_id, data_id, train_batches, val_batches, 
                                          contrastive_epochs, classifier_epochs, use_clip=False):
    """
    Build and train Private Siamese model
    
    :param embedding_id: embedding network to use ("Swin-Transformer", "ResNet", or CLIP model ["ViT-B/32", "ViT-L/14"])
    :param data_id: dataset name for class weight calculation
    :param train_batches: training data batches
    :param val_batches: validation data batches
    :param contrastive_epochs: epochs for contrastive embedding training
    :param classifier_epochs: epochs for classifier training
    :param use_clip: if True, use CLIP approach (fine-tune with contrastive loss + train classifier)
    :return: trained classifier model
    """
    if use_clip:
        contrastive_model = build_clip_contrastive_stage1(embedding_id)
        trained_contrastive = train_clip_contrastive_stage1(
            contrastive_model,
            data_id,
            train_batches,
            val_batches,
            contrastive_epochs
        )
        
        classifier_model = build_clip_classifier_stage2(trained_contrastive)
        trained_classifier = train_clip_classifier_stage2(
            classifier_model,
            data_id,
            train_batches,
            val_batches,
            classifier_epochs
        )
        
        return trained_classifier
    
    contrastive_model = build_contrastive_siamese_model(embedding_id)
    
    trained_contrastive_model = train_contrastive_siamese_model(
        contrastive_model, 
        data_id, 
        train_batches, 
        val_batches,
        contrastive_epochs
    )

    classifier_model = build_classifier_model(trained_contrastive_model)
    trained_classifier_model = train_classifier_model(
        classifier_model,
        data_id,
        train_batches,
        val_batches,
        classifier_epochs
    )

    return trained_classifier_model


def train_contrastive_siamese_model(contrastive_model, data_id, train_batches, val_batches, contrastive_epochs):
    """
    Train contrastive siamese model with cosine embedding loss
    
    :param contrastive_model: contrastive model (outputs concatenated normalized embeddings)
    :param data_id: dataset to use for class weight calculation
    :param train_batches: batches for training
    :param val_batches: batches for validation
    :param contrastive_epochs: number of epochs for contrastive embedding training
    :return: trained contrastive model (best weights restored)
    """
    contrastive_model.compile(
        loss=cosine_embedding_loss(margin=0.5),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=[]
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    contrastive_model.fit(
        train_batches,
        validation_data=val_batches,
        epochs=contrastive_epochs,
        use_multiprocessing=True,
        workers=-1,
        class_weight=data_preparation.get_class_weights(data_id),
        callbacks=[early_stopping]
    )
    
    return contrastive_model

def train_classifier_model(classifier_model, data_id, train_batches, val_batches, classifier_epochs):
    """
    Train classifier model with frozen embeddings using BCEWithLogitsLoss
    :param classifier_model: model to train
    :param data_id: dataset to use for class weight calculation
    :param train_batches: batches for training
    :param val_batches: batches for validation
    :param classifier_epochs: number of epochs for classifier training
    :return: trained classifier model
    """
    classifier_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        metrics=metrics
    )
    
    classifier_model = fit(classifier_model, data_id, train_batches, val_batches, classifier_epochs)

    return classifier_model
