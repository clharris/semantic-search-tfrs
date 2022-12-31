import tensorflow as tf
import tensorflow_recommenders as tfrs
import keras.backend as K
import pandas as pd
import logging
from typing import Dict, Text

logger = logging.getLogger(__name__)

class QueryProductsModel(tfrs.Model):
    # We derive from a custom base class to help reduce boilerplate. Under the hood,
    # these are still plain Keras Models.

    def __init__(
            self,
            query_model: tf.keras.Model,
            products_model: tf.keras.Model,
            task: tfrs.tasks.Retrieval):
        super().__init__()

        # Set up query and product representations.
        self.query_model = query_model
        self.products_model = products_model

        # Set up a retrieval task.
        self.task = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # Define how the loss is computed.

        query_embeddings = self.query_model(features["query"])
        product_embeddings = self.products_model(features["product_title"])

        return self.task(query_embeddings, product_embeddings)

# interaction_path, candidate_path, user_id_feature, user_model_class, 
def train_model(train_data_path = "data/train-v0.2_us.csv",
                product_data_path = "data/product_catalogue-v0.2_us.csv",
                max_tokens = 5000,
                embedding_dim = 64,
                epochs = 15,
                model_output_path = "models/index"):

    train_data = pd.read_csv(train_data_path)
    product_data = pd.read_csv(product_data_path)
    product_data = product_data[product_data["product_title"].notnull()]
    product_data.fillna("", inplace=True)

    features = ["query", "product_id", "product_title", "product_description", "product_brand"] + ["esci_label"]
    train_data = train_data.merge(product_data, left_on=['query_locale', 'product_id'],
                                  right_on=['product_locale', 'product_id'])[features]

    ## Would be better to give relative weightings between the labels exact, substitute, and complement
    positive_labels = train_data[train_data["esci_label"] != "irrelevant"]
    train_ds = tf.data.Dataset.from_tensor_slices(dict(positive_labels))

    # Select the basic features.
    train_ds = train_ds.map(lambda x: {
        "product_id": x["product_id"],
        "product_title": x["product_title"],
        # "product_description": x["product_description"],
        # "product_brand": x["product_brand"],
        "query": x["query"]
    })
    # TODO: add additional product fields
    products = train_ds.map(lambda x: x["product_title"])
    queries = train_ds.map(lambda x: x["query"])
    product_ds = tf.data.Dataset.from_tensor_slices(dict(product_data[["product_title"]]))

    text_vectorization_layer = tf.keras.layers.TextVectorization(max_tokens=max_tokens,
                                                                 ngrams=5,
                                                                 standardize='lower_and_strip_punctuation',
                                                                 output_mode='int',
                                                                 output_sequence_length=5
                                                                 )
    text_vectorization_layer.adapt(queries.batch(64))
    text_vectorization_layer.adapt(product_ds.map(lambda x: x["product_title"]).batch(64))

    # Define query and product models.
    query_model = tf.keras.Sequential([
        text_vectorization_layer,
        tf.keras.layers.Embedding(max_tokens, embedding_dim),
        tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim,))
    ])
    products_model = tf.keras.Sequential([
        text_vectorization_layer,
        tf.keras.layers.Embedding(max_tokens, embedding_dim),
        tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1), output_shape=(embedding_dim,))
    ])

    # Define your objectives.
    task = tfrs.tasks.Retrieval(metrics=tfrs.metrics.FactorizedTopK(
        products.batch(128).map(products_model),
        ks=(1, 5, 10)
    ))

    # Create a retrieval model.
    model = QueryProductsModel(query_model, products_model, task)
    model.compile(optimizer=tf.keras.optimizers.Adagrad(0.5))
    model.fit(train_ds.batch(4096), epochs=epochs)

    # Use brute-force search to set up retrieval using the trained representations.
    index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
    index.index_from_dataset(
        products.batch(100).map(lambda title: (title, model.products_model(title))))

    tf.saved_model.save(
        index,
        model_output_path)

