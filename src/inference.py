import tensorflow as tf
import numpy as np
import pandas as pd


def query_index(query_string, index, product_data):
    scores, product_ids = index(np.array([query_string]))
    top_10 = [s.decode("utf-8") for s in product_ids[0, :10].numpy()]
    scores_10 = [s for s in scores.numpy()[:10][0]]
    score_dict = dict(zip(top_10, scores_10))
    df = product_data[product_data["product_title"].isin(top_10)]
    df["score"] = df["product_title"].map(lambda title: score_dict[title])
    print(f"Top results for {query_string}")
    df.sort_values(["score"], inplace=True, ascending=False)
    print(df)


def run_inference(queries,
                  output_path="models/index",
                  product_data_path="data/product_catalogue-v0.2_us.csv"):
    index = tf.saved_model.load(output_path)
    product_data = pd.read_csv(product_data_path)

    queries = queries.split(",")

    for query in queries:
        query_index(query, index, product_data)
