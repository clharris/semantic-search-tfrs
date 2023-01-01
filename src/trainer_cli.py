import argparse
import logging
from typing import List, Optional
from train import train_model

logger = logging.getLogger("aicrowd.training_cli")


def main(parsed_args: argparse.Namespace):
    train_model(**vars(parsed_args))


def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("--train_data_path", type=str,
                        help="Path to training csv file containing queries and products.")
    parser.add_argument("--product_data_path", type=str, help="Path to product catalogue csv file.")
    parser.add_argument("--model_output_path", type=str, help="Path to saved model.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs used for training. ")
    parser.add_argument("--max_tokens", type=int, default=5000, help="Max number of tokens during text vectorization.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Dimension of product and query embeddings.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
