import argparse
import logging
from typing import List, Optional
from inference import run_inference

logger = logging.getLogger("training_cli")

def main(parsed_args: argparse.Namespace):
    run_inference(**vars(parsed_args))

def parse_args(args: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument("queries", type=str, help="Comma-separated list of queries.")
    parser.add_argument("--product_data_path", type=str, help="Path to product catalogue csv file.")
    parser.add_argument("--model_path", type=str, help="Path to saved model.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main(parse_args())
