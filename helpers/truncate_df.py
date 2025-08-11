import pandas as pd
import os
import sys
import argparse
import logging

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from helpers import helpers_main
import constants as c
helpers_main.log_config(f"logs/trunc_{helpers_main.curr_time()}.log")

def trunc(path, final_len):
    df_paths = helpers_main.get_files(path, extension=".pkl")
    logging.info(f"Found {len(df_paths)=} files.")

    for df_path in df_paths:
        df = pd.read_pickle(df_path)
        logging.info(f"Loaded {df_path=} with {len(df)=}.")
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.head(final_len)
        logging.info(f"Shortened to {len(df)=}.")

        savepath = os.path.join(
            os.path.dirname(df_path),
            f"trunc_{final_len}_{helpers_main.trim_name(df_path)}.pkl"
        )
        df.to_pickle(savepath)
        logging.info(f"Saved to {savepath}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Truncator4000",
        description="Truncates the given dataframe after shuffling, for easier data testing"
    )
    parser.add_argument(
        "--path", "-p", type=str, required=True,
        help='Path of file to truncate, or folder of files'
    )
    parser.add_argument(
        "--len", "-n", type=int, required=True,
        help="Length to truncate to."
    )
    args = parser.parse_args()
    
    trunc(args.path, args.len)