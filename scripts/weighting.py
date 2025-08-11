docstring = """
Adds a weight column to flatten the pT distribution,
 for pT invariance in the model.
This is intended to be applied post-processing to a processed dataframe
 that contains a preserved rawfj_pt column, and will add
 a rawfj_pt_weights column to both the background and signal datasets.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import argparse

# Add parent directory to import local project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import constants as c
from helpers import helpers_main
config = helpers_main.load_config()

import logging
helpers_main.log_config(f"logs/wt_{helpers_main.curr_time()}.log")

class Weighter:
    RAWFJ_PT_COL = c.RAW_FATJET_PROPERTIES_PREFIX + "pt"
    WEIGHTS_SUFFIX = "_weights"

    def __init__(self, cli_args):
        bg_paths = helpers_main.get_files(cli_args.bg, extension=".pkl")
        sg_paths = helpers_main.get_files(cli_args.sg, extension=".pkl")
        if len(bg_paths) != 1 or len(sg_paths) != 1:
            raise Exception(f"Either {cli_args.bg=} or {cli_args.sg=} is not a valid .pkl file path!")

        self.bg_path, self.sg_path = bg_paths[0], sg_paths[0]
        self.bins_no = cli_args.bins
        self.timer = helpers_main.LeTimer()
    
    def modify_dfs(self):
        bg_data, sg_data = self.load()
        self.calc_hist(bg_data)
        self.add_weights(bg_data), self.add_weights(sg_data)
        self.plot(bg_data), self.plot(sg_data)
        self.save_df(bg_data, self.bg_path)
        self.save_df(sg_data, self.sg_path)

    def load(self):
        bg_data = pd.read_pickle(self.bg_path)
        sg_data = pd.read_pickle(self.sg_path)
        if self.RAWFJ_PT_COL not in bg_data or self.RAWFJ_PT_COL not in sg_data:
            # raise Exception(f"{self.RAWFJ_PT_COL} column doesn't exist in either the background or signal data.\n{self.bg_path=}, {self.sg_path=}")
            bg_data[self.RAWFJ_PT_COL] = np.random.rand(len(bg_data)) * 500
            sg_data[self.RAWFJ_PT_COL] = np.random.rand(len(sg_data)) * 500
        logging.info(f"Loaded data; {len(bg_data)=}, {len(sg_data)=} {self.timer.time_taken()}")
        return bg_data, sg_data
    
    def calc_hist(self, bg_data):
        # calculate histogram using background data, to "fit" all data to
        self.freq, self.bins = np.histogram(bg_data[self.RAWFJ_PT_COL], bins=self.bins_no)
        self.freq = np.where(self.freq == 0, 1, self.freq)
        logging.info(f"Calculated histogram, {len(self.freq)=} {self.timer.time_taken()}")

    def add_weights(self, data):
        # get indices according to background bins
        indices = np.digitize(data[self.RAWFJ_PT_COL], bins=self.bins)
        indices = np.clip(indices, 1, self.bins_no) - 1
        weights = 1 / self.freq[indices]
        # normalise using w_i/sum_n(w) * n
        weights *= len(data) / np.sum(weights)
        data[self.RAWFJ_PT_COL + self.WEIGHTS_SUFFIX] = weights
        logging.info(f"Added weights {self.timer.time_taken()}")
    
    def save_df(self, data, data_path):
        path = os.path.join(
            config['data']['processed_data_dir'],
            "weighted",
            f"{helpers_main.trim_name(data_path)}_{helpers_main.curr_time()}.pkl"
        )
        helpers_main.create_missing_dir(path)
        data.to_pickle(path)
        logging.info(f"Saved to {path} {self.timer.time_taken()}")

    def plot(self, data):
        plt.hist(
            data[self.RAWFJ_PT_COL],
            bins=self.bins,
            weights=data[self.RAWFJ_PT_COL + self.WEIGHTS_SUFFIX]
        )
        fig_path = f"plots/distributions/wts_{helpers_main.curr_time()}.png"
        helpers_main.create_missing_dir(fig_path)
        plt.savefig(fig_path)
        logging.info(f"Saved plot into {fig_path}")
        plt.clf()


def main(args):
    weighter = Weighter(args)
    weighter.modify_dfs()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Weighter",
        description=docstring
    )
    parser.add_argument(
        "--bg", "-b", type=str, default=config["data"]["background_file"],
        help="Path to the processed background file to calculate the rawfj_pt histogram with and apply weights to."
    )
    parser.add_argument(
        "--sg", "-s", type=str, default=config["data"]["signal_file"],
        help="Path to the processed signal file to apply weights to."
    )
    parser.add_argument(
        "--bins", "-n", type=int, default=config["data"]["bins"],
        help="Number of histogram bins to calculate the weights with. Defaults to bins in config.yaml"
    )
    # add a range here, lower/upperpt

    args = parser.parse_args()
    main(args)
