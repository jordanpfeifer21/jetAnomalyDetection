"""
Adds a weight column to flatten the pT distribution,
for pT invariance in the model.
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

    def __init__(self, cli_args):
        self.files = helpers_main.get_files(cli_args.path, extension=".pkl")
        if len(self.files) == 0:
            raise Exception(f"{cli_args.path=} is not a valid .pkl file or folder path!")

        self.bins = cli_args.bins
        self.timer = helpers_main.LeTimer()
    
    def add_weights(self):
        for f in self.files:
            data = pd.read_pickle(f)
            if self.RAWFJ_PT_COL not in data:
                logging.warning(f"Skipping {f=} as it has no {self.RAWFJ_PT_COL} column.")
                continue
            
            hist = np.histogram(data[self.RAWFJ_PT_COL], bins=self.bins)

            # now, get the counts, clean if needed, calc inverse count for weights, and NORMALISE
            # plot!!!!
    
    def plot(self):
        pass



def main(args):
    weighter = Weighter(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Weighter",
        description="adds a weight column to the processed data, to flatten the pT distribution during training"
    )
    parser.add_argument(
        "--path", "-p", type=str, default=config["data"]["background_file"],
        help="Path to the processed background file to add rawfj_pt weights to, or a folder of such files. Defaults to background_file in config.yaml"
    )
    parser.add_argument(
        "--bins", "-b", type=int, default=config["data"]["bins"],
        help="Number of histogram bins to calculate the weights with. Defaults ot bins in config.yaml"
    )
    # add a range here, lower/upperpt

    args = parser.parse_args()
    main(args)
