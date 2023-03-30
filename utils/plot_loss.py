import matplotlib.pyplot as plt
import pandas as pd
import argparse
import os, sys

sys.path.append("/home/cheurte/Documents/data/")
from utils import read_json

def plot_loss(config, df_loss:pd.DataFrame)->None:
    assert not df_loss.empty
    df_loss.plot()
    plt.savefig(os.path.join(config['visu']['output'], f"losses.png"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot the train and validation loss')
    parser.add_argument("--config", '-c', default="/home/cheurte/Documents/biotec/process/config/fc/config_fc_8.json", help ="Config file")
    parser.add_argument("--save_file", '-f', help="Optional")
    args = parser.parse_args()
    if args.save_file is not None:
        df = pd.read_csv(args.save_file)
    else:
        raise ValueError("Nothing to print")
    config = read_json(args.config)
    plot_loss(config, df)
