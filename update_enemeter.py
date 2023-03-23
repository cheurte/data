import argparse
import os, sys
from glob import glob
import shutil

from utils import read_json

import pandas as pd


def seperate_files_into_folders(config):
    path_enemeter = config["Data"]["enermeter"]
    for file in glob(os.path.join(path_enemeter, "*.CSV")):
        try:
            df = pd.read_csv(file, sep=';',encoding="utf-8")
            if not os.path.exists(os.path.join(path_enemeter, df['Artikelbez 1'][0])):
                os.makedirs(os.path.join(path_enemeter, df['Artikelbez 1'][0]))
            shutil.move(file, os.path.join(path_enemeter, df['Artikelbez 1'][0]))        
        except:
            continue

def CSV2csv(config):
    if "win" in sys.platform:
        sep = "\\"
    elif "linux" in sys.platform:
        sep = "/"
    else:
        raise ValueError ("On which os are you ?")

    path_enermeter = config["Data"]["enermeter"]
    product = config["Data"]["bp110_02"]
    for file in glob(os.path.join(path_enermeter,f"{product}*.CSV")):
        file_name = file.split(sep)[-1].split('.')[0]+".csv"
        path = sep.join(file.split(sep)[:-1])
        shutil.move(file,os.path.join(path, file_name))

if __name__ == "__main__":
    if "win" in sys.platform:
        default_config = "C:\\Users\\corentin.heurte\\Documents\\data\\config\\config_win.json"
    elif "linux" in sys.platform:
        default_config = "/home/cheurte/Documents/biotec/sql/config/config.json"
    else:
        raise ValueError ("On which os are you ?")

    parser = argparse.ArgumentParser("Module for updating the enemeter file")
    parser.add_argument("--config", "-c",
                default=default_config)
    args = parser.parse_args()

    config = read_json(args.config)
    # seperate_files_into_folders(config)
    CSV2csv(config)
