import pandas as pd

import json
import argparse
import os, sys


def read_json(json_file):
    """Allow to read a json file"""
    with open(json_file, 'r', encoding='utf8') as j:
        out = json.load(j)
    return out

def save_backup_dataframe(config, df: pd.DataFrame, name:str):
    if "win" in sys.platform:
        name = name.split("\\")[-1][:-3]
    elif "linux" in sys.platform:
        name = name.split("/")[-1][:-3]
    else:
        raise ValueError("Wrong os")
    if not os.path.exists(config["Data"]["backup"]):
        os.makedirs(config["Data"]["backup"])
    df.to_csv(os.path.join(config["Data"]["backup"], f"{name}.csv"))

def process_everything(config)->pd.DataFrame:
    from enemeter import get_data_enemeter
    from charge import connect_header_data
    from quality import dowload_quality, clean_quality

    return clean_quality(config, dowload_quality(connect_header_data(config, get_data_enemeter(config))))


if __name__=="__main__":
    if "win" in sys.platform:
        default_config = "C:\\Users\\corentin.heurte\\Documents\\data\\config\\config_win.json"
        file_name = __file__.split("\\")[-1][:-3]
    elif "linux" in sys.platform:
        default_config = "/home/cheurte/Documents/biotec/sql/config/config.json"
        file_name = __file__.split("/")[-1][:-3]
    else:
        raise ValueError ("On which os are you ?")

    parser = argparse.ArgumentParser("Utils file and simple arguments")
    parser.add_argument("--config", "-c", default=default_config)
    config = read_json(parser.parse_args().config)
    # save_backup_dataframe(config=config, df=pd.DataFrame(), name=file_name)
    process_everything(config)
