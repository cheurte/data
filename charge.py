""" Get the data from bab and add them to the enermeter data """
import argparse
import sys

import pandas as pd
import numpy as np
from enemeter import get_data_enemeter
from db import SQLcust
from utils import read_json, save_backup_dataframe
import warnings

warnings.filterwarnings("ignore")

def connect_header_data(config, enemeter_df:pd.DataFrame)-> pd.DataFrame:
    """Create connection between charge and csnr number"""
    conn = SQLcust()
    bab_df = pd.read_sql("SELECT CAST(nr as varchar(10)) as nr, CAST(charge as varchar(10)) as charge FROM bab where charge is not null", conn.connectorQS)
    bab_df.charge = pd.to_numeric(bab_df.charge, errors="coerce", downcast="signed")
    bab_df.dropna(inplace=True)
    bab_df = bab_df.astype("int")
    charge = []
    corr= pd.DataFrame()

    """Remove all strange charge values"""
    for _, val in enemeter_df.iterrows():
        charge = bab_df[bab_df['nr'] == val['Nr']] ["charge"]
        if len(str(charge.values)) != 10:
            charge = pd.Series(np.nan)
        corr = pd.concat([corr, charge])
    corr.reset_index(inplace=True, drop=True)
    corr.rename(columns={corr.columns[0]:"charge"}, inplace=True)

    """Rename the dataframe into header_df"""
    header_df = pd.concat([enemeter_df, corr], axis=1)
    header_df.dropna(inplace=True, axis=0, how="any")
    header_df.charge = pd.to_numeric(header_df.charge, downcast="signed")
    
    """Saving backup"""
    if config["Data"]["save_backup"]:
        save_backup_dataframe(config=config, df=enemeter_df, name=__file__)
        print("charge saved")
    return header_df

if __name__=="__main__":
    if "win" in sys.platform:
        default_config = "C:\\Users\\corentin.heurte\\Documents\\data\\config\\config_win.json"
    elif "linux" in sys.platform:
        default_config = "/home/cheurte/Documents/biotec/sql/config/config.json"
    else:
        raise ValueError ("On which os are you ?")

    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", "-c",
                        default=default_config)
    args = parser.parse_args()
    config = read_json(args.config)

    enemeter_df = get_data_enemeter(config=config)
    df = connect_header_data(config=config, enemeter_df=enemeter_df)
