"""Module allowing to get the enemeter files"""
import argparse
from datetime import datetime
from glob import glob
import os
import sys

import numpy as np
import pandas as pd

from utils import read_json, save_backup_dataframe

def drop_strange_kst(df: pd.DataFrame)-> pd.DataFrame:
    """Remove value of Kst Id col wich are not KST smthing"""
    for i,val in enumerate(df['Kst Id']):
        if 'ZSK' not in val:
            df.loc[i, "Kst Id"] = np.nan
    df.dropna(axis=0, inplace=True)
    return df

def has_duplicate_kst(df: pd.DataFrame)-> bool:
    """Check if the DataFrame has any wrong values"""
    return len(df['Kst Id'].unique()) != len(df['Kst Id'])

def clean_by_weight(df: pd.DataFrame, kst: str)-> pd.DataFrame:
    """Clean by weights"""
    df_curr = df[df["Kst Id"] == kst]
    df_curr = df_curr[df_curr["Istmenge"] != max(df_curr["Istmenge"])]
    df = pd.concat([df, df_curr])
    df.drop_duplicates(keep=False, inplace=True)
    return df

def clean_by_time(df: pd.DataFrame, kst: str)->pd.DataFrame:
    """Function to clean with respect of the time"""
    df_curr = pd.DataFrame(df[df["Kst Id"] == kst])
    time =[]
    for _, value in df_curr.iterrows():
        start = datetime.strptime(value["Beginn"], "%d.%m.%Y %H:%M:%S") # type: ignore
        end = datetime.strptime(value['Ende'], "%d.%m.%Y %H:%M:%S") # type: ignore
        time.append((end - start).total_seconds())
    df_curr['time'] = time
    df_curr = df_curr[df_curr['time'] != max(df_curr['time'])]
    df_curr.drop(columns='time', inplace=True)
    df = pd.concat([df, df_curr])
    df.drop_duplicates(keep=False, inplace=True)
    return df

def clean_batches(df: pd.DataFrame)-> pd.DataFrame:
    """Clean a batch with: the weights asign, and then if necessary with the time"""
    df = pd.DataFrame(df[df['Istmenge'] > 0])
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True, drop=True)
    for kst in df['Kst Id'].unique():

        if len(df[df['Kst Id'] == kst]) == 1:
            continue
        df = clean_by_weight(df, kst)

        if len(df[df['Kst Id'] == kst]) == 1:
            continue
        df = clean_by_time(df, kst)
    return df

def get_data_enemeter(config)-> pd.DataFrame:
    """Function to get the enemeter data"""
    dfs = []
    path_files = os.path.join(config['Data']["enermeter"], config["Data"]["g21"])
    for file in glob(os.path.join(path_files, "*.csv")):
        df = pd.read_csv(file, sep=';', encoding_errors="ignore",
            usecols=["Typ","Nr","Artikelnr","Artikelbez 1","Istmenge","Kst Id","Beginn","Ende"])
        df.dropna(how='all', axis=1, inplace=True)
        df.dropna(how='any', axis=0, inplace=True)
        df = df.drop(columns=['Typ', 'Artikelbez 1'])
        df = drop_strange_kst(df)
        if has_duplicate_kst(df):
            df = clean_batches(df)
        dfs.append(df)
    output = pd.concat(dfs)
    output.reset_index(inplace=True, drop=True)
    if config["Data"]["save_backup"]:
        save_backup_dataframe(config=config, df=output, name=__file__)
        print("enemeter saved")
    return output

if __name__=="__main__":
    if "win" in sys.platform:
        default_config = "C:\\Users\\corentin.heurte\\Documents\\data\\config\\config_win.json"
    elif "linux" in sys.platform:
        default_config = "/home/cheurte/Documents/data/config/config_2.json"
    else:
        raise ValueError ("On which os are you ?")

    parser = argparse.ArgumentParser("Module for the enemeter file")
    parser.add_argument("--config", "-c",
                default=default_config)
    args = parser.parse_args()

    config = read_json(args.config)
    df = get_data_enemeter(config)
    # print(df)
    a = list()
    for val in df.Nr:
        a.append(f"nr = {str(val)} or ")
    a = "".join(a)
    print(a)
