import pandas as pd
import numpy as np

import os, sys
import argparse
import re
# from time import time
from datetime import datetime, timedelta
# from tqdm import tqdm

from utils import read_json, save_backup_dataframe
from quality import run_cleaning_colors 
from db import SQLcust

def put_time_correct_format(row: pd.Series)-> str:
    """ Put the time with : instead of . """
    if len(str(row["Uhrzeit"])) < 5 :
        return "0" + row["Uhrzeit"]
    else:
        return str(row["Uhrzeit"])

def get_exact_date(df_quality:pd.DataFrame)->pd.DataFrame:
    """ find the exact date """
    curr_time = list()
    for _, row in df_quality.iterrows():
        uhrzeit = put_time_correct_format(row)

        """ Get the proper date """
        res = re.match("[0-9][0-9].[0-9][0-9]", uhrzeit)
        if res is None:
            curr_time.append(np.nan)
            continue
        uhrzeit = datetime.strptime(res.group().replace(".", ":"),"%H:%M")
        curr_day_start = datetime.strptime(row.Beginn, "%d.%m.%Y %H:%M:%S")
        curr_day_end = datetime.strptime(row.Ende, "%d.%m.%Y %H:%M:%S")
        if (curr_day_end.day - curr_day_start.day) > 1:
            curr_time.append(np.nan)
            continue

        """ find which day it correspond """
        uhrzeit = uhrzeit.replace(day=curr_day_start.day, month= curr_day_start.month, year=curr_day_start.year)
        if (uhrzeit - curr_day_start).total_seconds() < 0:
            if (uhrzeit - curr_day_end).total_seconds() > 0:
                curr_time.append(np.nan)
                continue
            uhrzeit = uhrzeit.replace(day=curr_day_end.day, month= curr_day_end.month, year=curr_day_end.year)
        curr_time.append(uhrzeit)

    """ add to the dataframe and clean it """
    df_quality['current_time'] = curr_time
    df_quality.dropna(axis=0, inplace=True, how="any")
    df_quality.reset_index(drop=True, inplace=True)
    if config["Data"]["save_backup"]:
        save_backup_dataframe(config=config, df=df_quality, name=__file__)
        print("production saved")
    return df_quality

def dowload_production(config, df_quality:pd.DataFrame)->pd.DataFrame:
    """ Download the data from production """
    conn = SQLcust()
    column = "SELECT "
    production = pd.DataFrame()

    """ Getting the column names """
    for entry in config["Data"]["columns_production"]:
        column = column + f"CAST({entry} AS VARCHAR(10)) AS {entry}, "
    column = column[:-2]
    """ Downloading data """
    for _, (_, value) in enumerate(df_quality.iterrows()):
        line = str(value["Line"]).replace("ZSK ","")
        query = column + f" FROM AnlagenDaten WHERE (Hybrid BETWEEN \'{line}#{value.current_time}\' AND \'{line}#{value.current_time.replace(minute = value.current_time.minute +1)}\') ORDER BY Stamp DESC limit 1 "
        curr_prod = pd.read_sql(query, conn.connectorMess)
        if curr_prod.empty:
            curr_prod = pd.Series(np.nan)
        production = pd.concat([production, curr_prod])
    production.reset_index(inplace=True, drop=True)
    df_quality = pd.concat([df_quality, production], axis = 1)
    df_quality.dropna(axis=1, inplace=True, how="all")
    df_quality.dropna(axis=0, inplace=True, how="any")
    df_quality.reset_index(inplace=True, drop=True)
    if config["Data"]["save_backup"]:
        df_quality.to_csv(os.path.join(config["Data"]["backup"], "production_colors_4_5_9.csv"))
        print("production saved")
    return df_quality 

def dowload_production_uwg(config, df_quality:pd.DataFrame)->pd.DataFrame:
    """ Download unique data from production """
    conn = SQLcust()
    column = "SELECT "
    production = pd.DataFrame()

    """ Getting the column names """
    for entry in config["Data"]["column_uwg"]:
        column = column + f"CAST({entry} AS VARCHAR(10)) AS {entry}, "
    column = column[:-2]
    """ Downloading data """
    for _, (_, value) in enumerate(df_quality.iterrows()):
        print(value)
        line = str(value["Line"]).replace("ZSK ","")
        query = column + f" FROM AnlagenDaten WHERE (Hybrid BETWEEN \'UWG{line}#{value.current_time}\' AND \'UWG{line}#{value.current_time.replace(minute = value.current_time.minute +1)}\') ORDER BY Stamp DESC limit 1 "
        curr_prod = pd.read_sql(query, conn.connectorMess)
        if curr_prod.empty:
            curr_prod = pd.Series(np.nan)
        else:
            print(curr_prod)
            print(query)
            print(value)
            break
    #     production = pd.concat([production, curr_prod])
    # production.reset_index(inplace=True, drop=True)
    # df_quality = pd.concat([df_quality, production], axis = 1)
    # df_quality.dropna(axis=1, inplace=True, how="all")
    # df_quality.dropna(axis=0, inplace=True, how="any")
    # df_quality.reset_index(inplace=True, drop=True)
    # if config["Data"]["save_backup"]:
    #     df_quality.to_csv(os.path.join(config["Data"]["backup"], "production_colors_uwg.csv"))
    #     print("production saved")
    # return df_quality 

def process_time(value):
    """ Get the value of time between a range """
    current_time = value.current_time
    previous_time = current_time - timedelta(minutes=10)
    next_time = current_time + timedelta(minutes=10)
    return previous_time, next_time

def dowload_production_uwg_multiple(config:dict, df_quality: pd.DataFrame)->pd.DataFrame:
    """ Download multiple uwg data """
    conn = SQLcust()
    column = "SELECT "
    production = pd.DataFrame()

    """ Getting column names """
    for entry in config["Data"]["column_uwg"]:
        column = column + f"CAST({entry} AS VARCHAR(10)) AS {entry}, "
    column = column[:-2]
    """ Downloading data """
    for _, (_, value) in enumerate(df_quality.iterrows()):
        """ value : 
                - Produktname
                - Line
                - Charge
                - Uhrzeit
        """
        line = str(value["Line"]).replace("ZSK ","")
        previous_time, next_time=process_time(value)
        query = column + f" FROM AnlagenDaten WHERE (Hybrid BETWEEN \'UWG{line}#{previous_time}\' AND \'UWG{line}#{next_time}\') ORDER BY Stamp DESC"
        curr_prod = pd.read_sql(query, conn.connectorMess)
        if curr_prod.empty:
            curr_prod = pd.Series(np.nan)
        else:
            curr_prod = curr_prod.astype("float")
            curr_prod = curr_prod.mean(axis=0)
            curr_prod = pd.DataFrame(data=curr_prod.values.reshape(-1,1).transpose(),columns=config["Data"]["column_uwg"])
        production = pd.concat([production, curr_prod])
    production.reset_index(inplace=True, drop=True)
    production.dropna(axis=1, inplace=True, how="all")
    df_quality = pd.concat([df_quality, production], axis = 1)
    df_quality.dropna(axis=1, inplace=True, how="all")
    df_quality.dropna(axis=0, inplace=True, how="any")
    df_quality.reset_index(inplace=True, drop=True)
    print(df_quality)
    if config["Data"]["save_backup"]:
        df_quality.to_csv(os.path.join(config["Data"]["backup"], "production_colors_uwg_mean.csv"))
        print("production saved")
    return df_quality 


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
    df_quality = run_cleaning_colors(config)
    df_quality = get_exact_date(df_quality)
    # Get only line 8
    df_quality = df_quality[df_quality.Line == "ZSK 70.8"]
    df_quality.reset_index(drop=True, inplace=True)
    # df_quality = dowload_production(config, df_quality)
    # dowload_production_uwg(config, df_quality)
    dowload_production_uwg_multiple(config, df_quality)
