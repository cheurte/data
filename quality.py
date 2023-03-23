""" This file is only about dowloading the quality data. """
import pandas as pd
import numpy as np
# import datetime

import argparse
import sys 
import re
# from tqdm import tqdm
 
from utils import read_json
from charge import connect_header_data
from enemeter import get_data_enemeter
from db import SQLcust

def dowload_quality(df_header: pd.DataFrame)-> pd.DataFrame:
    """Dowload the quality thanks to the headers data (data from enermter & charge) """
    conn = SQLcust()
    query = list()
    [query.append(str(val)) for val in df_header.charge]
    query = f"SELECT CAST(qs_merkmal_id as VARCHAR(10)) as merkmal_id ,CAST(wertnum AS VARCHAR(10)) as wertnum, bemerkg, csnr FROM qs_ppmesswert where csnr={' or csnr='.join(query)}"
    df_quality = pd.read_sql(query, conn.connectorQS)
    df_quality.dropna(inplace=True, how="any", axis=0)
    return df_quality

def get_colors(config)->pd.DataFrame:
    """ Read and return the quality data """
    return pd.read_csv(
        config["Data"]["colors"],
        usecols=["Produktname","Uhrzeit","Line", "Charge", "L", "a", "b", "YI"],
        encoding="utf-8", 
        encoding_errors="backslashreplace")

def clean_general(df:pd.DataFrame, colum:str, uniform_size=None, max_size=None, min_size=None, *unwanted_values:str)->pd.DataFrame:
    """ Allow to clean a dataframe by deleting the unwanted values on a certain column, a size can be added if necessary """
    for _, row in df.iterrows():
        for unwanted in unwanted_values:
            if unwanted.lower() in str(row[colum]).lower():
                df[df[colum] == row[colum]] = np.nan
        if uniform_size is not None:
            if len(str(row[colum])) != uniform_size:
                df[df[colum] == row[colum]] = np.nan
        if max_size is not None:
            if len(str(row[colum])) > max_size:
                 df[df[colum] == row[colum]] = np.nan
        if min_size is not None:
            if len(str(df[colum])) < min_size:
                 df[df[colum] == row[colum]] = np.nan
    df.dropna(how="any", inplace=True, axis=0)
    df.reset_index(drop = True, inplace=True)
    return df

def clean_general_reverse(df: pd.DataFrame, colum:str, wanted_value:str)-> pd.DataFrame:
    """ Allow also to clean but by defining only the wanted value.  """
    for _, row in df.iterrows():
        res = re.search(wanted_value, row[colum])
        if res is None:
            df[df[colum] == row[colum]] = np.nan
    df.dropna(how="any", inplace=True, axis=0)
    df.reset_index(drop = True, inplace=True)
    return df

def connect_colors_headers(df_header: pd.DataFrame, df_quality: pd.DataFrame)->pd.DataFrame:
    """ Connect quality data with headers data """
    start_time = pd.DataFrame()
    end_time = pd.DataFrame()
    for _, value in df_quality.iterrows():
        """ Looking first with the charge number, then with line """
        df_curr = df_header[df_header["charge"] == np.int32(value.Charge)]
        df_curr = df_curr[df_curr['Kst Id'] == value.Line.replace(".", "/")]
        if not df_curr.empty :
            start_time = pd.concat([start_time, df_curr.Beginn])
            end_time = pd.concat([end_time, df_curr.Ende])
        else:
            # print(value)
            start_time = pd.concat([start_time, pd.Series(np.nan)])
            end_time = pd.concat([end_time, pd.Series(np.nan)])

    """ Reset start and end """
    start_time.reset_index(drop=True, inplace=True)
    start_time.rename(columns={start_time.columns[0]:"Beginn"}, inplace=True)

    end_time.reset_index(drop=True, inplace=True)
    end_time.rename(columns={end_time.columns[0]:"Ende"}, inplace=True)

    """ Create output """
    df_quality = pd.concat([df_quality, start_time], axis=1)
    df_quality = pd.concat([df_quality, end_time], axis=1)
    df_quality.dropna(inplace=True, axis=0, how="any") 
    df_quality.reset_index(drop=True, inplace=True)
    return df_quality

def run_cleaning_colors(config)->pd.DataFrame:
    """ Get the data from the raw file """
    df_raw = get_colors(config)
    df_raw.drop_duplicates(inplace=True, ignore_index=True)
    df_raw = clean_general(df_raw, "Line", None, None, None, "m", "o", "a", "°", "+", "e", "..", ":")
    df_raw = clean_general_reverse(df_raw, "Produktname", "BP 110/02")
    df_raw = clean_general(df_raw, "Produktname", None, None, None, "EXP")
    df_raw = clean_general(df_raw, "Charge", 8, None, None, "-", "A", "N", ".", "U")
    df_raw = clean_general(df_raw, "Uhrzeit", None, 5, None, "a", "e", "i")
    """ Connection of the production with the quality """
    df_header = connect_header_data(config,get_data_enemeter(config))
    df_out = connect_colors_headers(df_header, df_raw)
    return df_out

if __name__=="__main__":
    if "win" in sys.platform:
        default_config = "C:\\Users\\corentin.heurte\\Documents\\data\\config\\config_win.json"
    elif "linux" in sys.platform:
        default_config = "/home/cheurte/Documents/biotec/sql/config/config.json"
    else:
        raise ValueError ("On which os are you ?")

    parser = argparse.ArgumentParser("download and clean quality data")
    parser.add_argument("--config", "-c", default=default_config)
    args = parser.parse_args()
    config = read_json(args.config)
    df = run_cleaning_colors(config)
    # print(df)

