""" Simple script for basic stat """
import argparse
import os
# import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.preprocessing import StandardScaler

from utils import read_json


def preprocessing_quality(df: pd.DataFrame, column:str, min_val=None):
    return df[df[column] > min_val]

def preprocessing_low(df:pd.DataFrame, value_small_quantile= 0.05, *columns:str | int)->pd.DataFrame:
    """ Preprocess each entries for an existing dataframe """
    
    for column in columns:
        if isinstance(column, int):
            column = df.columns[column]
        print(column)
        lim = np.quantile(df[column], value_small_quantile)
        df = df[df[column] > lim]
    return df

def preprocessing_high(df:pd.DataFrame, value_small_quantile= 0.95, *columns:str | int)->pd.DataFrame:
    """ Preprocess each entries for an existing dataframe """
    for column in columns:
        if isinstance(column, int):
            column = df.columns[column]
        lim = np.quantile(df[column], value_small_quantile)
        df = df[df[column] < lim]

    return df

def print_simple(data: dict | pd.DataFrame)-> None:
    """ Print all the input compared to the 4 colors """
    if isinstance(data, dict):
        df_print = pd.read_csv(
            os.path.join(config["Data"]["backup"],"production_colors.csv"),
            usecols=config["Data"]["columns_uwg"])
        df_print = preprocessing_quality(df_print, "a", 0)
        df_print["YI"] = pd.to_numeric(df_print["YI"], downcast="float")
    else:
        df_print = data
    df_print = preprocessing_low(df_print, 0.07, 5, 15, 21)
    # df_print = preprocessing_low(df_print, 0.10, 17)
    # df_print = preprocessing_high(df_print, 0.95, "A11")
    # """ Printing """
    for i in range(5, len(df_print.columns)):

        # df_print = preprocessing(config, i)
        _, axes = plt.subplots(nrows=2, ncols=2, layout="constrained")
        axes[0, 0].plot(df_print[df_print.columns[i]], df_print[df_print.columns[1]], "*")
        axes[0, 1].plot(df_print[df_print.columns[i]], df_print[df_print.columns[2]], "*")
        axes[1, 0].plot(df_print[df_print.columns[i]], df_print[df_print.columns[3]], "*")
        axes[1, 1].plot(df_print[df_print.columns[i]], df_print[df_print.columns[4]], "*")

        axes[0, 0].set_ylabel("L")
        axes[0, 1].set_ylabel("a")
        axes[1, 0].set_ylabel("b")
        axes[1, 1].set_ylabel("YI")

        axes[0, 0].set_xlabel(df_print.columns[i])
        axes[0, 1].set_xlabel(df_print.columns[i])
        axes[1, 0].set_xlabel(df_print.columns[i])
        axes[1, 1].set_xlabel(df_print.columns[i])

        plt.title("".join([config["Data"]["columns_normal"][i], str(i)]))

        plt.show()

def print_qqplot(df:pd.DataFrame)->None:
    r""" Plot the QQ plot with the Shapiro Wilk test """
    for _ ,(_, item) in enumerate(df.items()):
        k2, p = stats.normaltest(item.values)
        values = np.squeeze(StandardScaler().fit_transform(np.array(item.values).reshape(-1, 1)))
        sm.qqplot(values, line="45")
        plt.title(f"{item.name} \n p = {p}, stat = {k2}")
        plt.show()

def list_entries(config: dict)-> None:
    """ List all entries with the corresponding indice """
    df_list = pd.read_csv(
        os.path.join(config["Data"]["backup"],"production_colors.csv"),
        usecols=config["Data"]["columns_stats"])

    _ = [print(i, col) for i, col in enumerate(df_list.columns)]

def print_relationship( df: pd.DataFrame, *columns: str ):
    """ Plot graph between two variables """
    plt.plot(df[columns[0]], df[columns[1]], "*")
    plt.title(f"Relation bewteen {columns[0]} and {columns[1]}")
    plt.xlabel(columns[0])
    plt.ylabel((columns[1]))
    plt.show()

def print_independant(df:pd.DataFrame, scale:bool ,*columns: str | int):
    """ plot data on the same line """

    _, axes = plt.subplots(nrows=1, ncols=2, layout="constrained")
    for column in columns:

        if isinstance(column, int):
            column = df.columns[column]
        if scale:
            serie = StandardScaler().fit_transform(df[column].values.reshape(-1,1))
        else:
            serie = df[column]

        axes[0].scatter(np.arange(len(df[column])), serie)
        axes[0].legend([df.columns[column] for column in columns])

    axes[1].scatter(df[df.columns[columns[1]]],df[df.columns[columns[0]]] )
    axes[1].set_xlabel(df.columns[columns[1]])
    axes[1].set_ylabel(df.columns[columns[0]])

    plt.legend([df.columns[column] if isinstance(column, int) else column for column in columns])
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()

if __name__=="__main__":
    if "win" in sys.platform:
        default_config = "C:\\Users\\corentin.heurte\\Documents\\data\\config\\config_win.json"
    elif "linux" in sys.platform:
        default_config = "/home/cheurte/Documents/data/config/config_2.json"
    else:
        raise ValueError ("On which os are you ?")

    parser = argparse.ArgumentParser("Basic config")
    parser.add_argument("--config", "-c", default=default_config)
    config = read_json(parser.parse_args().config)

    df= pd.read_csv(
        os.path.join(config["Data"]["backup"],"production_colors_4_5_9.csv"),
        usecols=config["Data"]["columns_normal"])
    # print(df)
    print_simple(df)
    # df = df[df.Line == "ZSK 70.8"]
    # df = preprocessing_quality(df, "a", 0)
    # df["YI"] = pd.to_numeric(df["L"], downcast="float")
    # df = preprocessing_low(df, 0.05, 5)
    # df = preprocessing_high(df, 0.95, 10)
    # df= preprocessing_low(df, 0.07, 5, 10, 16)
    # df= preprocessing_high(df, 0.95, 15, 11, 16)
    # print(df)
    # for i, column in enumerate(df.columns):
    #     if i<1:
    #         continue
    #     s = gdf[column].values + 1)
    #     pd.Series(s).hist()
    #     plt.title(column)
    #     plt.show()
        # if i==0:
        #     continue
        # print(f"{column} :\
        #     mean : {df[column].mean()} \
        #     std : {df[column].std()}, \
        #     variance : {df[column].var()}")

    # df = preprocessing_low(df, 0.05, 7)
    # df = preprocessing_high(df, 0.98, 5)
    # print(df)
    # print_simple(df)
    # print_qqplot(df)
