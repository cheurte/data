""" Simple script for basic stat """
import argparse
import os
from ssl import DefaultVerifyPaths
import sys
from matplotlib import use

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr, pearsonr, normaltest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

from utils import read_json

def preprocessing_quality(df: pd.DataFrame, column:str, min_val=None):
    return df[df[column] > min_val]

def preprocessing_low(df:pd.DataFrame, value_small_quantile= 0.05, *columns:str | int)->pd.DataFrame:
    """ Preprocess each entries for an existing dataframe """
    
    for column in columns:
        if isinstance(column, int):
            column = df.columns[column]
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

    # Printing
    for i in range(5, len(df_print.columns)):

        # df_print = preprocessing(config, i)
        _, axes = plt.subplots(nrows=2, ncols=2, layout="constrained")

        # plt.title(config["Data"]["columns_uwg_names"][i])
        plt.title(str(i))
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

        plt.show()

def print_qqplot(df:pd.DataFrame)->None:
    r""" Plot the QQ plot with the Shapiro Wilk test """
    for _ ,(_, item) in enumerate(df.items()):
        k2, p = normaltest(item.values)
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

def interquantile_range(df:pd.DataFrame):
    for column in df.columns:
        try:
            plt.boxplot(df[column])
            plt.title(column)
            plt.show()
        except BaseException as _:
            continue


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
    df = pd.read_csv(
        os.path.join(config["Data"]["backup"],"production_colors_uwg_mean_normal.csv"),
        usecols=config["Data"]["columns_uwg_normales"])
    # df = df[df.Line=="ZSK 69.8"]

####################################################
# preprocessing
###################################################
    # for line 8 not uwg
    # df= preprocessing_low(df, 0.10, 5, 9, 10, 15, 33)
    # df = preprocessing_high(df, 0.95, 5, 11)
    # df= preprocessing_low(df, 0.10, 5,  9)
    # df= preprocessing_low(df, 0.05,   9)
    #
    # df= preprocessing_high(df, 0.90, 5, 7, 8, 9, 10, 11, 12, 13)
    # df= preprocessing_high(df, 0.90, 7)


    # not 8 not uwg
    # df= preprocessing_low(df, 0.10, 5, 17)
    # df= preprocessing_high(df, 0.95,13, 14,  16)

    # For not line 8 not uwg
    # df = preprocessing_low(df, 0.13, 5)
    # df = preprocessing_low(df, 0.05, 5, 11, 14, 16)
    # df = preprocessing_high(df, 0.95, 6, 9, 18)
    # df = preprocessing_high(df, 0.90, 7, 18)


    # for mean uwg line 8
    # df= preprocessing_low(df, 0.07, "A0", "A2","A12", "A18")
    # df= preprocessing_high(df, 0.95, "A0","A5","A6")

    # for mean uwg line 8 but only when a few inputs in     
    # df = preprocessing_low(df, 0.07, "A4", "A5")
    # df = preprocessing_high(df, 0.95, "A5", "A13")

    # Special with chosen inputs
    # df= preprocessing_low(df, 0.06, "A4", "A14")

#####################################################
# print simple
#####################################################
    # print_simple(data=df)
#####################################################
# simple regression
#####################################################
    column_tested = [
            'Temperatures_Reg11_Sps_Istwert',
            'Temperatures_Reg12_Sps_Istwert',
            'Misc_Hat_Sps_Drehmoment_Istwert',
            'Feeder_Dos04_Sps_MotorStellwert',
            'Feeder_Dos02_Sps_Dosierfaktor',
            'Feeder_Dos04_Sps_MotorDrehzahl',
            'Feeder_Dos05_Sps_MotorStellwert',
            'Feeder_Dos04_Sps_Dosierfaktor',
            'Feeder_Dos05_Sps_Dosierfaktor',
            'Feeder_Dos05_Sps_MotorDrehzahl',
            "A2",
            "A4",
            "A5",
            "A13",
            "A26" ,
            "A14",
            "A15",
            "A16"
            ]
    ss = StandardScaler()
    mm = MinMaxScaler()
    #
    df_train = df[:np.int32(len(df)*0.8)]
    df_test = df[np.int32(len(df)*0.8):]

    X = df_train[column_tested]
    y = df_train["YI"]
    #
    X = ss.fit_transform(X)
    y = mm.fit_transform(y.values.reshape(-1,1))
    #
    X_test = df_test[column_tested]
    y_test = df_test["YI"]
    X_test = ss.fit_transform(X_test)
    #
    svr_rbf = SVR(kernel="rbf", C=200 , gamma="auto", degree=3, epsilon=1e-5)#, tol=1e-5)
    svr_lin = SVR(kernel="linear", C=100, gamma="auto")
    svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=2, epsilon=1e-1, coef0=1)
    #
    model = svr_rbf.fit(X,np.squeeze(y))
    # model = svr_poly.fit(X, np.squeeze(y))
    # # model = svr_lin.fit(X, np.squeeze(y))
    fig, axes = plt.subplots(nrows=2, ncols=2)
    pred = model.predict(X_test)

    pred = mm.inverse_transform(pred.reshape(-1,1))
    pred_train = model.predict(X)
    pred_train = mm.inverse_transform(pred_train.reshape(-1,1))


    axes[0, 0].set_title(mean_squared_error(mm.inverse_transform(y), pred_train))
    axes[0, 0].plot(np.arange(len(pred_train)), mm.inverse_transform(y), "*")
    axes[0, 0].plot(np.arange(len(pred_train)), pred_train, "*")
    axes[0, 0].legend(["real values", "prediction"])

    axes[0, 1].set_title(mean_squared_error(y_test.values,np.squeeze(pred)))
    axes[0, 1].plot(np.arange(len(pred)), y_test, "*")
    axes[0, 1].plot(np.arange(len(pred)), pred, "*")
    axes[0, 1].legend(["real values", "prediction"])

    regr = LinearRegression().fit(X, y)
    pred = mm.inverse_transform(regr.predict(X_test))
    pred_train = regr.predict(X)
    pred_train = mm.inverse_transform(pred_train.reshape(-1,1))

    axes[1, 0].set_title(mean_squared_error(mm.inverse_transform(y), pred_train))
    axes[1, 0].plot(np.arange(len(pred_train)), mm.inverse_transform(y), "*")
    axes[1, 0].plot(np.arange(len(pred_train)), pred_train, "*")
    axes[1, 0].legend([ "real value", "prediction"])

    axes[1, 1].set_title(f"{mean_squared_error(y_test.values, np.squeeze(pred))}")
    axes[1, 1].plot(np.arange(len(pred)), y_test, "*")
    axes[1, 1].plot(np.arange(len(pred)), pred, "*")
    axes[1, 1].legend([ "real value","prediction"])
    plt.show()
    #
####################################################
# plot linear regression
###################################################

    # out = pd.DataFrame(columns=["Column", "Distribution", "Relation", "r_squared"])
    # for i, col in enumerate(df.columns[5:]):
    #     x = np.squeeze(StandardScaler().fit_transform(df[col].values.reshape(-1, 1)))
    #     y = np.squeeze(StandardScaler().fit_transform(df["YI"].values.reshape(-1, 1)))
    #
    #     if len(np.unique(x))==1:
    #         print(df.columns[i])
    #         continue
    #
    #     k2, p = normaltest(x)
    #     sm.qqplot(x, line="45")
    #     plt.title(col + "stat" + str(k2) + "pval : "+ str(p))
    #     plt.show()
    #
    #     slope, intercept, rvalue, pvalue, stderr= linregress(x, y)
    #     stat_sp, pval_sp = spearmanr(x, y)
    #     stat_p, pval_p = pearsonr(x, y)
    #     plt.figure(figsize=(10,10))
    #     plt.plot(x, y, 'o', label='original data')
    #     plt.plot(x, intercept + slope*x, 'r', label='fitted line')
    #     # plt.xlabel(df.columns[i])
    #     plt.xlabel(col)
    #     plt.ylabel('YI')
    #     plt.title(col+" - " +"plot for each YI" +"\nspearman stat:"+str(stat_sp)+" pval:"+ str(pval_sp) +"\npearson stat:"+str(stat_p) + "pval:" + str(pval_p))
    #     plt.show()
    #     plt.clf()
    #     norm = p
    #     if pval_p< 0.05 or pval_sp<0.05:
    #         relation = True
    #         relation_pow = np.square(np.maximum(np.abs(np.float32(stat_p)),np.abs(np.float32(stat_sp))))*100
    #     else:
    #         relation = False
    #         relation_pow=np.square(np.maximum(np.abs(np.float32(stat_p)),np.abs(np.float32(stat_sp))))*100
    #
    #     out.loc[i] = [col, norm, relation, relation_pow]
    # out.to_csv("data/lineNot8_notUwg.csv")
