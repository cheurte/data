""" Simple script for basic stat """
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import pyperclip

# from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.svm import SVR
import statsmodels.api as sm
from scipy.stats import normaltest
from tqdm import tqdm

from utils import read_json


def preprocessing_quality(df: pd.DataFrame, column: str, min_val=None):
    return df[df[column] > min_val]


def preprocessing_low(df: pd.DataFrame, value_small_quantile=0.05, *columns: str | int) -> pd.DataFrame:
    """ Preprocess each entries for an existing dataframe """

    for column in columns:
        if isinstance(column, int):
            column = df.columns[column]
        lim = np.quantile(df[column], value_small_quantile)
        df = df[df[column] > lim]
    return df


def preprocessing_high(df: pd.DataFrame, value_small_quantile=0.95, *columns: str | int) -> pd.DataFrame:
    """ Preprocess each entries for an existing dataframe """
    for column in columns:
        if isinstance(column, int):
            column = df.columns[column]
        lim = np.quantile(df[column], value_small_quantile)
        df = df[df[column] < lim]

    return df


def print_simple(data: dict | pd.DataFrame) -> None:
    """ Print all the input compared to the 4 colors """
    if isinstance(data, dict):
        df_print = pd.read_csv(
            os.path.join(config["Data"]["backup"], "production_colors.csv"),
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
        axes[0, 0].plot(df_print[df_print.columns[i]],
                        df_print[df_print.columns[1]], "*")
        axes[0, 1].plot(df_print[df_print.columns[i]],
                        df_print[df_print.columns[2]], "*")
        axes[1, 0].plot(df_print[df_print.columns[i]],
                        df_print[df_print.columns[3]], "*")
        axes[1, 1].plot(df_print[df_print.columns[i]],
                        df_print[df_print.columns[4]], "*")

        axes[0, 0].set_ylabel("L")
        axes[0, 1].set_ylabel("a")
        axes[1, 0].set_ylabel("b")
        axes[1, 1].set_ylabel("YI")

        axes[0, 0].set_xlabel(df_print.columns[i])
        axes[0, 1].set_xlabel(df_print.columns[i])
        axes[1, 0].set_xlabel(df_print.columns[i])
        axes[1, 1].set_xlabel(df_print.columns[i])

        plt.show()


def print_qqplot(df: pd.DataFrame) -> None:
    r""" Plot the QQ plot with the Shapiro Wilk test """
    for _, (_, item) in enumerate(df.items()):
        k2, p = normaltest(item.values)
        values = np.squeeze(StandardScaler().fit_transform(
            np.array(item.values).reshape(-1, 1)))
        sm.qqplot(values, line="45")
        plt.title(f"{item.name} \n p = {p}, stat = {k2}")
        plt.show()


def list_entries(config: dict) -> None:
    """ List all entries with the corresponding indice """
    df_list = pd.read_csv(
        os.path.join(config["Data"]["backup"], "production_colors.csv"),
        usecols=config["Data"]["columns_stats"])

    _ = [print(i, col) for i, col in enumerate(df_list.columns)]


"""
Function to print the relation between two inputs particulary"""


def print_independant(df: pd.DataFrame, scale: bool, *columns: str | int) -> None:
    """ plot data on the same line """

    _, axes = plt.subplots(nrows=1, ncols=2, layout="constrained")
    for column in columns:

        if isinstance(column, int):
            column = df.columns[column]
        if scale:
            serie = StandardScaler().fit_transform(
                df[column].values.reshape(-1, 1))
        else:
            serie = df[column]

        axes[0].scatter(np.arange(len(df[column])), serie)
        axes[0].legend([df.columns[column] for column in columns])

    axes[1].scatter(df[df.columns[columns[1]]], df[df.columns[columns[0]]])
    axes[1].set_xlabel(df.columns[columns[1]])
    axes[1].set_ylabel(df.columns[columns[0]])

    plt.legend([df.columns[column] if isinstance(
        column, int) else column for column in columns])
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()


"""
Plotting box plot for all the inputs"""


def interquantile_range(df: pd.DataFrame) -> None:
    for column in df.columns:
        try:
            plt.boxplot(df[column])
            plt.title(column)
            plt.show()
        except BaseException as _:
            continue


"""
Plot the prediction by using the SVR predictor and the linear regression
predictor"""


def print_svr_linear(df_data: pd.DataFrame, scale_valid_test: float, c_reg: float, column: list, epsilon: np.float32) -> None:
    """ Print prediction with SVR compared to linear one """
    standart_scal = StandardScaler()
    min_max_scal = MinMaxScaler()

    df_train = df_data[:np.int32(len(df_data)*scale_valid_test)]
    df_test = df_data[np.int32(len(df_data)*scale_valid_test):]

    x_train = df_train[column]
    y_train = df_train["YI"]

    x_train = standart_scal.fit_transform(x_train)
    y_train = min_max_scal.fit_transform(y_train.values.reshape(-1, 1))

    x_test = df_test[column]
    y_test = df_test["YI"]
    x_test = standart_scal.fit_transform(x_test)
    y_test = min_max_scal.fit_transform(y_test.values.reshape(-1, 1))
#
    model = SVR(kernel="rbf", C=c_reg, gamma="scale",
                epsilon=epsilon.item()).fit(x_train, y_train.ravel())
    # svr_rbf = SVR(kernel="rbf")

    # model = SVR.svr_rbf.fit(x_train, np.squeeze(y_train))
    # model = MLPRegressor(random_state=1, max_iter=500, activation='tanh').fit(x_train, y_train)
    # model = DecisionTreeRegressor(random_state=0).fit(x_train, y_train)

    _, axes = plt.subplots(nrows=2, ncols=2)
    plt.suptitle(f"c : {c_reg}, epsilon : {epsilon}")
    pred = model.predict(x_test)

    # pred = min_max_scal.inverse_transform(np.array(pred).reshape(-1, 1))
    pred_train = model.predict(x_train)
    pred_train = min_max_scal.inverse_transform(
        np.array(pred_train).reshape(-1, 1))

    # axes[0, 0].set_title(mean_absolute_error(
    # min_max_scal.inverse_transform(y_train), pred_train))
    axes[0, 0].set_title(model.score(x_train, y_train))
    axes[0, 0].plot(np.arange(len(pred_train)),
                    min_max_scal.inverse_transform(y_train), "*")
    axes[0, 0].plot(np.arange(len(pred_train)), pred_train, "*")
    axes[0, 0].legend(["real values", "prediction"])

    df_pred = {"y": np.squeeze(y_test), "pred": np.squeeze(pred)}
    df_pred = pd.DataFrame(df_pred)
    df_pred = df_pred.sort_values(by=["y"])
    mean_y_test = y_test.mean()

    axes[0, 1].set_title(
        f"{model.score(x_test, y_test)}\n {r2_score(df_pred.y, np.ones(len(df_pred.y))*mean_y_test)}")
    axes[0, 1].plot(np.arange(len(pred)), df_pred.y, "*")
    axes[0, 1].plot(np.arange(len(pred)), df_pred.pred, "*")
    axes[0, 1].legend(["real values", "prediction"])

    regr = LinearRegression().fit(x_train, y_train)
    # pred = min_max_scal.inverse_transform(regr.predict(x_test))
    pred_train = regr.predict(x_train)
    pred_train = min_max_scal.inverse_transform(pred_train.reshape(-1, 1))

    axes[1, 0].set_title(regr.score(x_train, y_train))
    axes[1, 0].plot(np.arange(len(pred_train)),
                    min_max_scal.inverse_transform(y_train), "*")
    axes[1, 0].plot(np.arange(len(pred_train)), pred_train, "*")
    axes[1, 0].legend(["real value", "prediction"])

    df_pred = {"y": np.squeeze(y_test), "pred": np.squeeze(pred)}
    df_pred = pd.DataFrame(df_pred)
    df_pred = df_pred.sort_values(by=["y"])

    axes[1, 1].set_title(f"{regr.score(x_test, y_test)}, \n ")
    axes[1, 1].plot(np.arange(len(df_pred)), df_pred.y, "*")
    axes[1, 1].plot(np.arange(len(df_pred)), df_pred.pred, "*")
    axes[1, 1].legend(["real value", "prediction"])
    plt.show()


"""
Function to find the best c parameter for the support vector regression"""


def find_best_c_reg(df_data: pd.DataFrame, column_tested: list,  scale_valid_test: float, plot: bool, *range_for: int):
    standart_scal = StandardScaler()
    min_max_scal = MinMaxScaler()

    df_train = df_data[:np.int32(len(df_data)*scale_valid_test)]
    df_test = df_data[np.int32(len(df_data)*scale_valid_test):]

    x_train = df_train[column_tested]
    y_train = df_train["YI"]

    x_train = standart_scal.fit_transform(x_train)
    y_train = min_max_scal.fit_transform(y_train.values.reshape(-1, 1))

    x_test = df_test[column_tested]
    y_test = df_test["YI"]
    x_test = standart_scal.fit_transform(x_test)
    res = []
    for c in range(range_for[0], range_for[1]):

        svr_rbf = SVR(kernel="rbf", C=c, gamma="scale",
                      epsilon=1e-1)  # , tol=1e-5)

        model = svr_rbf.fit(x_train, np.squeeze(y_train))
        pred = model.predict(x_test)

        pred = min_max_scal.inverse_transform(pred.reshape(-1, 1))
        pred_train = model.predict(x_train)
        pred_train = min_max_scal.inverse_transform(pred_train.reshape(-1, 1))

        df_pred = {"y": np.squeeze(y_test.values), "pred": np.squeeze(pred)}
        df_pred = pd.DataFrame(df_pred)
        df_pred = df_pred.sort_values(by=["y"])

        res.append((mean_absolute_error(y_test.values, np.squeeze(pred)), c))
    res = np.squeeze(np.array(res))
    min_error = np.ndarray.min(res, axis=0)[0]
    c = np.where(res[:, 0] == min_error)[0]
    if plot:
        plt.plot(res[:, 1], res[:, 0])
        plt.show()
    return c[0]+1, min_error


"""
Find best epsilon"""


def find_best_epsilon(df_data: pd.DataFrame, column_tested: list,  scale_valid_test: float, plot: bool, c: int):
    standart_scal = StandardScaler()
    min_max_scal = MinMaxScaler()

    df_train = df_data[:np.int32(len(df_data)*scale_valid_test)]
    df_test = df_data[np.int32(len(df_data)*scale_valid_test):]

    x_train = df_train[column_tested]
    y_train = df_train["YI"]

    x_train = standart_scal.fit_transform(x_train)
    y_train = min_max_scal.fit_transform(y_train.values.reshape(-1, 1))

    x_test = df_test[column_tested]
    y_test = df_test["YI"]
    x_test = standart_scal.fit_transform(x_test)
    res = []
    for eps in range(1, 6):
        eps = 1/np.power(10, eps)

        svr_rbf = SVR(kernel="rbf", C=c, gamma="scale",
                      epsilon=eps)  # , tol=1e-5)

        model = svr_rbf.fit(x_train, np.squeeze(y_train))
        pred = model.predict(x_test)

        pred = min_max_scal.inverse_transform(pred.reshape(-1, 1))
        pred_train = model.predict(x_train)
        pred_train = min_max_scal.inverse_transform(pred_train.reshape(-1, 1))

        df_pred = {"y": np.squeeze(y_test.values), "pred": np.squeeze(pred)}
        df_pred = pd.DataFrame(df_pred)
        df_pred = df_pred.sort_values(by=["y"])

        res.append((mean_absolute_error(y_test.values, np.squeeze(pred)), eps))
    res = np.squeeze(np.array(res))
    min_error = np.ndarray.min(res, axis=0)[0]
    id = np.where(res[:, 0] == min_error)[0][0]
    if plot:
        plt.plot(res[:, 1], res[:, 0])
        plt.show()
    return res[:, 1][id]+0, min_error


"""
preprocessing, very specific function
"""


def preprocessing(type: int, df: pd.DataFrame) -> pd.DataFrame:
    """
    type :
    - 1 : Line 8 not uwg
    -2 : Not sure...
    - 3 : not 8 not uwg
    - 4 : mean uwg line 8
    - 5 : mean uwg line 8
    - 6 :  for mean uwg line 8 but only when a few inputs in
    - 7 : special with chosen inputs
    """
    if type == 1:
        # for line 8 not uwg
        df = preprocessing_low(df, 0.10, 5, 9, 10, 15, 33)
        df = preprocessing_high(df, 0.95, 5, 11)
        df = preprocessing_low(df, 0.10, 5,  9)
        df = preprocessing_low(df, 0.05,   9)
    elif type == 2:
        df = preprocessing_high(df, 0.90, 5, 7, 8, 9, 10, 11, 12, 13)
        df = preprocessing_high(df, 0.90, 7)

    elif type == 3:
        # not 8 not uwg
        df = preprocessing_low(df, 0.10, 5, 17)
        df = preprocessing_high(df, 0.95, 13, 14,  16)
    elif type == 4:
        # For not line 8 not uwg
        df = preprocessing_low(df, 0.13, 5)
        df = preprocessing_low(df, 0.05, 5, 11, 14, 16)
        df = preprocessing_high(df, 0.95, 6, 9, 18)
        df = preprocessing_high(df, 0.90, 7, 18)
    elif type == 5:
        # for mean uwg line 8
        df = preprocessing_low(df, 0.07, "A0", "A2", "A12", "A18")
        df = preprocessing_high(df, 0.95, "A0", "A5", "A6")
    elif type == 6:
        # for mean uwg line 8 but only when a few inputs in
        df = preprocessing_low(df, 0.07, "A4", "A5")
        df = preprocessing_high(df, 0.95, "A5", "A13")
    elif type == 7:
        # Special with chosen inputs
        df = preprocessing_low(df, 0.06, "A4", "A14")
    else:
        df = df
    return df


"""
cross validation with svr
"""


def cross_validation(df_data: pd.DataFrame, scale_valid_test: float, column: list) -> None:
    """ Print prediction with SVR compared to linear one """
    standart_scal = StandardScaler()
    min_max_scal = MinMaxScaler()

    df_train = df_data[:np.int32(len(df_data)*scale_valid_test)]
    df_test = df_data[np.int32(len(df_data)*scale_valid_test):]

    x_train = df_train[column]
    y_train = df_train["YI"]

    x_test = df_test[column]
    y_test = df_test["YI"]

    x_train = standart_scal.fit_transform(x_train)
    y_train = min_max_scal.fit_transform(y_train.values.reshape(-1, 1))
    x_test = standart_scal.fit_transform(x_test)
    y_test = min_max_scal.fit_transform(y_test.values.reshape(-1, 1))

    # degrees = np.arange(1, 10)
    # for degree in degrees:
    #     print(degree)
    #     model = SVR(kernel='poly', C=5.0, epsilon=1e-1, degree=degree)
    # model = MLPRegressor(random_state=1, max_iter=500)
    # model.fit(x_train, y_train.ravel())
    # score = model.score(x_test, y_test.ravel())
    # print(score)
    model = gpr().fit(x_train, y_train)
    scores = cross_val_score(
        model, x_train, np.squeeze(y_train), cv=5)
    print(f"%0.2f accuracy with a standard deviation of %0.2f" %
          (scores.mean(), scores.std()))
    # print(scores)


"""
polynomial regression
"""


def polynomial_regr(column_tested: list, df: pd.DataFrame):
    scale_valid_test = 0.8
    degrees = np.arange(1, 10)  # [1, 2, 3, 6, 10, 15, 20]
    # degrees = [3]
    column_tested = column_tested[5:]
    df_train = df[:np.int32(len(df)*scale_valid_test)]
    df_test = df[np.int32(len(df)*scale_valid_test):]
    ss = StandardScaler()
    x_train = df_train[column_tested]
    y_train = df_train["YI"]
    x_test = df_test[column_tested]
    y_test = df_test["YI"]

    x_train = ss.fit_transform(x_train)
    x_test = ss.fit_transform(x_test)

    y_train_pred = np.zeros((len(x_train), len(degrees)))
    y_test_pred = np.zeros((len(x_test), len(degrees)))

    for i, degree in tqdm(enumerate(degrees)):

        # make pipeline: create features, then feed them to linear_reg model
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(x_train, y_train)

        # predict on test and train data
        # store the predictions of each degree in the corresponding column
        y_train_pred[:, i] = model.predict(x_train)
        y_test_pred[:, i] = model.predict(x_test)

    plt.subplot(121)
    plt.scatter(np.arange(len(y_train)), y_train)
    plt.yscale('log')
    plt.title("Train data")
    for i, degree in enumerate(degrees):
        plt.scatter(np.arange(len(y_train_pred)),
                    y_train_pred[:, i], s=15, label=str(degree))
        plt.legend(loc='upper left')

    # test data
    plt.subplot(122)
    plt.scatter(np.arange(len(y_test)), y_test)
    plt.yscale('log')
    plt.title("Test data")
    for i, degree in enumerate(degrees):
        plt.scatter(np.arange(len(y_test_pred)),
                    y_test_pred[:, i], label=str(degree))
        plt.legend(loc='upper left')
    plt.show()

    print("R-squared values: \n")
    for i, degree in enumerate(degrees):
        train_r2 = r2_score(y_train, y_train_pred[:, i])
        test_r2 = r2_score(y_test, y_test_pred[:, i])
        print(
            f"Polynomial degree {degree}: train score={train_r2}, test score={test_r2}")

"""
Main function"""
if __name__ == "__main__":
    if "win" in sys.platform:
        default_config = "C:\\Users\\corentin.heurte\\Documents\\data\\config\\config_win.json"
    elif "linux" in sys.platform:
        default_config = "/home/cheurte/Documents/data/config/config_g21.json"
    else:
        raise ValueError("On which os are you ?")
    parser = argparse.ArgumentParser("Basic config")
    parser.add_argument("--config", "-c", default=default_config)
    config = read_json(parser.parse_args().config)
    df = pd.read_csv(
        os.path.join(config["Data"]["backup"],
                    "production_quality_bp880_mean_all.csv"),
                    usecols=config["Data"]["columns_normal"])
    # print(df.qs_merkmal_id.value_counts())
    for line in df.Line.unique():
        print(line)
        df_s = df[df.Line == line]
        for id in df_s.qs_merkmal_id.unique():
            df_ss = df_s[df_s.qs_merkmal_id == id]

            # break
        # break
            print(f"merkmal id : {id}")
            print(df_ss.info(False))
            print("-"*30)
        print("*"*30)
    # df : DataFrame
    # print(df.Line.unique())
    # print(df.Produktname.value_counts())
    # df = df[df.Produktname == "BP 110/02 (717221)"]
    #
    # column_tested = [
    #     "Line",
    #     "L",
    #     "a",
    #     "b",
    #     "YI",
    #     'Temperatures_Reg11_Sps_Istwert',
    #     'Temperatures_Reg12_Sps_Istwert',
    #     'Misc_Hat_Sps_Drehmoment_Istwert',
    #     'Feeder_Dos04_Sps_MotorStellwert',
    #     'Feeder_Dos02_Sps_Dosierfaktor',
    #     'Feeder_Dos04_Sps_MotorDrehzahl',
    #     'Feeder_Dos05_Sps_MotorStellwert',
    #     'Feeder_Dos04_Sps_Dosierfaktor',
    #     'Feeder_Dos05_Sps_Dosierfaktor',
    #     'Feeder_Dos05_Sps_MotorDrehzahl',
    #     "A2",
    #     "A4",
    #     "A5",
    #     "A13",
    #     "A14",
    #     "A15",
    #     "A16",
    #     "A26"
    # ]
#
    # df = df[column_tested]
    # df = preprocessing_low(df, 0.06, 7,"A5", "A26")
    # print_simple(df)
    # column_tested = column_tested[4:]
    # print(column_tested)
    # # c, _ = find_best_c_reg(df, column_tested, 0.8, False, 1, 200)
    # # eps, _ = find_best_epsilon(df, column_tested, 0.8, False, c)
    # print_svr_linear(df, 0.8, c, column_tested, np.float32(eps))
    # cross_validation(df, 0.8, column_tested)
    # polynomial_regr(column_tested, df)
    # print_svr_linear(df, 0.8, 5.0, column_tested, np.float32(0.1))
#    method :
#         - svr
#         - decisiontree
#         - gpr
#         - ai
#         - linear

#
#   ####################################################
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
