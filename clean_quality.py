import pandas as pd
import numpy as np
import re

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


df = pd.read_csv(
        "data/data_colors.csv",
        usecols=["Produktname","Uhrzeit","Produktionsanlage", "Charge", "L*(D65)", "a*(D65)", "b*(D65)", "YI(E313-98)(D65)"],
        encoding="utf-8", 
        encoding_errors="backslashreplace")
# new_df = []
# for _,row in df.iterrows():
#     # print(row)
#     if "BP 110/02" in row["Produktname"]:
#         new_df.append(row)
# df = pd.DataFrame(new_df)     
df.drop_duplicates(ignore_index=True, inplace=True)
df= clean_general(df, "Produktionsanlage", None, None, None, "m", "o", "a", "°", "+", "e", "..", ":")
df= clean_general_reverse(df, "Produktname", "BP 110/02")
df= clean_general(df, "Produktname", None, None, None, "EXP")
df= clean_general(df, "Charge", 8, None, None, "-", "A", "N", ".", "U")
df= clean_general(df, "Uhrzeit", None, 5, None, "a", "e", "i")
df.drop_duplicates(ignore_index=True, inplace=True)
# df = df[df.Produktionsanlage == "ZSK 70.8"]

df.to_csv("data_colors_cleaned.csv")
