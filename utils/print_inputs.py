import numpy as np
import torch
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import sys,os
import argparse

sys.path.append("/home/cheurte/Documents/data/")

from utils import read_json

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

def print_data_line(data: pd.DataFrame, config:dict)->None:
    r"""
    # Function to print data point per line of production

    # Argument : 

        data: 
            dataframe of data, with the output as index

        config :
            config file for saving results

    """
    data.plot(subplots=True, layout=(2, 3), style="*")

    plt.savefig(os.path.join(config["visu"]["output"],f"inputs.png"))

def evaluate(config, testDataLoader, model)->pd.DataFrame:
    out = list()
    model.eval()

    with open(f"{config['visu']['output']}mm.bin",'rb') as f:
        mm = joblib.load(f)
    # print(mm.get_feature_names_out())
    y_ = list()
    with torch.no_grad():
        for (x,y) in testDataLoader:
            y_.append(torch.detach(y).to('cpu').numpy())
            outputs = model.forward(x)
            out.append(torch.detach(outputs).to('cpu').numpy()) 
    out = np.squeeze(np.array(out))
    y = np.array(y_)
    y = mm.inverse_transform(y.reshape(-1,1))

    y_hat = mm.inverse_transform(out.reshape(-1,1))
    df = pd.DataFrame({'y':np.squeeze(y), 'y_hat':np.squeeze(y_hat), 'mean':np.ones((len(out)))*np.mean(y), 'median': np.ones((len(out)))*np.median(y)})

    return df

def print_during_training( config, 
                epoch, 
                file_number,
                model,
                testDataLoader):
    
    path = os.path.join(config['visu']['output'],config['model']['type'])
    
    if not os.path.exists(path):
        os.makedirs(path)

    df_result = evaluate(config, testDataLoader, model)
    
    x = np.arange(df_result.shape[0])
    fig = plt.figure(figsize=(8,6))
    plt.plot(x, df_result.y,'*')
    plt.plot(x, df_result.y_hat, 'o')

    plt.hlines(np.unique(df_result['mean']),0,len(df_result.y_hat))
    plt.hlines(np.unique(df_result['median']),0,len(df_result.y_hat),'r')

    plt.legend(['y','y_pred'])
    plt.title(f'iteration : {str(epoch)}')
    plt.ylabel('Value of 197')
    plt.xlabel('Test set')
    title = '0'*(6-len(str(file_number)))+str(file_number)
    fig.savefig(f'{path}/{title}.png', bbox_inches='tight')
    fig.clf()


if __name__=="__main__":
    if "win" in sys.platform:
        default_config = "C:\\Users\\corentin.heurte\\Documents\\data\\config\\config_win.json"
    elif "linux" in sys.platform:
        default_config = "/home/cheurte/Documents/data/config/config_train_demo.json"
    else:
        raise ValueError ("On which os are you ?")

    parser = argparse.ArgumentParser("Config path mostly")
    parser.add_argument("--config", "-c", default=default_config)
    
    config_colors = read_json(parser.parse_args().config)

    df_colors = pd.read_csv(
        os.path.join(config_colors["Data"]["backup"],"production_colors_uwg.csv"),
        usecols=config_colors["Data"]["columns_uwg_training"],
        index_col=0)
    # print(df_colors) 
    df_colors = preprocessing_low(df_colors, 0.07, "A2", "A5")
    df_colors = preprocessing_high(df_colors, 0.95, "A11")
    print_data_line(df_colors, config_colors)
   
