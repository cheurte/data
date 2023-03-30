""" training manager for the colors """
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import os, sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler


sys.path.append("/home/cheurte/Documents/data/")
sys.path.append("/home/cheurte/Documents/data/models/")

from utils import read_json
from custom_dataset import CustomDataset
from save import save_preprocessing
from print_inputs import print_data_line

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

def load_data(config_colors: dict):
    device = "cuda:0"
    df_colors = pd.read_csv(
        os.path.join(config_colors["Data"]["backup"],"production_colors_uwg.csv"),
        usecols=config_colors["Data"]["columns_uwg_training"])
    df_colors = preprocessing_low(df_colors, 0.07, "A11", "A14")
    df_colors = preprocessing_high(df_colors, 0.95, "A11")
    
    outputs = df_colors.drop(columns=["A11", "A14"])
    inputs = df_colors.drop(columns= "YI")    

    ss = StandardScaler()
    mm = MinMaxScaler()
    inputs_ss = ss.fit_transform(inputs)
    outputs_mm = mm.fit_transform(outputs)

    save_preprocessing(config=config_colors, mm=mm, ss=ss)

    size_training_set = np.int32(0.8*len(inputs))

    inputs_train       = inputs_ss[:size_training_set]
    outputs_train      = outputs_mm[:size_training_set]
    inputs_validation  = inputs_ss[size_training_set:]
    outputs_validation = outputs_mm[size_training_set:]

    print_data_line(data=df_colors, config=config_colors)

    inputs_train_tensors        = Variable(torch.Tensor(inputs_train))
    inputs_validation_tensors  = Variable(torch.Tensor(inputs_validation))
    outputs_train_tensors      = Variable(torch.Tensor(outputs_train))
    outputs_validation_tensors = Variable(torch.Tensor(outputs_validation))

    train_dataset    = CustomDataset(inputs_train_tensors.to(device), outputs_train_tensors.to(device))
    train_data_loader = DataLoader(train_dataset, batch_size=config_colors['Data']['batch_size'], shuffle=config_colors['Data']['shuffle'])

    validation_dataset    = CustomDataset(inputs_validation_tensors.to(device), outputs_validation_tensors.to(device))
    validation_data_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    return train_data_loader, validation_data_loader, df_colors

def train(model, criterion, optimizer, trainDataLoader, train_loss):
    for (x,y) in trainDataLoader:
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    return train_loss 

def eval(model, validate_loader, criterion, validation_loss):
    model.eval()
    with torch.no_grad():
        for (x,y) in validate_loader:
            output = model(x)
            validation_loss+= criterion(output, y).item()
    return validation_loss
 
if __name__=="__main__":
    if "win" in sys.platform:
        default_config = "C:\\Users\\corentin.heurte\\Documents\\data\\config\\config_win.json"
    elif "linux" in sys.platform:
        default_config = "/home/cheurte/Documents/data/config/config_train_demo.json"
    else:
        raise ValueError ("On which os are you ?")

    parser = argparse.ArgumentParser("Config path mostly")
    parser.add_argument("--config", "-c", default=default_config)
    
    config = read_json(parser.parse_args().config)
    valid_data, valid_data, df_colors  = load_data(config)