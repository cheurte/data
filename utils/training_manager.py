""" training manager for the colors """
import pandas as pd
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import argparse
import os, sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import read_json
from custom_dataset import CustomDataset
from save import save_preprocessing
from print_inputs import print_data_line
sys.path.append("/home/cheurte/Documents/data/models/")
from fc1 import linearModel as fc1
from fc2 import linearModel as fc2
from fc3 import linearModel as fc3
from fc4 import linearModel as fc4
from fcc3 import LinearModel as fcc3
from fcc4 import LinearModel as fcc4
from fcc5 import LinearModel as fcc5



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

def seperate_input_output(df: pd.DataFrame, output:str, *inputs: str):
    """ Allow to  seperate all the inputs from the outputs """
    if len([*inputs]) == 0:
        output_index = df.columns.drop(output)
        output = df.drop(columns=list(output_index.values))
        inputs = df.drop(columns=output)
    else:
        inputs = df.drop(columns= output)
        output = df.drop(columns=[*inputs])
    return inputs, output

def load_data(config_colors: dict):
    """ Load the data for the color """
    device = "cuda:0"
    df_colors = pd.read_csv(
        os.path.join(config_colors["Data"]["backup"],"production_colors_uwg_mean.csv"),
        usecols=config_colors["Data"]["columns_uwg_training"])
    
    df_colors= preprocessing_low(df_colors, 0.07, "A4", "A5")
    df_colors= preprocessing_high(df_colors, 0.95, "A5","A13")

 

    inputs, outputs = seperate_input_output(df_colors, "YI")

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
 
def chose_model(config:dict, device:str, df_colors:pd.DataFrame):
    if config['model']['type']=="fc1":
        return fc1(size_input = len(df_colors.columns)-1,
               num_class = config['model']['output_classes'],
               size_hidden = config['model']['size_hidden_unit']).to(device)
    elif config['model']['type']=="fc2":
        return fc2(size_input = len(df_colors.columns)-1,
               num_class = config['model']['output_classes'],
               size_hidden = config['model']['size_hidden_unit']).to(device)
    elif config['model']['type']=="fc3":
        return fc3(size_input = len(df_colors.columns)-1,
               num_class = config['model']['output_classes'],
               size_hidden = config['model']['size_hidden_unit']).to(device)
    elif config['model']['type']=="fc4":
        return fc4(size_input = len(df_colors.columns)-1,
               num_class = config['model']['output_classes'],
               size_hidden = config['model']['size_hidden_unit']).to(device)
    elif config['model']['type']=="fcc3":
        return fcc3(size_input = len(df_colors.columns)-1,
               num_class = config['model']['output_classes'],
               size_hidden = config['model']['size_hidden_unit']).to(device)
    elif config['model']['type']=="fcc4":
        return fcc4(size_input = len(df_colors.columns)-1,
               num_class = config['model']['output_classes'],
               size_hidden = config['model']['size_hidden_unit']).to(device)
    elif config['model']['type']=="fcc5":
        return fcc4(size_input = len(df_colors.columns)-1,
               num_class = config['model']['output_classes'],
               size_hidden = config['model']['size_hidden_unit']).to(device)

    else:
        assert ValueError("Wrong configuration")

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
