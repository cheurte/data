import argparse
import shutil
import sys
# from tqdm import tqdm
import torch
import torch.nn as nn
import os
from time import time
import matplotlib.pyplot as plt
from pandas import DataFrame

from utils import read_json
sys.path.append("/home/cheurte/Documents/data/utils/")
from training_manager import load_data, train, eval, chose_model
from print_inputs import print_during_training
from plot_loss import plot_loss
from create_video import create_video
from save import save_fc_Model
from test_models import test_model

"""
To start a training run the command : 
nohup python -u <path/to/sript> --args > output.log & 
"""

def train_fc(config)->None:
    '''
    Start this command to launch a training in background.
    nohup python -u process/train.py -c process/config/fc/config_fc_i.json  > log/fci.log &
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(config['visu']['output']):
        os.makedirs(config['visu']['output'])

    train_loader, validate_loader, df_colors = load_data(config_colors=config)

    model = chose_model(config, device, df_colors)
    criterion = nn.MSELoss() # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    print('TRAINING PHASE')
    file_number = 0
    train_loss  = 0.0
    validation_loss = 0.0
    loss = list()

    start = time()
    # last_loss_epoch = 0
    # try:
    for epoch in range(config['training']['max_epoch']):
        model.train()
        train_loss = train(model=model,
                           criterion=criterion,
                           optimizer= optimizer,
                           trainDataLoader=train_loader,
                           train_loss=train_loss)

        validation_loss = eval(model=model,
                               validate_loader=validate_loader,
                               criterion=criterion,
                               validation_loss=validation_loss)

        if epoch %1000 == 999:
            print(f'Train Epoch: {epoch}\tTrain loss: {train_loss/len(train_loader)}\tValidation Loss : {validation_loss/len(validate_loader)}')
            loss.append((train_loss, validation_loss))
                        
            # if epoch > 5000:
            #     if validation_loss > last_loss_epoch:
            #         print("Breaking early")
            #         break

            # last_loss_epoch = validation_loss
            train_loss = 0.0
            validation_loss = 0.0

        if epoch%20 == 19:
            print_during_training(config, epoch, file_number, model, validate_loader)
            file_number += 1
            plt.close("all") 
        

    plot_loss(config=config, df_loss=DataFrame(loss, columns=['Train_loss', 'Validate_loss']))
    plt.clf()
    end=time()
    create_video(os.path.join(config['visu']['output'], config['model']['type']), config['visu']['name_model'])   
    # Delete images
    if os.path.exists(os.path.join(config['visu']['output'],config['model']['type'])):
        shutil.rmtree(os.path.join(config['visu']['output'],config['model']['type']))

    save_fc_Model(model=model, config=config)
    print("Test of the model")
    test_model(config, True)
    print(f"time taken : {end-start}")
    
if __name__=='__main__':
    if "win" in sys.platform:
        default_config = "C:\\Users\\corentin.heurte\\Documents\\data\\config\\config_win.json"
    elif "linux" in sys.platform:
        default_config = "/home/cheurte/Documents/data/config/config_train_demo.json"
    else:
        raise ValueError ("On which os are you ?")

    parser = argparse.ArgumentParser("Config path mostly")
    parser.add_argument("--config", "-c", default=default_config)
    config = read_json(parser.parse_args().config)

    train_fc(config)
