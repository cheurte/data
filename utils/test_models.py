import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

from training_manager import *
from print_inputs import evaluate

def test_model(config, save_result = True):

    assert os.path.exists(os.path.join(config['visu']['output'], f"model_{config['model']['type']}.pt")), "No model trained, maybe create one first"
    assert torch.cuda.is_available()
    with open(os.path.join(config['visu']['output'], f"model_{config['model']['type']}.pt"), 'rb') as f:
        model = torch.load(f)
    model.eval()
    print(model.parameters)
    _, valid_data_loader, _ = load_data(config)   

    df_result = evaluate(config, valid_data_loader, model)
    
    x = np.arange(df_result.shape[0]) 

    plt.plot(x, df_result.y,'*')
    plt.plot(x, df_result.y_hat, 'o')

    plt.hlines(np.unique(df_result['mean']),0,len(df_result.y_hat))
    plt.hlines(np.unique(df_result['median']),0,len(df_result.y_hat),'r')

    plt.legend(['y','y_pred'])
    
    plt.ylabel('Value of YI')
    plt.xlabel('Test set')

    path = os.path.join(config['visu']['output'],config['visu']['name_model'])
    if save_result:
        plt.savefig(f'{path}.png')

    print(f"MSE : {np.sqrt(MSE(df_result.y, df_result.y_hat))}\tMSE to beat : {np.sqrt((MSE(df_result.y, df_result['mean'])))}")
    print(f"MAE : {np.sqrt(MAE(df_result.y, df_result.y_hat))}\tMAE to beat : {np.sqrt((MAE(df_result.y, df_result['mean'])))}")

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

    test_model(config, True)
