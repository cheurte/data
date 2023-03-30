from joblib import dump 
import os
import torch

def save_preprocessing(config, mm, ss)->None:
    dump(ss, os.path.join(config['visu']['output'],'ss.bin'), compress=True)
    dump(mm, os.path.join(config['visu']['output'],'mm.bin'), compress=True)

def save_fc_Model(model, config)->None:
    with open(os.path.join(config['visu']['output'],f"model_{config['model']['type']}.pt"), 'wb') as f:
        torch.save(model, f)
