from tqdm import tqdm
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from models.GraphSAGE import GraphSAGE as Model
from utils.dgraphfin import load_data, AdjacentNodesDataset
from utils.evaluator import Evaluator

# 设置gpu设备
device = 0
device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)

data = load_data('./datasets/632d74d4e2843a53167ee9a1-momodel/', 'DGraph', force_to_symmetric=True)
data = data.to(device)

model_params = {
    "h_c": 128,
    "num_layers": 3,
    "dropout": 0.25,
    "aggr": "mean",    
}

model = Model(
    in_c=20,
    out_c=2,
    ** model_params
)
model_desc = f'GraphSAGE-{"-".join([f"{k}_{v}" for k, v in model_params.items() ])}'
model_save_path = f'results/model-{model_desc}.pt'
model.load_state_dict(torch.load(model_save_path, map_location=device))

cache_path = f'./results/out-best-{model_desc}.pt'


def predict(data, node_id):
    if os.path.exists(cache_path):
        out = torch.load(cache_path, map_location=device)
    else:
        with torch.no_grad():
            model.eval()
            out = model(data.x, data.adj_t)

    pred = out[node_id].exp()
    return pred.squeeze(0)
