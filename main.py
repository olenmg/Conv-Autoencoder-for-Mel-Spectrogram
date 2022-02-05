import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import set_seed, get_num_workers, get_arg_parser
from dataset import MelSpectDataset
from model import ConvAutoencoder


DATA_ROOT_PATH = '/data1/melon/arena_mel'
DATA_DEBUG_PATH = '/home/jhkim/workspace/sample_data/arena_mel'
SAVE_PATH = './ckpt'

 #TODO: set hyperparameters by user arguments
_hp = {
    'batch_size': 128,
    'lr': 3e-4,
    'num_epochs': 100,
    'logging_step': 10000,
}

def main():
    # Arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(SAVE_PATH, exist_ok=True)

    #TODO: 추후 train 코드 분리

    # Dataset & Dataloader
    print("- Loading dataset: This will take few minutes ... ")
    dataset = MelSpectDataset(DATA_DEBUG_PATH if args.debug else DATA_ROOT_PATH, args.debug)
    dataloader = DataLoader(
        dataset, 
        batch_size=_hp['batch_size'], 
        shuffle=True,
        num_workers=get_num_workers()
    )
    print(f"- Complete to load the dataset - size: {len(dataset)}")

    # Model & Loss & Optimizer
    print("- Model setting ...")
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_hp['lr'])

    # Train
    print("- Start to train ...")
    total_loss = 0.
    for epoch in range(_hp['num_epochs']):
        for step, data in tqdm(enumerate(dataloader)):
            data = data.to(device)
            
            optimizer.zero_grad()
            out, _ = model(data)
            loss = criterion(out, data)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        
            if (step + 1) % _hp['logging_step'] == 0:
                print(f"[Epoch {epoch + 1}/{_hp['num_epochs']}] Step {step  + 1}/{len(dataloader)} | loss: {total_loss/(step + 1): .3f}")
        
        if (epoch + 1) % 1 == 0:
            print(f"Success to save the model at epoch [{epoch + 1}/{_hp['num_epochs']}]")
            torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"epoch_{epoch + 1}.pt"))



if __name__ == "__main__":
    main()