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

 #TODO: set hyperparameters by user arguments
_hp = {
    'batch_size': 128,
    'lr': 3e-4,
    'num_epochs': 10
}

def main():
    # Arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #TODO: 추후 train 코드 분리

    # Dataset & Dataloader
    print("- Dataset loading ...")
    dataset = MelSpectDataset(DATA_DEBUG_PATH if args.debug else DATA_ROOT_PATH)
    dataloader = DataLoader(
        dataset, 
        batch_size=_hp['batch_size'], 
        shuffle=True,
        num_workers=get_num_workers()
    )
    logging_step = len(dataloader) // 20
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
        
            if (step + 1) % logging_step == 0:
                print(f"[Epoch {epoch + 1}/{_hp['num_epochs']}] Step {step  + 1}/{len(dataloader)} | loss: {total_loss/(step + 1): .3f}")


if __name__ == "__main__":
    main()