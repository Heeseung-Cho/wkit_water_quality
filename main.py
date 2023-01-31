import os
import pickle
import random
import argparse

import wandb
import torch
from torch.utils.data import DataLoader
import numpy as np

from utils.train import fit, test
from models import DLinear
from data_provider.data_factory import data_provider

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


def main(args):
    use_wandb = args.use_wandb    
    project_wandb = "WKIT"
    run_name = f"DLinear_{args.data_path}_{args.target}_{args.seq_len}"    

    save_path = f'{args.checkpoints}/{args.data_path}'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f'CUDA is available. Your device is {device}.')
    else:
        print(f'CUDA is not available. Your device is {device}. It can take long time training in CPU.')    

    ## Dataloader
    train_ds, train_loader = data_provider(args, flag='train')
    _, vali_loader = data_provider(args, flag='val')    
    test_ds, test_loader = data_provider(args, flag='test')    

    args.enc_in = train_ds.x_data.shape[1]
    modelPath = os.path.join(save_path, f"DLinear_{args.target}_{args.seq_len}.pth")

    ## Model
    model = DLinear.Model(args)
    if use_wandb:
        wandb.init(project = project_wandb, config = args, name = run_name)
        wandb.watch(model, log="all")
    # train/validate now
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params = model.parameters(),lr = args.learning_rate, weight_decay= args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.5, patience = 3)                
    history = fit(epochs = args.train_epochs,
                    model = model,
                    train_loader = train_loader,
                    val_loader = vali_loader,
                    criterion = criterion, 
                    optimizer = optimizer, 
                    path = modelPath, 
                    scheduler= scheduler,
                    earlystop = 20,                      
                    device = device,
                    use_wandb = use_wandb)    
    with open(os.path.join(save_path, f"hist_{args.target}_{args.seq_len}.pkl"), "wb") as file:
        pickle.dump(history, file)    
    if use_wandb:
        wandb.finish()
    pred = test(model, test_loader, device=device)
    
    import pandas as pd
    df = pd.DataFrame([pred, test_ds.y_data], columns = range(10), index = [run_name+"_pred",run_name+"_real"])
    df.to_csv(os.path.join(save_path,args.target+"_prediction.csv"))
    
if __name__ == "__main__":    
    ## Argparse
    parser = argparse.ArgumentParser(description='WKIT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data loader
    parser.add_argument('--root_path', type=str, default="dataset/Water_Week", help='root path of the data file')
    parser.add_argument('--data_path', type=str, default="D001청주정", help='data file')    
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--target', type=str, default='TOC', help='target feature in S or MS task')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=120, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=1, help='start token length')
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

   # optimization
    parser.add_argument('--num_workers', type=int, default=4, help='data loader num workers')    
    parser.add_argument('--train_epochs', type=int, default=3, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')

    parser.add_argument('--use_wandb', type=int, default=0, help='(Option) logging on wandb')
    args = parser.parse_args()
    set_seed(42)
    main(args)
