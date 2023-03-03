import os
import pickle
import random
import argparse

import wandb
import torch
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import DLinear
from utils.train import fit, test
from utils.metrics import metric
from data_provider.data_factory_wq import data_provider


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


def main(args):
    # Setting Parameters
    DATA_PATH = args.data_path
    SAVE_PATH = f'{args.checkpoints}/{args.data_path}/{args.target}'
    if not os.path.exists(SAVE_PATH):
        os.mkdir(SAVE_PATH)
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f'CUDA is available. Your device is {DEVICE}.')
    else:
        print(f'CUDA is not available. Your device is {DEVICE}.')
    # Model Parameters
    TARGET = args.target
    SEQ = args.seq_len
    # Wandb Parameters
    USE_WANDB = args.use_wandb
    PROJECT_WANDB = "WKIT"
    RUN_NAME = f"DLinear_{DATA_PATH}_{TARGET}_{SEQ}"
    # Dataloader
    train_ds, train_loader = data_provider(args, flag='train')
    _, vali_loader = data_provider(args, flag='val')
    test_ds, test_loader = data_provider(args, flag='test')
    args.enc_in = train_ds.x_data.shape[1]
    MODEL_PATH = os.path.join(SAVE_PATH, f"DLinear_{TARGET}_{SEQ}.pth")

    # Model
    model = DLinear.Model(args)
    if USE_WANDB:
        wandb.init(project=PROJECT_WANDB, config=args, name=RUN_NAME)
        wandb.watch(model, log="all")
    # train/validate now
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.learning_rate
                                 )
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
    history = fit(epochs=args.train_epochs,
                  model=model,
                  train_loader=train_loader,
                  val_loader=vali_loader,
                  criterion=criterion,
                  optimizer=optimizer,
                  path=MODEL_PATH,
                  scheduler=scheduler,
                  earlystop=30,
                  device=DEVICE,
                  use_wandb=USE_WANDB
                  )
    with open(os.path.join(SAVE_PATH, f"hist_{TARGET}_{SEQ}.pkl"), "wb") as f:
        pickle.dump(history, f)
    if USE_WANDB:
        wandb.finish()

    # Metrics
    metrics_df = dict()
    ypred, ytrue = test(model, test_loader, device=USE_WANDB)
    mae, mse, rmse, mape, _ = metric(ypred, ytrue)
    print(f'MSE:{mse:.3f}, MAE:{mae:.3f}, MAPE:{mape:.3f}')
    metrics_df["metrics_scale"] = np.array([mae, mse, rmse, mape])
    if args.is_scale:
        ypred = test_ds.inverse_transform(ypred.reshape(-1, 1)).flatten()
        ytrue = test_ds.inverse_transform(ytrue.reshape(-1, 1)).flatten()
        mae, mse, rmse, mape, _ = metric(ypred, ytrue)
        metrics_df["metrics_real"] = np.array([mae, mse, rmse, mape])
    metrics_df = pd.DataFrame.from_dict(metrics_df,
                                        orient='index',
                                        columns=["mae", "mse", "rmse", "mape"]
                                        )
    metrics_df.to_csv(os.path.join(SAVE_PATH,
                                   TARGET+f"test_metrics_{SEQ}_report.csv")
                      )

    df = pd.DataFrame([ypred, ytrue],
                      columns=range(10),
                      index=[RUN_NAME+"_pred", RUN_NAME+"_real"])
    df.to_csv(os.path.join(SAVE_PATH, TARGET+f"_test_{SEQ}_report.csv"))


if __name__ == "__main__":
    # Argparse
    parser = argparse.ArgumentParser(description='WKIT')

    # data loader
    parser.add_argument('--root_path', type=str, default="dataset/Water_Week",
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default="D001청주정",
                        help='data file')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')
    parser.add_argument('--target', type=str, default='blue-green_algae',
                        help='target feature in S or MS task')
    parser.add_argument('--is_scale', type=int, default=1,
                        help='Scaling to standardization, default True')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=120,
                        help='input sequence length')
    parser.add_argument('--label_len', type=int, default=1,
                        help='start token length')
    parser.add_argument('--pred_len', type=int, default=1,
                        help='prediction sequence length')
    # optimization
    parser.add_argument('--num_workers', type=int, default=4,
                        help='data loader num workers')
    parser.add_argument('--train_epochs', type=int, default=2,
                        help='train epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3,
                        help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='mse',
                        help='loss function')
    # Wandb Option
    parser.add_argument('--use_wandb', type=int, default=0,
                        help='(Option) logging on wandb')
    args = parser.parse_args()
    set_seed(42)
    main(args)
