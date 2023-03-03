import time

import torch
import wandb
import numpy as np
from tqdm import tqdm


def fit(epochs,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        path,
        scheduler=None,
        earlystop=20,
        device="cuda",
        use_wandb=True):
    # Initial
    torch.cuda.empty_cache()
    train_losses = []
    val_losses = []
    min_loss = np.inf
    i = 0
    warmup = 30
    model.to(device)
    if use_wandb:
        wandb.watch(model, log="all")
    fit_time = time.time()

    # Run epochs
    for e in range(epochs):
        since = time.time()
        running_loss = 0
        # training loop
        model.train()
        for _, data in enumerate(tqdm(train_loader)):
            # training phase
            inputs, labels = data
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            # reset gradient
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            outputs = outputs[:, :, -1].flatten()
            loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0
        # validation loop
        with torch.no_grad():
            for _, data in enumerate(tqdm(val_loader)):
                inputs, labels = data
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                # forward
                outputs = model(inputs)
                outputs = outputs[:, :, -1].flatten()
                # evaluation metrics
                # loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        # calculate mean for each batch
        t_loss = running_loss / len(train_loader)
        v_loss = val_loss / len(val_loader)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        # Warmup and Earlystopping
        if e + 1 > warmup:
            scheduler.step(val_loss)
            if min_loss - v_loss > 0.0001:
                print(f'Update {min_loss:.3f} >> {v_loss:.3f}. Save to {path}')
                min_loss = v_loss
                torch.save(model.state_dict(), path)
                i = 0
            else:
                i += 1
                print(f'Loss Not Decrease for {i} time')
                if i == earlystop:
                    print(f'Loss not decrease for {i} times, Stop Training')
                    break
        else:
            print(f"Warmup until {warmup} epochs")

        print(
            "Epoch:{}/{}..".format(e + 1, epochs),
            "Train Loss: {:.3f}..".format(t_loss),
            "Val Loss: {:.3f}..".format(v_loss),
            "LR:{}".format(scheduler.optimizer.param_groups[0]['lr']),
            "Time: {:.2f}m".format((time.time() - since) / 60)
            )
        if use_wandb:
            metrics_log = {
                "train_epoch": e + 1,
                "Train Loss": running_loss / len(train_loader),
                "Val Loss": val_loss / len(val_loader),
                "Learning Rate": scheduler.optimizer.param_groups[0]['lr'],
                "Time": (time.time() - since) / 60
                }
            wandb.log(metrics_log)
    history = {'train_loss': train_losses, 'val_loss': val_losses}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history


def test(model,
         test_loader,
         device="cuda"):
    model.eval()
    model.to(device)
    yPred = []
    yTrue = []
    with torch.no_grad():
        for _, data in enumerate(tqdm(test_loader)):
            inputs, labels = data
            inputs = inputs.to(device).float()
            outputs = model(inputs)
            outputs = outputs[:, :, -1].flatten()
            yPred.extend(outputs.detach().cpu().numpy())
            yTrue.extend(labels.flatten().detach().cpu().numpy())
    return np.array(yPred), np.array(yTrue)
