import torch
import time
import numpy as np
from tqdm import tqdm
import wandb

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, path, scheduler = None, earlystop = 20, device="cuda", use_wandb = True):
    ## Initial
    torch.cuda.empty_cache()    
    train_losses = []
    val_losses = []
    min_loss = np.inf
    not_improve = 0
    warmup = 0
    model.to(device)
    if use_wandb:
        wandb.watch(model, log="all")
    fit_time = time.time()

    ## Run epochs
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
            optimizer.zero_grad()  # reset gradient
            
            # forward
            
            outputs = model(inputs)
            outputs = outputs[:,:,-1].flatten()
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
                outputs = outputs[:,:,-1].flatten()
                # evaluation metrics
                # loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()                

        # calculate mean for each batch
        train_losses.append(running_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))


        ## Warmup and Earlystopping
        if e + 1 > warmup:
            if min_loss - (val_loss / len(val_loader)) > 0.0001 :
                print('Loss Decreasing {:.3f} >> {:.3f} . Save model to {} '.format(min_loss, (val_loss / len(val_loader)), path))
                min_loss = (val_loss / len(val_loader))            
#                torch.save(model.state_dict(), path)
                not_improve = 0
                
            else:
                not_improve += 1
                print(f'Loss Not Decrease for {not_improve} time')
                if not_improve == earlystop:
                    print(f'Loss not decrease for {not_improve} times, Stop Training')
                    break
            if scheduler != None:   
                scheduler.step(val_loss)            
        else:
            print(f"Warmup until {warmup} epochs")

        print("Epoch:{}/{}..".format(e + 1, epochs),
                "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                "Val Loss: {:.3f}..".format(val_loss / len(val_loader)),
                "Learning Rate:{}..".format(scheduler.optimizer.param_groups[0]['lr']),
                "Time: {:.2f}m".format((time.time() - since) / 60))
        if use_wandb:
            metrics_log = {"train_epoch": e + 1,
                        "Train Loss": running_loss / len(train_loader),
                        "Val Loss": val_loss / len(val_loader),
                        "Learning Rate":scheduler.optimizer.param_groups[0]['lr'],
                        "Time": (time.time() - since) / 60
                        }
            wandb.log(metrics_log)


    history = {'train_loss': train_losses, 'val_loss': val_losses}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history

def test(model, test_loader, device="cuda"):
    model.eval()
    model.to(device)
    Pred = []
    with torch.no_grad():
        for _, data in enumerate(tqdm(test_loader)):
            inputs, _ = data                    
            inputs = inputs.to(device).float()            
            outputs = model(inputs)
            outputs = outputs[:,:,-1].flatten()
            Pred.extend(outputs.detach().cpu().numpy())                    
    return np.array(Pred)
