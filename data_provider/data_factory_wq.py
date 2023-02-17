from torch.utils.data import DataLoader

from data_provider.data_loader_wq import Water_Dataset

def data_provider(args, flag = "train"):
    
    if flag == 'test':
        shuffle_flag = False
        batch_size = args.batch_size
    elif flag == 'pred':
        shuffle_flag = False
        batch_size = 1
    else:
        shuffle_flag = True
        batch_size = args.batch_size

    data_set = Water_Dataset(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=args.seq_len, 
        target=args.target,
        scale=args.is_scale
    )
    
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=False)
    return data_set, data_loader
