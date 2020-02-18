from torch.utils.data import ConcatDataset

# Local imports
from data_loader.dataloader_registry import DATALOADER


def create_dataset(cfg, split, split_name, length=0, transform=None):
    """
    Create the complete dataloader for each of the splits.

    cfg = configuration information for the program
    split = list of the dataloaders name in registry
    split_name = 'train' || 'test' || 'val'
    """
    data_list = []
    for dataloader in split:
        act_class = DATALOADER[dataloader]
        act_data = act_class(split_name, cfg.DATASET.FULL_SIZE,
                             cfg.DATASET.CONF,
                             length=length,
                             transform=transform)
        data_list.append(act_data)
    dataset = ConcatDataset(data_list)
    return dataset
