import time
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os
import os.path as osp

from config import cfg
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.arguments import get_args
from utils import AverageMeter
from utils.utils import create_dataset
from models.model_registry import MODEL_EXAMPLE
from training_routine.routine_registry import EXAMPLE_ROUTINE

# Load arguments for the experiment
args = get_args()
args_dict = vars(args)
print('Argument list to program')
print('\n'.join(['--{0} {1}'.format(arg, args_dict[arg])
                for arg in args_dict]))
print('\n')

# Load arguments from the configuration file
cfg.merge_from_file(args.config)
cfg.freeze()
print(cfg)

is_cuda = not cfg.NO_CUDA and torch.cuda.is_available()

torch.manual_seed(cfg.TRAIN.SEED)
if is_cuda:
    torch.cuda.manual_seed(cfg.TRAIN.SEED)

# TODO: Transforms for the images
transform = transforms.Compose([])

print(transform)

# Define trainset
trainset = create_dataset(cfg, cfg.DATASET.SPLIT, 'train',
                          length=cfg.DATASET.LENGTH,
                          transform=transform)

# Load train data
train_loader = DataLoader(trainset,
                          batch_size=cfg.TRAIN.BATCH_SIZE,
                          shuffle=True,
                          pin_memory=False,
                          num_workers=cfg.DATASET.WORKERS)

# Start epoch for the model
start_epoch = cfg.TRAIN.START_EPOCH

if cfg.DATASET.VAL:
    # Define validation dataset
    valset = create_dataset(cfg, cfg.DATASET.VAL, 'val')

    # Load validation dataset
    # TODO: Check batch size
    val_loader = DataLoader(valset,
                            batch_size=1,
                            pin_memory=False,
                            num_workers=cfg.DATASET.WORKERS)

if cfg.DATASET.TEST:
    # Define evaluation dataset
    evalset = create_dataset(cfg, cfg.DATASET.TEST, 'test')

    # Load evaluation dataset
    # TODO: Check batch size
    eval_loader = DataLoader(evalset,
                             batch_size=1,
                             pin_memory=False,
                             num_workers=cfg.DATASET.WORKERS)

if not osp.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)

# Instance of the model
net = MODEL_EXAMPLE[cfg.MODEL]()

# Check if there are snapshots of previous states
if osp.exists(cfg.SNAPSHOT):
    net.load_state_dict(torch.load(cfg.SNAPSHOT))

# Check cuda
if is_cuda:
    net = net.cuda('cuda:{}'.format(cfg.CUDA_DEVICE))

# TODO: Instance the given routine for training
routines = EXAMPLE_ROUTINE[cfg.TRAIN.ROUTINE](net, cfg.CUDA_DEVICE, is_cuda)

# TODO: Create loss
criterion = nn.MSELoss()

# TODO: Define optimizer for the model
optimizer = optim.SGD(net.parameters(),
                      lr=cfg.TRAIN.LR,
                      momentum=cfg.TRAIN.MOMENTUM)

# TODO: Define cheduler
scheduler = ReduceLROnPlateau(optimizer, patience=cfg.TRAIN.PATIENCE)


def train(epoch):
    """Train of the net."""
    net.train()
    total_loss = AverageMeter()
    epoch_loss_stats = AverageMeter()
    start_time = time.time()

    bar = tqdm(enumerate(train_loader))
    for batch_idx, sample in bar:
        optimizer.zero_grad()
        # TODO: Call the train routine for the net
        # outputs, label = routines.train_routine(sample)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        total_loss.update(loss.data, n=outputs.size(0))
        epoch_loss_stats.update(loss.data, n=outputs.size(0))

        if batch_idx % cfg.TRAIN.BACKUP_ITERS == 0:
            filename = '{0}_snapshot.pth'.format(cfg.DATASET.SPLIT)
            filename = osp.join(cfg.OUTPUT_DIR, filename)
            state_dict = net.state_dict()
            torch.save(state_dict, filename)

            optim_filename = '{0}_optim.pth'.format(cfg.DATASET.SPLIT)
            optim_filename = osp.join(cfg.OUTPUT_DIR, optim_filename)
            state_dict = optimizer.state_dict()
            torch.save(state_dict, optim_filename)

        if batch_idx % cfg.TRAIN.LOG_INTERVAL == 0:
            elapsed_time = time.time() - start_time
            bar.set_description('[{:5d}] ({:5d}/{:5d}) | ms/batch {:.6f} |'
                                ' loss {:.6f} | lr {:.7f}'.format(
                                    epoch, batch_idx, len(train_loader),
                                    elapsed_time * 1000, total_loss.avg,
                                    optimizer.param_groups[0]['lr']))
            total_loss.reset()

        start_time = time.time()

    epoch_total_loss = epoch_loss_stats.avg
    return epoch_total_loss


def val(epoch, loader):
    """Validate the net."""
    net.eval()
    acc_meter = AverageMeter()
    epoch_loss_stats = AverageMeter()

    bar = tqdm(enumerate(loader))
    for batch_idx, sample in bar:
        # TODO: Call the val routine for the net
        # outputs, label = routines.validation_routine(sample)
        loss = criterion(outputs, label)
        epoch_loss_stats.update(loss.data, n=outputs.size(0))
        # TODO: Get the error of the validation
        out = mean_squared_error(label.cpu().flatten(),
                                 outputs.cpu().detach().numpy().flatten())
        acc_meter.update(out, n=outputs.size(0))

    return epoch_loss_stats.avg, acc_meter.avg


def test(loader):
    """Evaluate images with best weights."""
    net.eval()
    try:
        filename = osp.join(cfg.OUTPUT_DIR, 'best_val_acc_weights.pth')
        net.load_state_dict(torch.load(filename))
    except FileNotFoundError:
        net.load_state_dict(torch.load(cfg.OPTIM_SNAPSHOT))
    bar = tqdm(enumerate(loader))
    for _, sample in bar:
        start = time.time()
        # TODO: Call the test routine for the net
        # outputs, id = routines.test_routine(sample)
        total_time = time.time() - start
        # TODO: Save porosity value here


def main():
    """Run the program."""
    best_val_acc = None
    if cfg.TRAIN.EVAL_FIRST:
        print('Test begins...')
        test(eval_loader)
    try:
        print('Train begins...')
        for epoch in range(start_epoch, cfg.TRAIN.EPOCHS + 1):
            epoch_start_time = time.time()
            train_loss = train(epoch)
            val_loss, val_acc = val(epoch, val_loader)
            scheduler.step(val_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s '
                  '| epoch loss {:.6f} | val loss {:.6f} | val acc {:.6f} |'
                  '| val acc {:.6f} |'.format(
                      epoch, time.time() - epoch_start_time, train_loss,
                      val_loss, val_acc, val_acc))
            print('-' * 89)
            if best_val_acc is None or val_acc > best_val_acc:
                best_val_acc = val_acc
                filename = osp.join(cfg.OUTPUT_DIR,
                                    'best_val_acc_weights.pth')
                torch.save(net.state_dict(), filename)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
    finally:
        filename = osp.join(cfg.OUTPUT_DIR, 'best_weights.pth')
        if osp.exists(filename):
            net.load_state_dict(torch.load(filename))
        test(eval_loader)


if __name__ == '__main__':
    main()
