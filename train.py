import torch
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np

import os
from pathlib import Path


import models
from args import parse_args
from trainer import train
from utils.logging import load_config
from utils.schedules import get_optimizer



def create_checkpoint_dir(args):
    """
    create folder for storing checkpoints and tensorboard logs
    """
    result_main_dir = os.path.join(Path(args.result_dir), args.exp_name, args.exp_mode)
    if os.path.exists(result_main_dir):
        n = len(next(os.walk(result_main_dir))[-2])  # prev experiments with same name
        result_sub_dir = os.path.join(
            result_main_dir,
            "{}--lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                n + 1,
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )
    else:
        os.makedirs(result_main_dir, exist_ok=True)
        result_sub_dir = os.path.join(
            result_main_dir,
            "1--lr-{}_epochs-{}_warmuplr-{}_warmupepochs-{}".format(
                args.lr,
                args.epochs,
                args.warmup_lr,
                args.warmup_epochs,
            ),
        )

    os.mkdir(result_sub_dir)
    os.mkdir(os.path.join(result_sub_dir, "checkpoint"))

    return result_main_dir, result_sub_dir


def set_seed(args):
    """
    set seeds for result reproducibility
    """
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)


def get_device(args):
    """
    get device for training
    """
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpu_list = [int(i) for i in args.gpu.strip().split(",")]
    return torch.device(f"cuda:{gpu_list[0]}" if use_cuda else "cpu")


def get_model(args):
    """
    get model with specified architecture in args --arch
    the architecture of model should be implemented in models/__init__.py
    """
    Model = models.__dict__[args.arch]
    if args.arch in ["DAN"]:
        model = Model(args.vocab_size,
                      args.embed_dim,
                      args.hidden_dim,
                      args.output_dim,
                      args.dropout)
    else:
        raise NotImplementedError

    return model


def main():
    # load configuration file specified by --config
    args = parse_args()
    load_config(args)

    # create checkpoint dir
    result_main_dir, result_sub_dir = create_checkpoint_dir(args)

    # set seed
    set_seed(args)

    # get device for training
    device = get_device(args)

    # tensorboard writer
    writer = SummaryWriter(os.path.join(result_sub_dir, "tensorboard"))

    # obtain model
    model = get_model(args)

    # TODO: get data loaders
    # the data loading logic should be implemented in data.py and returns train_loader, val_loader and test_loader
    train_loader = None
    val_loader = None
    test_loader = None

    # Autograd, config optimizer and loss function
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, model)


    # TODO: train model
    train(model, device, train_loader,  )


    # TODO: evaluate model

    pass


if __name__ == "__main__":
    pass
