import torch
from torch.utils.tensorboard import SummaryWriter

import random
import numpy as np

import os
from pathlib import Path

import models
from args import parse_args
from trainer import train
from utils.logging import load_config, save_checkpoint
from utils.schedules import get_optimizer, get_lr_policy
from utils.eval import val
from utils.models import prepare_model

import data


def create_checkpoint_dir(args):
    """
    create folder for storing checkpoints and tensorboard logs
    """
    result_main_dir = os.path.join(Path(args.result_dir), args.exp_name)
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

    # if not using cuda, use cpu or metal-gpu
    # this part is to detect if there are any metal device
    # available for acceleration (for macos)
    # you can delete it if causing any issue
    if not use_cuda and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available():
            print("Using MPS")
            return torch.device("mps")

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
    # if checkpoint is given for transfer learning
    # load the weight and freeze it, and replace the
    # old fully connected layer with a new one
    prepare_model(model, args)
    model.to(device)

    # the data loading logic should be implemented in ./data and returns train_loader, val_loader and test_loader
    get_data_loaders = data.__dict__[args.dataset]
    train_loader, val_loader, test_loader = get_data_loaders(args.batch_size)

    # Config Loss function and optimizers
    criterion = torch.nn.CrossEntropyLoss()  # Todo: using BCELoss for model with one output
    # optimizer is by args.optimizer (--optimizer), support sgd, adam, rmsprop
    optimizer = get_optimizer(model, args)
    # adjust the learning rate
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)

    # Model training iterations start here
    best_val_acc = 0
    for epoch in range(args.epochs + args.warmup_epochs):
        lr_policy(epoch)  # adjust learning rate

        train(model, device, train_loader, criterion, optimizer, epoch, args, writer)

        # do model validation after each training epoch
        val_acc = val(model, device, val_loader, criterion, args, writer, epoch=epoch)

        is_best = val_acc > best_val_acc

        if (is_best):
            best_val_acc = val_acc

        training_state = {
            "epoch": epoch + 1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_prec1": best_val_acc,
            "optimizer": optimizer.state_dict(),

        }

        # only save the checkpoint of current  epoch  and checkpoint having the best validation accuracy
        save_checkpoint(training_state, is_best, os.path.join(result_sub_dir, "checkpoint"))
        print(f"Epoch {epoch}, val_acc: {val_acc}, best_val_acc: {best_val_acc}")

    # Test model
    test_acc = val(model, device, test_loader, criterion, args, writer, epoch="test")
    print(f"Training finished, test_acc: {test_acc}")


if __name__ == "__main__":
    main()
