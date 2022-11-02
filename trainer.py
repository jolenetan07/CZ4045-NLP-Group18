import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils.logging import AverageMeter, ProgressMeter
from utils.logging import accuracy
from utils.eval import val


def train(
        model: nn.Module,
        device: torch.device,
        train_loader,
        criterion,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        args,
        writer: SummaryWriter):
    print(" ->->->->->->->->->-> ONE EPOCH TRAINING <-<-<-<-<-<-<-<-<-<-")

    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        input_data, target = data[0].to(device), data[1].to(device)

        # basic properties of training
        if i == 0:
            print(
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )

        input_data = input_data.type(torch.LongTensor)
        output = model(input_data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target)[0]
        losses.update(loss.item(), input_data.size(0))
        top1.update(acc1[0], input_data.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
            progress.write_to_tensorboard(
                writer, "train", epoch * len(train_loader) + i
            )




