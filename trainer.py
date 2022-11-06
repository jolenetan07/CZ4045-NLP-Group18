import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from utils.logging import AverageMeter, ProgressMeter
from utils.logging import accuracy, precision, recall, f1


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
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    precisions = AverageMeter("Precision", ":.4f")
    recalls = AverageMeter("Recall", ":.4f")
    f1_scores = AverageMeter("F1", ":.4f")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, precisions, recalls, f1_scores],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        input_data, target = data[0], data[1]

        # basic properties of training
        if i == 0:
            print(
                f"Batch_size from args: {args.batch_size}",
                "lr: {:.5f}".format(optimizer.param_groups[0]["lr"]),
            )

        input_data = input_data.type(torch.LongTensor)
        input_data = input_data.to(device)
        target = target.to(device)
        output = model(input_data)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target)
        precision_score = precision(output, target)
        recall_score = recall(output, target)
        f1_score = f1(output, target)

        losses.update(loss.item(), input_data.size(0))
        top1.update(acc1, input_data.size(0))
        precisions.update(precision_score, input_data.size(0))
        recalls.update(recall_score, input_data.size(0))
        f1_scores.update(f1_score, input_data.size(0))

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
