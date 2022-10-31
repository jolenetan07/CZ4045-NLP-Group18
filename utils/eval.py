import torch
from utils.logging import AverageMeter, ProgressMeter

import time
from utils.logging import accuracy


def val(model, device, val_loader, criterion, args, writer, epoch=0):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1], prefix="Test: "
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            data_samples, target = data[0].to(device), data[1].to(device)

            # compute output
            output = model(data_samples)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target)[0]
            losses.update(loss.item(), data_samples.size(0))
            top1.update(acc1[0], data_samples.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                if epoch == "test":
                    progress.write_to_tensorboard(
                        writer, "test",i
                    )
                else:
                    progress.write_to_tensorboard(
                        writer, "val", epoch * len(val_loader) + i
                    )

        progress.display(i)  # print final results

    return top1.avg
