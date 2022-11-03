import torch
from utils.logging import AverageMeter, ProgressMeter

import time
from utils.logging import accuracy, precision, recall, f1


def val(model, device, val_loader, criterion, args, writer, epoch=0):
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    precisions = AverageMeter("Precision", ":.4f")
    recalls = AverageMeter("Recall", ":.4f")
    f1_scores = AverageMeter("F1", ":.4f")

    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, precisions, recalls, f1_scores], prefix="Test: "
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
            acc1 = accuracy(output, target)
            losses.update(loss.item(), data_samples.size(0))
            top1.update(acc1, data_samples.size(0))
            precision_score = precision(output, target)
            recall_score = recall(output, target)
            f1_score = f1(output, target)

            precisions.update(precision_score, data_samples.size(0))
            recalls.update(recall_score, data_samples.size(0))
            f1_scores.update(f1_score, data_samples.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                if epoch == "test":
                    progress.write_to_tensorboard(
                        writer, "test", i
                    )
                else:
                    progress.write_to_tensorboard(
                        writer, "val", epoch * len(val_loader) + i
                    )

        if i % args.print_freq == 0:
            progress.display(i)
            progress.write_to_tensorboard(
                writer, "val", epoch * len(val_loader) + i
            )

    return top1.avg, precisions.val, recalls.val, f1_scores.val
