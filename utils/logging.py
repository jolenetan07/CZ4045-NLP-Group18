import sys
import yaml
import torch
import os
import shutil


def load_config(args):
    arg_override = []

    for arg in sys.argv:
        arg_name = get_arg(arg)
        if arg.startswith("-") and arg_name != "config":
            arg_override.append(arg_name)

    yaml_file = open(args.configs).read()

    # override args
    loaded_yaml = yaml.load(yaml_file, Loader=yaml.FullLoader)
    for v in arg_override:
        loaded_yaml[v] = getattr(args, v)

    print(f"Loading configuration from: {args.configs}")
    args.__dict__.update(loaded_yaml)


def get_arg(arg: str):
    # omit '--' in the argument
    i = 0
    while arg[i] == "-":
        i += 1
    arg = arg[i:]

    arg = arg.replace("-", "_")

    return arg.split("=")[0]


def accuracy(output, target, topk=(1,)):
    """
    Obtaine the topk accuracy, which means the if the correct
    label is in the topk largest logits, then the prediction considered
    is correct.


    """
    # for multi-class prediction
    if output.size(1) != 1:
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res
    else:
        # binary class with only logit output
        with torch.no_grad():
            batch_size = target.size(0)
            output = torch.round(output)
            correct = output.eq(target.view_as(output)).sum()
            return correct.mul_(100.0 / batch_size)


def save_checkpoint(
        state, is_best, result_dir, filename="checkpoint.pth.tar", save_dense=False
):
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(result_dir, filename),
            os.path.join(result_dir, "model_best.pth.tar"),
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def write_to_tensorboard(self, writer, prefix, global_step):
        for meter in self.meters:
            writer.add_scalar(f"{prefix}/{meter.name}", meter.val, global_step)


if __name__ == '__main__':
    from args import parse_args

    args = parse_args()
    load_config(args)
