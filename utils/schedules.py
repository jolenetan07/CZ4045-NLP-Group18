import sys
import torch


def get_optimizer(model, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    elif args.optimizer == "adam":
        return torch.optim.Adam(model.parameters(),
                                lr=args.lr,
                                weight_decay=args.wd)
    elif args.optimizer == "rmsprop":
        return torch.optim.RMSprop(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.wd,
        )
    else:
        print(f"{args.optimizer} is not supported.")
        sys.exit(0)
