import torch
from torch import nn


def prepare_model(model, args):
    """
    if checkpoint is given, load the checkpoint and adapt target domain
    """
    if args.checkpoint:
        # load checkpoint
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        print(f"=> loaded checkpoint '{args.checkpoint}'")

        if args.freeze_source:
            print("freeze weight and bias of the source model")
            freeze_layers(model, "weight")
            freeze_layers(model, "bias")

        # get the last module of the model, probably the final fully connected layer
        modules = list(model.named_modules())
        old_fc = modules[-1][1]
        attr_name = modules[-1][0]
        in_features = old_fc.in_features
        setattr(model, attr_name, nn.Linear(in_features, args.target_output_dim))
        # unfreeze the last layer
        getattr(model, attr_name).requires_grad = True
        print_gradient_update(model)


def print_gradient_update(model):
    print("->->-> Gradient update <-<-<-")
    for i, v in model.named_parameters():
        print(i, v.requires_grad)


def freeze_layers(model, var_name):
    assert var_name in ["weight", "bias"]
    for i, v in model.named_modules():
        if getattr(v, var_name) is not None:
            getattr(v, var_name).requires_grad = False
