import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    # primary
    parser.add_argument(
        "--configs", type=str, default="", help="configs file",
    )
    parser.add_argument(
        "--result-dir",
        default="./trained_models",
        type=str,
        help="directory to save results",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        help="Name of the experiment (creates dir with this name in --result-dir)",
    )

    # Model
    # Model implemented in models.py
    parser.add_argument("--arch", type=str, help="Model architecture")

    parser.add_argument(
        "--output-dim",
        type=int,
        default=2,
        help="Output dimension, default: 2",
    )

    parser.add_argument(
        "--embed-dim",
        type=int,
        default=300,
        help="Embedding output dimension, default: 300",
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="Hidden dimension, default: 256",
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate, default: 0.2",
    )

    # Data
    # TODO
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("",),
        help="Dataset for training and eval"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 128)",
    )

    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for testing (default: 128)",
    )

    parser.add_argument(
        "--data-dir", type=str, default="./dataset", help="path to datasets"
    )

    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of images used from training set",
    )

    parser.add_argument(
        "--epochs", type=int, default=100, metavar="N", help="number of epochs to train"
    )
    parser.add_argument(
        "--optimizer", type=str, default="sgd", choices=("sgd", "adam", "rmsprop")
    )

    parser.add_argument("--wd", default=1e-4, type=float, help="Weight decay")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")

    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="cosine",
        choices=("step", "cosine"),
        help="Learning rate schedule",
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="SGD momentum")

    parser.add_argument(
        "--warmup-epochs", type=int, default=0, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--warmup-lr", type=float, default=0.1, help="warmup learning rate"
    )

    # Evaluate
    parser.add_argument(
        "--evaluate", action="store_true", default=False, help="Evaluate model"
    )

    # Restart
    parser.add_argument(
        "--start-epoch",
        type=int,
        default=0,
        help="manual start epoch (useful in restarts)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="path to latest checkpoint (default:None)",
    )

    # Additional
    parser.add_argument(
        "--gpu", type=str, default="0", help="Comma separated list of GPU ids"
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--print-freq",
        type=int,
        default=100,
        help="Number of batches to wait before printing training logs",
    )

    parser.add_argument(
        "--schedule_length",
        type=int,
        default=0,
        help="Number of epochs to schedule the training epsilon.",
    )

    return parser.parse_args()
