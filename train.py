import argparse
import os
import warnings

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
import torch
from torch import set_num_threads as t_set_num_threads
from torch.utils.data import DataLoader

import features
import model as M
import nnue_dataset


# Optional: silence a common, harmless CuPy warning when no full CUDA Toolkit is installed.
warnings.filterwarnings(
    "ignore",
    message=r"CUDA path could not be detected\..*",
    category=UserWarning,
    module=r"cupy\._environment"
)


def make_data_loaders(train_filename, val_filename, feature_set, num_workers, batch_size, filtered, random_fen_skipping, main_device, epoch_size, val_size):
    features_name = feature_set.name
    train_infinite = nnue_dataset.SparseBatchDataset(features_name, train_filename, batch_size, num_workers=num_workers, filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device)
    val_infinite = nnue_dataset.SparseBatchDataset(features_name, val_filename, batch_size, filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device)
    train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
    val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
    return train, val


def add_trainer_args(parser):
    parser.add_argument("--gpus", type=int, default=None, dest="gpus", help="Number of GPUs to use. Backwards-compatible alias for Lightning 2.x devices/accelerator.")
    parser.add_argument("--devices", type=int, default=None, dest="devices", help="Number of devices to use (Lightning 2.x).")
    parser.add_argument("--accelerator", type=str, default=None, dest="accelerator", help="Accelerator type (cpu/gpu/auto).")
    parser.add_argument("--max_epochs", type=int, default=None, dest="max_epochs", help="Stop training once this number of epochs is reached.")
    parser.add_argument("--min_epochs", type=int, default=None, dest="min_epochs", help="Force training for at least this number of epochs.")
    parser.add_argument("--default_root_dir", type=str, default=None, dest="default_root_dir", help="Default path for logs and checkpoints.")
    parser.add_argument("--precision", type=str, default=None, dest="precision", help="Precision (e.g. 32, 16-mixed, bf16-mixed).")
    parser.add_argument("--log_every_n_steps", type=int, default=None, dest="log_every_n_steps", help="How often to log within steps.")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=None, dest="check_val_every_n_epoch", help="How often to run validation.")
    parser.add_argument("--val_check_interval", type=float, default=None, dest="val_check_interval", help="How often to check validation within an epoch.")
    parser.add_argument("--limit_train_batches", type=float, default=None, dest="limit_train_batches", help="Fraction/number of training batches to use.")
    parser.add_argument("--limit_val_batches", type=float, default=None, dest="limit_val_batches", help="Fraction/number of validation batches to use.")
    parser.add_argument("--accumulate_grad_batches", type=int, default=None, dest="accumulate_grad_batches", help="Accumulate gradients over k batches.")
    parser.add_argument("--gradient_clip_val", type=float, default=None, dest="gradient_clip_val", help="Clip gradients at this value.")
    return parser


def build_trainer_kwargs(args):
    trainer_kwargs = {}
    if args.gpus is not None:
        if args.gpus > 0:
            trainer_kwargs["accelerator"] = args.accelerator or "gpu"
            trainer_kwargs["devices"] = args.gpus
        else:
            trainer_kwargs["accelerator"] = args.accelerator or "cpu"
            trainer_kwargs["devices"] = 1
    if args.devices is not None:
        trainer_kwargs["devices"] = args.devices
    if args.accelerator is not None:
        trainer_kwargs["accelerator"] = args.accelerator
    for key in (
        "max_epochs",
        "min_epochs",
        "default_root_dir",
        "precision",
        "log_every_n_steps",
        "check_val_every_n_epoch",
        "val_check_interval",
        "limit_train_batches",
        "limit_val_batches",
        "accumulate_grad_batches",
        "gradient_clip_val",
    ):
        value = getattr(args, key)
        if value is not None:
            trainer_kwargs[key] = value
    return trainer_kwargs


def main():
    parser = argparse.ArgumentParser(description="Trains the network.")
    parser.add_argument("train", help="Training data (.bin)")
    parser.add_argument("val", help="Validation data (.bin)")
    parser = add_trainer_args(parser)
    parser.add_argument("--lambda", default=1.0, type=float, dest="lambda_", help="lambda=1.0 = train on evals, lambda=0.0 = train on results.")
    parser.add_argument("--num-workers", default=1, type=int, dest="num_workers", help="Workers for data loading. Sparse loaders often need 0.")
    parser.add_argument("--batch-size", default=-1, type=int, dest="batch_size", help="Positions per batch/iteration.")
    parser.add_argument("--threads", default=-1, type=int, dest="threads", help="Torch CPU threads.")
    parser.add_argument("--seed", default=42, type=int, dest="seed", help="Seed.")
    parser.add_argument("--smart-fen-skipping", action="store_true", dest="smart_fen_skipping_deprecated", help="Deprecated, ignored.")
    parser.add_argument("--no-smart-fen-skipping", action="store_true", dest="no_smart_fen_skipping", help="Disable smart fen skipping.")
    parser.add_argument("--random-fen-skipping", default=3, type=int, dest="random_fen_skipping", help="Randomly skip fens on average N before using one.")
    parser.add_argument("--resume-from-model", dest="resume_from_model", help="Initialize from a .pt model")
    parser.add_argument("--epoch-size", type=int, default=20000000, dest="epoch_size", help="Positions per epoch.")
    parser.add_argument("--validation-size", type=int, default=1000000, dest="validation_size", help="Positions per validation step.")
    features.add_argparse_args(parser)
    args = parser.parse_args()

    if not os.path.exists(args.train):
        raise FileNotFoundError(args.train)
    if not os.path.exists(args.val):
        raise FileNotFoundError(args.val)

    feature_set = features.get_feature_set_from_name(args.features)

    if args.resume_from_model is None:
        nnue = M.NNUE(feature_set=feature_set, lambda_=args.lambda_)
    else:
        nnue = torch.load(args.resume_from_model, weights_only=False, map_location="cpu")
        nnue.set_feature_set(feature_set)
        nnue.lambda_ = args.lambda_

    print("Feature set: {}".format(feature_set.name))
    print("Num real features: {}".format(feature_set.num_real_features))
    print("Num virtual features: {}".format(feature_set.num_virtual_features))
    print("Num features: {}".format(feature_set.num_features))
    print("Training with {} validating with {}".format(args.train, args.val))

    pl.seed_everything(args.seed, workers=True)
    print("Seed {}".format(args.seed))

    batch_size = args.batch_size
    if batch_size <= 0:
        batch_size = 16384
    print("Using batch size {}".format(batch_size))
    print("Smart fen skipping: {}".format(not args.no_smart_fen_skipping))
    print("Random fen skipping: {}".format(args.random_fen_skipping))

    if args.threads > 0:
        print("limiting torch to {} threads.".format(args.threads))
        t_set_num_threads(args.threads)

    logdir = args.default_root_dir if args.default_root_dir else "logs/"
    print("Using log dir {}".format(logdir), flush=True)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logdir)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True, every_n_epochs=1, save_top_k=-1)

    trainer_kwargs = build_trainer_kwargs(args)
    trainer = pl.Trainer(callbacks=[checkpoint_callback], logger=tb_logger, **trainer_kwargs)

    main_device = str(trainer.strategy.root_device)

    print("Using c++ data loader")
    train, val = make_data_loaders(args.train, args.val, feature_set, args.num_workers, batch_size, not args.no_smart_fen_skipping, args.random_fen_skipping, main_device, args.epoch_size, args.validation_size)

    trainer.fit(nnue, train_dataloaders=train, val_dataloaders=val)


if __name__ == "__main__":
    main()
