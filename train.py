import argparse
import model as M
import nnue_dataset
import pytorch_lightning as pl
import features
import score_targets
import os
import torch
from torch import set_num_threads as t_set_num_threads
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, Dataset

class PrpLoaderState(pl.Callback):
  """Save traversal keys at epoch boundaries alongside the trainer checkpoint."""
  def __init__(self, dataset):
    self.dataset = dataset

  def state_dict(self):
    return self.dataset.state_dict()

  def load_state_dict(self, state):
    self.dataset.load_state_dict(state)

  def setup(self, trainer, pl_module, stage):
    if trainer.world_size != 1:
      raise ValueError('PRP currently supports one trainer process; use --num-workers for native parallelism.')

  def on_load_checkpoint(self, trainer, pl_module, checkpoint):
    if self.state_key not in checkpoint.get('callbacks', {}):
      raise ValueError('PRP checkpoint has no loader state. Use a PRP epoch-boundary checkpoint, or initialize weights with --resume-from-model.')


def make_data_loaders(train_filename, val_filename, feature_set, num_workers, batch_size, filtered, random_fen_skipping, main_device, epoch_size, val_size, data_loader='sequential', seed=42, record_bytes=72):
  features_name = feature_set.name
  config = getattr(feature_set, "config", None)
  if data_loader == 'prp':
    dataset = nnue_dataset.PrpBatchDataset(features_name, train_filename, batch_size, num_workers=num_workers,
                                          filtered=filtered, random_fen_skipping=random_fen_skipping,
                                          device=main_device, seed=seed, record_bytes=record_bytes, config=config)
    if epoch_size is not None and epoch_size != dataset.record_count:
      raise ValueError('--epoch-size must match the dataset record count ({}) with --data-loader prp.'.format(dataset.record_count))
    train = DataLoader(dataset, batch_size=None, batch_sampler=None)
    print('PRP epoch: one traversal of {} source records.'.format(dataset.record_count))
  elif data_loader == 'sequential':
    epoch_size = 20000000 if epoch_size is None else epoch_size
    train_infinite = nnue_dataset.SparseBatchDataset(features_name, train_filename, batch_size, num_workers=num_workers,
                                                   filtered=filtered, random_fen_skipping=random_fen_skipping, device=main_device, record_bytes=record_bytes, config=config)
    train = DataLoader(nnue_dataset.FixedNumBatchesDataset(train_infinite, (epoch_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  else:
    raise ValueError('Unknown data loader: ' + data_loader)
  val_infinite = nnue_dataset.SparseBatchDataset(features_name, val_filename, batch_size, filtered=filtered,
                                                   random_fen_skipping=random_fen_skipping, device=main_device, record_bytes=record_bytes, config=config)
  # num_workers has to be 0 for sparse, and 1 for dense
  # it currently cannot work in parallel mode but it shouldn't need to
  val = DataLoader(nnue_dataset.FixedNumBatchesDataset(val_infinite, (val_size + batch_size - 1) // batch_size), batch_size=None, batch_sampler=None)
  return train, val

def parse_args(argv=None):
  parser = argparse.ArgumentParser(description="Trains the network.")
  parser.add_argument("train", help="Training data (.bin)")
  parser.add_argument("val", help="Validation data (.bin)")
  parser = pl.Trainer.add_argparse_args(parser)
  parser.add_argument("--lambda", default=1.0, type=float, dest='lambda_', help="lambda=1.0 = train on evaluations, lambda=0.0 = train on game results, interpolates between (default=1.0).")
  parser.add_argument("--num-workers", default=1, type=int, dest='num_workers', help="Number of worker threads to use for data loading. Currently only works well for bin.")
  parser.add_argument("--batch-size", default=-1, type=int, dest='batch_size', help="Number of positions per batch / per iteration. Default on GPU = 8192 on CPU = 128.")
  parser.add_argument("--threads", default=-1, type=int, dest='threads', help="Number of torch threads to use. Default automatic (cores) .")
  parser.add_argument("--seed", default=42, type=int, dest='seed', help="torch seed to use.")
  parser.add_argument("--data-loader", choices=('sequential', 'prp'), default='sequential', help="Training record order. Default: sequential. PRP traverses the whole dataset once per epoch.")
  def record_bytes(text):
    try:
      return nnue_dataset.validate_record_bytes(int(text))
    except ValueError as exc:
      raise argparse.ArgumentTypeError(str(exc)) from exc
  parser.add_argument('--record-bytes', type=record_bytes, default=72,
                      help='Bytes per record in both input files, integer 7..144 (default: 72).')
  parser.add_argument("--smart-fen-skipping", action='store_true', dest='smart_fen_skipping_deprecated', help="Enable smart fen skipping (default: enabled for sequential, disabled for PRP). --no-smart-fen-skipping takes precedence if both are supplied.")
  parser.add_argument("--no-smart-fen-skipping", action='store_true', default=None, dest='no_smart_fen_skipping', help="Disable smart fen skipping (default for PRP). Sequential enables it unless disabled explicitly.")
  parser.add_argument("--random-fen-skipping", default=None, type=int, dest='random_fen_skipping', help="Skip fens randomly on average this many before using one (default: 3 for sequential, 0 for PRP).")
  parser.add_argument("--resume-from-model", dest='resume_from_model', help="Initializes training using the weights from the given .pt model")
  parser.add_argument("--epoch-size", type=int, default=None, dest='epoch_size', help="Positions per sequential epoch (default: 20000000). If specified for PRP, must equal the dataset record count.")
  parser.add_argument("--validation-size", type=int, default=1000000, dest='validation_size', help="Number of positions per validation step.")
  score_targets.add_argparse_args(parser)
  features.add_argparse_args(parser)
  args = parser.parse_args(argv)
  if args.no_smart_fen_skipping is None:
    args.no_smart_fen_skipping = args.data_loader == 'prp' and not args.smart_fen_skipping_deprecated
  if args.random_fen_skipping is None:
    args.random_fen_skipping = 0 if args.data_loader == 'prp' else 3
  return args

def main():
  args = parse_args()

  if not os.path.exists(args.train):
    raise Exception('{0} does not exist'.format(args.train))
  if not os.path.exists(args.val):
    raise Exception('{0} does not exist'.format(args.val))

  feature_set = features.get_feature_set_from_name(args.features, features.config_from_args(args))

  if args.resume_from_model is None:
    nnue = M.NNUE(feature_set=feature_set, lambda_=args.lambda_)
  else:
    # Load with weights_only=False to avoid safe_globals complexity
    # This is safe since we trust the checkpoint source
    nnue = torch.load(args.resume_from_model, weights_only=False)
    nnue.set_feature_set(feature_set)
    nnue.lambda_ = args.lambda_

  nnue.set_score_target_overrides(args.score_target, args.score_target_scale)

  print("Feature set: {}".format(feature_set.name))
  print("Num real features: {}".format(feature_set.num_real_features))
  print("Num virtual features: {}".format(feature_set.num_virtual_features))
  print("Num features: {}".format(feature_set.num_features))

  print("Training with {} validating with {}".format(args.train, args.val))

  pl.seed_everything(args.seed)
  print("Seed {}".format(args.seed))

  batch_size = args.batch_size
  if batch_size <= 0:
    batch_size = 16384
  print('Using batch size {}'.format(batch_size))

  print('Smart fen skipping: {}'.format(not args.no_smart_fen_skipping))
  print('Random fen skipping: {}'.format(args.random_fen_skipping))

  if args.threads > 0:
    print('limiting torch to {} threads.'.format(args.threads))
    t_set_num_threads(args.threads)

  logdir = args.default_root_dir if args.default_root_dir else 'logs/'
  print('Using log dir {}'.format(logdir), flush=True)

  tb_logger = pl_loggers.TensorBoardLogger(logdir)
  checkpoint_callback = pl.callbacks.ModelCheckpoint(save_last=True, every_n_epochs=1, save_top_k=-1)
  trainer = pl.Trainer.from_argparse_args(args, callbacks=[checkpoint_callback], logger=tb_logger)

  main_device = trainer.strategy.root_device if trainer.strategy.root_device.index is None else 'cuda:' + str(trainer.strategy.root_device.index)

  print('Using c++ data loader ({})'.format(args.data_loader))
  train, val = make_data_loaders(args.train, args.val, feature_set, args.num_workers, batch_size, not args.no_smart_fen_skipping, args.random_fen_skipping, main_device, args.epoch_size, args.validation_size, args.data_loader, args.seed, args.record_bytes)
  if args.data_loader == 'prp':
    trainer.callbacks.append(PrpLoaderState(train.dataset))

  trainer.fit(nnue, train, val)

if __name__ == '__main__':
  main()
