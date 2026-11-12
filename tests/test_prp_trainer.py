"""Exercise actual Lightning epoch/checkpoint mechanics with a small CPU model."""
import pathlib
import tempfile
import unittest

import pytorch_lightning as pl
import torch

import nnue_dataset as data
from train import make_data_loaders, parse_args, PrpLoaderState
from test_prp_loader import write_records


class ToyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))
        self.observed = {}

    def training_step(self, batch, batch_idx):
        scores = batch[7].reshape(-1)
        self.observed.setdefault(self.current_epoch, []).extend(int(x) for x in scores.tolist())
        return (self.weight - scores.mean() / 257).square()

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


class FeatureSet:
    name = 'HalfKAv2'


class PrpTrainerTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.path = str(self.root / 'numbered.bin')
        write_records(self.path, 257)

    def tearDown(self):
        self.temp.cleanup()

    def loaders(self, mode='prp', epoch_size=None, skipping=0, workers=4):
        return make_data_loaders(self.path, self.path, FeatureSet(), workers, 64, False,
                                 skipping, 'cpu', epoch_size, 0, mode, 42)

    def trainer(self, loader, epochs, folder):
        checkpoint = pl.callbacks.ModelCheckpoint(dirpath=str(self.root / folder), save_last=True)
        trainer = pl.Trainer(max_epochs=epochs, accelerator='cpu', devices=1,
                             logger=False, enable_progress_bar=False, enable_model_summary=False,
                             limit_val_batches=0, num_sanity_val_steps=0,
                             callbacks=[checkpoint, PrpLoaderState(loader.dataset)])
        return trainer, checkpoint

    def expected(self, epoch, skipping=0):
        provider = data.PrpSparseBatchProvider('HalfKAv2', self.path, 64, num_workers=1,
                                              seed=42, epoch=epoch, random_fen_skipping=skipping)
        try:
            return [int(x) for batch in provider for x in batch[7].reshape(-1).tolist()]
        finally:
            provider.close()

    def test_full_epochs_and_checkpoint_resumption(self):
        train, val = self.loaders()
        self.assertIsInstance(val.dataset, data.FixedNumBatchesDataset)
        trainer, checkpoint = self.trainer(train, 2, 'first')
        model = ToyModel()
        trainer.fit(model, train)
        self.assertEqual(trainer.global_step, 10)
        for epoch in (0, 1):
            self.assertEqual(model.observed[epoch], self.expected(epoch))
            self.assertEqual(sorted(model.observed[epoch]), list(range(257)))
        state = torch.load(checkpoint.last_model_path, map_location='cpu', weights_only=False)
        self.assertEqual(state['callbacks']['PrpLoaderState']['next_epoch'], 2)
        self.assertTrue(state['callbacks']['PrpLoaderState']['epoch_complete'])

        resumed_train, _ = self.loaders()
        resumed, _ = self.trainer(resumed_train, 3, 'resumed')
        resumed_model = ToyModel()
        resumed.fit(resumed_model, resumed_train, ckpt_path=checkpoint.last_model_path)
        self.assertEqual(resumed.global_step, 15)
        self.assertEqual(resumed_model.observed, {2: self.expected(2)})

    def test_filtered_epochs_stop_at_file_boundary(self):
        train, _ = self.loaders(skipping=3)
        trainer, _ = self.trainer(train, 2, 'filtered')
        model = ToyModel()
        trainer.fit(model, train)
        for epoch in (0, 1):
            self.assertEqual(model.observed[epoch], self.expected(epoch, skipping=3))
            self.assertTrue(0 < len(model.observed[epoch]) < 257)

    def test_epoch_size_and_sequential_compatibility(self):
        # Legacy multi-worker batches are delivered in completion order, so
        # ordered sequential parity must use one worker. PRP ordering is tested
        # independently with multiple workers above and in test_prp_loader.py.
        sequential, _ = self.loaders('sequential', epoch_size=257, workers=1)
        self.assertIsInstance(sequential.dataset, data.FixedNumBatchesDataset)
        batches = list(sequential)
        self.assertEqual([len(b[7]) for b in batches], [64] * 5)
        self.assertEqual([int(x) for b in batches for x in b[7].reshape(-1).tolist()],
                         list(range(257)) + list(range(63)))
        prp, _ = self.loaders(epoch_size=257)
        self.assertEqual(sum(len(b[7]) for b in prp), 257)
        with self.assertRaisesRegex(ValueError, '--epoch-size'):
            self.loaders(epoch_size=256)

    def test_prp_cli_defaults_keep_every_record(self):
        args = parse_args([self.path, self.path, '--data-loader', 'prp'])
        self.assertTrue(args.no_smart_fen_skipping)
        self.assertEqual(args.random_fen_skipping, 0)
        train, _ = make_data_loaders(args.train, args.val, FeatureSet(), 4, 64,
                                    not args.no_smart_fen_skipping, args.random_fen_skipping,
                                    'cpu', None, 0, args.data_loader, args.seed)
        scores = [int(x) for batch in train for x in batch[7].reshape(-1).tolist()]
        self.assertEqual(sorted(scores), list(range(257)))

    def test_sequential_cli_defaults_unchanged(self):
        for options in ([], ['--data-loader', 'sequential']):
            with self.subTest(options=options):
                args = parse_args([self.path, self.path, *options])
                self.assertFalse(args.no_smart_fen_skipping)
                self.assertEqual(args.random_fen_skipping, 3)

    def test_cli_skipping_overrides(self):
        for loader in ('sequential', 'prp'):
            for random_skip in (0, 7):
                with self.subTest(loader=loader, random_skip=random_skip):
                    args = parse_args([self.path, self.path, '--data-loader', loader,
                                       '--smart-fen-skipping', '--random-fen-skipping', str(random_skip)])
                    self.assertFalse(args.no_smart_fen_skipping)
                    self.assertEqual(args.random_fen_skipping, random_skip)
            for smart_options in (['--no-smart-fen-skipping'],
                                  ['--smart-fen-skipping', '--no-smart-fen-skipping'],
                                  ['--no-smart-fen-skipping', '--smart-fen-skipping']):
                with self.subTest(loader=loader, options=smart_options):
                    args = parse_args([self.path, self.path, '--data-loader', loader, *smart_options])
                    self.assertTrue(args.no_smart_fen_skipping)


if __name__ == '__main__':
    unittest.main()
