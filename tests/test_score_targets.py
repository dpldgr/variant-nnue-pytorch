"""Target curves, their packed-score adapter, and training/checkpoint integration."""
import argparse
import contextlib
import io
import pathlib
import tempfile
import unittest
from unittest import mock

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

import model as models
import score_targets as targets


DTM_NAMES = targets.NAMES[1:]


class TinyFeatures:
    name = 'test-features'
    hash = 0x12345678
    num_features = num_real_features = 16
    num_psqt_buckets = num_ls_buckets = 8

    def get_virtual_feature_ranges(self):
        return []

    def get_initial_psqt_features(self):
        return [0] * self.num_features

    def get_virtual_to_real_features_gather_indices(self):
        return [[i] for i in range(self.num_features)]


def batch():
    us = torch.tensor([[1], [0], [1], [0], [1]], dtype=torch.float32)
    wi = torch.tensor([[0, 1, -1], [2, 3, -1], [4, 5, -1], [6, 7, -1], [8, 9, -1]], dtype=torch.int32)
    bi = wi.flip(0)
    values = torch.tensor([[1, 1, 0]] * 5, dtype=torch.float32)
    score = torch.tensor([[32000], [31999], [31936], [0], [-31998]], dtype=torch.float32)
    outcome = (score.sign() + 1) / 2
    indices = torch.zeros(5, dtype=torch.int64)
    return us, 1 - us, wi, values, bi, values, outcome, score, indices, indices


class ScoreTargetTest(unittest.TestCase):
    def test_sigmoid_is_unchanged(self):
        scores = torch.arange(-32768, 32768, dtype=torch.float32)
        scores = torch.cat((scores, torch.tensor([-0.5, 0.25, 1.125])))
        mapping = targets.ScoreTarget()
        self.assertIsNone(mapping.table)
        torch.testing.assert_close(mapping(scores), (scores / 410).sigmoid(), rtol=0, atol=0)

    def test_all_packed_scores_and_lookup_properties(self):
        scores = torch.arange(-32768, 32768, dtype=torch.float32).reshape(-1, 1)
        for name in DTM_NAMES:
            with self.subTest(name=name):
                mapping = targets.ScoreTarget(name)
                actual = mapping(scores)
                self.assertEqual(actual.shape, scores.shape)
                self.assertEqual(mapping.table.numel() * mapping.table.element_size(), 262144)
                self.assertFalse(mapping.table.requires_grad)
                self.assertEqual(mapping.state_dict(), {})
                self.assertTrue(torch.isfinite(actual).all())
                self.assertTrue((actual[1:] >= actual[:-1]).all())
                self.assertEqual(mapping(torch.tensor([0.0])).item(), 0.5)
                self.assertEqual(mapping(torch.tensor([32000.0, 32767.0])).tolist(), [1, 1])
                self.assertEqual(mapping(torch.tensor([-32000.0, -32768.0])).tolist(), [0, 0])
                positive = mapping(torch.arange(1, 32001, dtype=torch.float32))
                negative = mapping(-torch.arange(1, 32001, dtype=torch.float32))
                torch.testing.assert_close(negative, 1 - positive, rtol=0, atol=0)
                self.assertTrue((positive >= 0.6).all())
                self.assertTrue((negative <= 0.4).all())
                with self.assertRaises((IndexError, RuntimeError)):
                    mapping(torch.tensor([-32769.0]))

    def test_matches_report_points(self):
        # Independently tabulated values at d/T=2 from the reviewed report.
        expected = {'tanh': 0.6953623376176941, 'mu-law': 0.8657909356455138,
                    'a-law': 0.8071045695533157, 'linear-recip': 0.7,
                    'linear-exp': 0.6735758882342885, 'linear-invlog': 0.7181232218299283}
        for name, value in expected.items():
            self.assertAlmostEqual(targets.winning_probability(128, name).item(), value, places=12)
            self.assertAlmostEqual(targets.ScoreTarget(name)(torch.tensor([31872.0])).item(), value, places=6)

    def test_initial_slopes_join_and_unbounded_tail(self):
        for name in DTM_NAMES:
            f = lambda x: targets.winning_probability(torch.tensor(x, dtype=torch.float64), name, 1).item()
            self.assertAlmostEqual(f(0), 1)
            self.assertAlmostEqual(f(float('inf')), 0.6)
            self.assertAlmostEqual((f(0) - f(1e-5)) / 1e-5, 0.2, places=4)
            if name.startswith('linear-'):
                self.assertAlmostEqual(f(0.25), 0.95)
                self.assertAlmostEqual(f(1), 0.8)
                self.assertAlmostEqual((f(1) - f(1 - 1e-5)) / 1e-5,
                                       (f(1 + 1e-5) - f(1)) / 1e-5, places=5)
        self.assertGreater(targets.winning_probability(1000000, 'linear-invlog').item(), 0.6)

    def test_scale_changes_the_distance_not_the_formula(self):
        for name in DTM_NAMES:
            torch.testing.assert_close(targets.winning_probability(20, name, 10),
                                       targets.winning_probability(200, name, 100))

    def test_cli_and_invalid_configuration(self):
        parser = argparse.ArgumentParser()
        targets.add_argparse_args(parser)
        self.assertIsNone(parser.parse_args([]).score_target)
        for name in targets.NAMES:
            args = parser.parse_args(['--score-target', name, '--score-target-scale', '32.5'])
            self.assertEqual((args.score_target, args.score_target_scale), (name, 32.5))
        for scale in ('0', '-1', 'nan', 'inf'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(['--score-target-scale', scale])
        with self.assertRaises(ValueError):
            targets.ScoreTarget('unknown')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_device_lookup(self):
        scores = torch.tensor([-32000, -31999, 0, 31999, 32000], dtype=torch.float32)
        for name in targets.NAMES:
            mapping = targets.ScoreTarget(name)
            expected = mapping(scores)
            actual = mapping.cuda()(scores.cuda()).cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)


class ModelTargetTest(unittest.TestCase):
    def test_checkpoint_uses_position_weighted_epoch_loss(self):
        from train import make_checkpoint_callback

        class CaptureLosses(pl.Callback):
            def __init__(self):
                self.batches = []

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                self.batches.append((outputs['loss'].detach().item(), batch[7].shape[0]))

        with tempfile.TemporaryDirectory() as folder:
            data = batch()
            loader = DataLoader([tuple(t[:4] for t in data), tuple(t[4:] for t in data)], batch_size=None)
            checkpoint = make_checkpoint_callback(dirpath=folder)
            captured = CaptureLosses()
            model = models.NNUE(TinyFeatures(), score_target='linear-recip')
            trainer = pl.Trainer(max_epochs=1, accelerator='cpu', devices=1, logger=False,
                                 enable_progress_bar=False, enable_model_summary=False,
                                 limit_val_batches=0, callbacks=[checkpoint, captured])
            trainer.fit(model, loader)
            expected = sum(loss * size for loss, size in captured.batches) / 5
            self.assertAlmostEqual(checkpoint.best_model_score.item(), expected, places=7)
            self.assertIn('train_loss', trainer.logged_metrics)
            self.assertTrue(pathlib.Path(checkpoint.last_model_path).is_file())

    def test_loss_gradients_and_lambda_blending(self):
        model = models.NNUE(TinyFeatures())
        b = batch()
        for name in targets.NAMES:
            model.set_score_target(name)
            for weight in (0.0, 0.37, 1.0):
                model.lambda_ = weight
                model.zero_grad()
                model._clip_weights()
                q = (model(*b[:6], *b[8:]) * 600 / 361).sigmoid()
                p = targets.ScoreTarget(name)(b[7])
                expected = weight * (p - q).square().mean() + (1 - weight) * (q - b[6]).square().mean()
                with mock.patch.object(model, 'log'):
                    actual = model.step_(b, 0, 'train_loss')
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.backward()
                self.assertTrue(torch.isfinite(actual))
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_weights_and_nnue_serialization_do_not_change(self):
        from serialize import NNUEWriter
        model = models.NNUE(TinyFeatures())
        original_keys = set(model.state_dict())
        with contextlib.redirect_stdout(io.StringIO()):
            original_bytes = bytes(NNUEWriter(model).buf)
            for name in DTM_NAMES:
                model.set_score_target(name)
                self.assertEqual(set(model.state_dict()), original_keys)
                self.assertEqual(bytes(NNUEWriter(model).buf), original_bytes)

    def test_legacy_models_and_checkpoint_configuration(self):
        model = models.NNUE(TinyFeatures())
        del model.score_target  # A pre-change .pt model has no target module.
        model.set_score_target_overrides()
        self.assertEqual(model.score_target.name, 'sigmoid')
        model.set_score_target('tanh', 32)
        model.set_score_target_overrides()
        model.on_load_checkpoint({})  # Legacy checkpoint has the original mapping.
        self.assertEqual(model.score_target.name, 'sigmoid')
        model.set_score_target_overrides('mu-law', 128)
        model.on_load_checkpoint({})
        self.assertEqual((model.score_target.name, model.score_target.scale), ('mu-law', 128))
        checkpoint = {'score_target': dict(version=1, name='a-law', scale=32)}
        model.set_score_target_overrides('tanh')
        model.on_load_checkpoint(checkpoint)
        self.assertEqual((model.score_target.name, model.score_target.scale), ('tanh', 32))
        model.set_score_target_overrides(scale=128)
        model.on_load_checkpoint(checkpoint)
        self.assertEqual((model.score_target.name, model.score_target.scale), ('a-law', 128))
        with self.assertRaisesRegex(ValueError, 'version'):
            model.on_load_checkpoint({'score_target': {'version': 999}})

    def test_lightning_checkpoint_restore_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as folder:
            def trainer(epochs, suffix):
                checkpoint = pl.callbacks.ModelCheckpoint(dirpath=str(pathlib.Path(folder) / suffix), save_last=True)
                fit = pl.Trainer(max_epochs=epochs, accelerator='cpu', devices=1,
                                 logger=False, enable_progress_bar=False, enable_model_summary=False,
                                 limit_val_batches=0, callbacks=[checkpoint])
                return fit, checkpoint
            loader = DataLoader([batch()], batch_size=None)
            model = models.NNUE(TinyFeatures(), score_target='a-law', score_target_scale=32)
            fit, callback = trainer(1, 'first')
            fit.fit(model, loader)
            restored = models.NNUE.load_from_checkpoint(callback.last_model_path, feature_set=TinyFeatures())
            self.assertEqual((restored.score_target.name, restored.score_target.scale), ('a-law', 32))
            resumed = models.NNUE(TinyFeatures())
            fit, _ = trainer(2, 'resumed')
            fit.fit(resumed, loader, ckpt_path=callback.last_model_path)
            self.assertEqual((resumed.score_target.name, resumed.score_target.scale), ('a-law', 32))
            self.assertEqual(fit.global_step, 2)
            override = models.NNUE(TinyFeatures())
            override.set_score_target_overrides('linear-recip', 128)
            fit, _ = trainer(2, 'override')
            fit.fit(override, loader, ckpt_path=callback.last_model_path)
            self.assertEqual((override.score_target.name, override.score_target.scale), ('linear-recip', 128))


if __name__ == '__main__':
    unittest.main()
