"""Run from the trainer root after building the shared library.

    python -m unittest discover -s tests -p test_prp_loader.py -v
"""
import pathlib
import struct
import tempfile
import unittest

import torch

import nnue_dataset as data


# Valid packed Metal Chess K+N versus K record from the mate-table exporter.
# The same 8x8/six-type layout is used by the stock chess loader. Tests replace
# its score with a unique record ID; this makes dropped/repeated records visible
# independently of the PRP implementation.
BASE_RECORD = bytes.fromhex(
    '0002000000000000800400000000000000000002000000000000000000000000'
    '0000000000000000000000000000000000000000000000000000000000000000'
    'cd7c000001000100')


def write_records(path, count):
    with open(path, 'wb') as output:
        for index in range(count):
            record = bytearray(BASE_RECORD)
            struct.pack_into('<h', record, 64, index)
            output.write(record)


def collect(provider):
    result, sizes = [], []
    try:
        for batch in provider:
            sizes.append(len(batch[7]))
            result.extend(int(x) for x in batch[7].reshape(-1).tolist())
    finally:
        provider.close()
    return result, sizes


class PrpLoaderTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = str(pathlib.Path(self.temp.name) / 'numbered.bin')
        write_records(self.path, 257)

    def tearDown(self):
        self.temp.cleanup()

    def provider(self, **kwargs):
        return data.PrpSparseBatchProvider('HalfKAv2', self.path, 64, **kwargs)

    def dataset(self, **kwargs):
        return data.PrpBatchDataset('HalfKAv2', self.path, 64, **kwargs)

    def test_complete_epochs_and_partial_batch(self):
        dataset = self.dataset(num_workers=4)
        orders = []
        for epoch in range(3):
            batches = list(dataset)
            order = [int(x) for batch in batches for x in batch[7].reshape(-1).tolist()]
            self.assertEqual(sorted(order), list(range(257)))
            self.assertEqual([len(batch[7]) for batch in batches], [64, 64, 64, 64, 1])
            self.assertEqual(dataset.next_epoch, epoch + 1)
            orders.append(order)
        self.assertNotEqual(orders[0], orders[1])
        self.assertNotEqual(orders[1], orders[2])

    def test_order_independent_of_native_worker_count(self):
        expected, _ = collect(self.provider(num_workers=1, seed=17, epoch=5))
        for workers in (2, 4, 8):
            actual, _ = collect(self.provider(num_workers=workers, seed=17, epoch=5))
            self.assertEqual(actual, expected)
        different, _ = collect(self.provider(num_workers=4, seed=18, epoch=5))
        self.assertNotEqual(different, expected)

    def test_single_record_and_exact_batch_boundary(self):
        for count, sizes in ((1, [1]), (64, [64]), (128, [64, 64])):
            write_records(self.path, count)
            records, actual_sizes = collect(self.provider(num_workers=8))
            self.assertEqual(sorted(records), list(range(count)))
            self.assertEqual(actual_sizes, sizes)

    def test_filtering_is_finite_and_reproducible(self):
        first, _ = collect(self.provider(num_workers=1, random_fen_skipping=3, epoch=2))
        second, _ = collect(self.provider(num_workers=8, random_fen_skipping=3, epoch=2))
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertTrue(0 < len(first) < 257)

    def test_epoch_boundary_state(self):
        uninterrupted = self.dataset(num_workers=4, random_fen_skipping=3)
        list(uninterrupted)
        state = uninterrupted.state_dict()
        expected = [batch[7].clone() for batch in uninterrupted]
        restored = self.dataset(num_workers=1, random_fen_skipping=3)
        restored.load_state_dict(state)
        actual = [batch[7] for batch in restored]
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(restored.next_epoch, 2)
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.dataset(seed=99, random_fen_skipping=3).load_state_dict(state)

    def test_early_close_joins_prefetch_workers(self):
        for _ in range(10):
            provider = self.provider(num_workers=8)
            self.assertEqual(len(next(provider)[7]), 64)
            provider.close()
            provider.close()

    def test_incomplete_epoch_cannot_silently_resume_at_next_epoch(self):
        dataset = self.dataset(num_workers=4)
        iterator = iter(dataset)
        next(iterator)
        state = dataset.state_dict()
        iterator.close()
        with self.assertRaisesRegex(ValueError, 'completed-epoch'):
            self.dataset().load_state_dict(state)

    def test_existing_sequential_provider(self):
        provider = data.SparseBatchProvider('HalfKAv2', self.path, 64, cyclic=False)
        records, sizes = collect(provider)
        self.assertEqual(records, list(range(257)))
        self.assertEqual(sizes, [64, 64, 64, 64, 1])

    def test_existing_feature_sets(self):
        for name in ('HalfKP', 'HalfKP^', 'HalfKA', 'HalfKA^', 'HalfKAv2', 'HalfKAv2^'):
            with self.subTest(feature_set=name):
                sequential = data.SparseBatchProvider(name, self.path, 64, cyclic=False)
                prp = data.PrpSparseBatchProvider(name, self.path, 64, num_workers=4)
                try:
                    baseline = next(sequential)
                    # Position payloads are identical; only their unique scores
                    # differ. All actual feature tensors must therefore match.
                    for batch in prp:
                        for field in (0, 1, 2, 3, 4, 5, 6, 8, 9):
                            expected = baseline[field][:1].expand_as(batch[field])
                            torch.testing.assert_close(batch[field], expected, rtol=0, atol=0)
                finally:
                    sequential.close()
                    prp.close()

    def test_backend_errors_reach_python(self):
        self.assertEqual(data.training_data_record_count(self.path), 257)
        with self.assertRaisesRegex(RuntimeError, 'Cannot open'):
            data.training_data_record_count(self.path + '.missing.bin')
        bad = pathlib.Path(self.temp.name) / 'bad.bin'
        bad.write_bytes(b'')
        with self.assertRaisesRegex(RuntimeError, 'empty'):
            data.training_data_record_count(str(bad))
        bad.write_bytes(b'x' * 73)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete'):
            data.PrpSparseBatchProvider('HalfKAv2', str(bad), 64)
        with self.assertRaisesRegex(RuntimeError, 'backend'):
            data.training_data_record_count(self.path + '.other')
        with self.assertRaisesRegex(RuntimeError, 'Unknown feature_set'):
            data.PrpSparseBatchProvider('unknown', self.path, 64)


if __name__ == '__main__':
    unittest.main()
