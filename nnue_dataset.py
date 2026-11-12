import numpy as np
import ctypes
import torch
import os
import sys
import glob
from torch.utils.data import Dataset

local_dllpath = [n for n in glob.glob('./*training_data_loader.*') if n.endswith('.so') or n.endswith('.dll') or n.endswith('.dylib')]
if not local_dllpath:
    print('Cannot find data_loader shared library.')
    sys.exit(1)
dllpath = os.path.abspath(local_dllpath[0])
dll = ctypes.cdll.LoadLibrary(dllpath)

class SparseBatch(ctypes.Structure):
    _fields_ = [
        ('num_inputs', ctypes.c_int),
        ('size', ctypes.c_int),
        ('is_white', ctypes.POINTER(ctypes.c_float)),
        ('outcome', ctypes.POINTER(ctypes.c_float)),
        ('score', ctypes.POINTER(ctypes.c_float)),
        ('num_active_white_features', ctypes.c_int),
        ('num_active_black_features', ctypes.c_int),
        ('max_active_features', ctypes.c_int),
        ('white', ctypes.POINTER(ctypes.c_int)),
        ('black', ctypes.POINTER(ctypes.c_int)),
        ('white_values', ctypes.POINTER(ctypes.c_float)),
        ('black_values', ctypes.POINTER(ctypes.c_float)),
        ('psqt_indices', ctypes.POINTER(ctypes.c_int)),
        ('layer_stack_indices', ctypes.POINTER(ctypes.c_int)),
    ]

    def get_tensors(self, device):
        # Pinned memory only helps (and is only available) for transfers to a GPU.
        pin = torch.device(device).type == 'cuda'
        def to_device(t):
            # The numpy arrays alias the C++ batch, which is freed right after this call,
            # so the data must always be copied (pin_memory() copies, .to('cpu') would not).
            if pin:
                return t.pin_memory().to(device=device, non_blocking=True)
            return t.to(device=device, copy=True)
        white_values = to_device(torch.from_numpy(np.ctypeslib.as_array(self.white_values, shape=(self.size, self.max_active_features))))
        black_values = to_device(torch.from_numpy(np.ctypeslib.as_array(self.black_values, shape=(self.size, self.max_active_features))))
        white_indices = to_device(torch.from_numpy(np.ctypeslib.as_array(self.white, shape=(self.size, self.max_active_features))))
        black_indices = to_device(torch.from_numpy(np.ctypeslib.as_array(self.black, shape=(self.size, self.max_active_features))))
        us = to_device(torch.from_numpy(np.ctypeslib.as_array(self.is_white, shape=(self.size, 1))))
        them = 1.0 - us
        outcome = to_device(torch.from_numpy(np.ctypeslib.as_array(self.outcome, shape=(self.size, 1))))
        score = to_device(torch.from_numpy(np.ctypeslib.as_array(self.score, shape=(self.size, 1))))
        psqt_indices = to_device(torch.from_numpy(np.ctypeslib.as_array(self.psqt_indices, shape=(self.size,))).long())
        layer_stack_indices = to_device(torch.from_numpy(np.ctypeslib.as_array(self.layer_stack_indices, shape=(self.size,))).long())
        return us, them, white_indices, white_values, black_indices, black_values, outcome, score, psqt_indices, layer_stack_indices

SparseBatchPtr = ctypes.POINTER(SparseBatch)


class TrainingDataProvider:
    def __init__(
        self,
        feature_set,
        create_stream,
        destroy_stream,
        fetch_next,
        destroy_part,
        filename,
        cyclic,
        num_workers,
        batch_size=None,
        filtered=False,
        random_fen_skipping=0,
        device='cpu',
        stream_args=()):

        self.feature_set = feature_set.encode('utf-8')
        self.create_stream = create_stream
        self.destroy_stream = destroy_stream
        self.fetch_next = fetch_next
        self.destroy_part = destroy_part
        self.filename = filename.encode('utf-8')
        self.cyclic = cyclic
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping
        self.device = device

        self.stream = self.create_stream(
            self.feature_set,
            self.num_workers,
            self.filename,
            self.batch_size,
            cyclic,
            filtered,
            random_fen_skipping,
            *stream_args
        )

    def __iter__(self):
        return self

    def __next__(self):
        v = self.fetch_next(self.stream)

        if v:
            tensors = v.contents.get_tensors(self.device)
            self.destroy_part(v)
            return tensors
        else:
            raise StopIteration

    def __del__(self):
        self.close()

    def close(self):
        if getattr(self, "stream", None):
            self.destroy_stream(self.stream)
            self.stream = None

create_sparse_batch_stream = dll.create_sparse_batch_stream
create_sparse_batch_stream.restype = ctypes.c_void_p
create_sparse_batch_stream.argtypes = [
    ctypes.c_char_p,  # feature_set
    ctypes.c_int,     # num_workers
    ctypes.c_char_p,  # filename
    ctypes.c_int,     # batch_size
    ctypes.c_bool,    # cyclic
    ctypes.c_bool,    # filtered
    ctypes.c_int      # random_fen_skipping
]
destroy_sparse_batch_stream = dll.destroy_sparse_batch_stream
destroy_sparse_batch_stream.argtypes = [ctypes.c_void_p]

fetch_next_sparse_batch = dll.fetch_next_sparse_batch
fetch_next_sparse_batch.restype = SparseBatchPtr
fetch_next_sparse_batch.argtypes = [ctypes.c_void_p]
destroy_sparse_batch = dll.destroy_sparse_batch


class SparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu'):
        super(SparseBatchProvider, self).__init__(
            feature_set,
            create_sparse_batch_stream,
            destroy_sparse_batch_stream,
            fetch_next_sparse_batch,
            destroy_sparse_batch,
            filename,
            cyclic,
            num_workers,
            batch_size,
            filtered,
            random_fen_skipping,
            device)

class SparseBatchDataset(torch.utils.data.IterableDataset):
  def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1, filtered=False, random_fen_skipping=0, device='cpu'):
    super(SparseBatchDataset).__init__()
    self.feature_set = feature_set
    self.filename = filename
    self.batch_size = batch_size
    self.cyclic = cyclic
    self.num_workers = num_workers
    self.filtered = filtered
    self.random_fen_skipping = random_fen_skipping
    self.device = device

  def __iter__(self):
    return SparseBatchProvider(self.feature_set, self.filename, self.batch_size, cyclic=self.cyclic, num_workers=self.num_workers, filtered=self.filtered, random_fen_skipping=self.random_fen_skipping, device=self.device)

class FixedNumBatchesDataset(Dataset):
  def __init__(self, dataset, num_batches):
    super(FixedNumBatchesDataset, self).__init__()
    self.dataset = dataset
    self.iter = iter(self.dataset)
    self.num_batches = num_batches

  def __len__(self):
    return self.num_batches

  def __getitem__(self, idx):
    return next(self.iter)


def _prp_api():
    # Resolve only for PRP, so sequential mode still works with an older DLL.
    try:
        create = dll.create_prp_sparse_batch_stream
        fetch = dll.fetch_next_prp_sparse_batch
        count = dll.training_data_record_count
        error = dll.get_prp_last_error
    except AttributeError as exc:
        raise RuntimeError('Rebuild the existing data-loader library to enable PRP loading.') from exc
    create.restype = ctypes.c_void_p
    create.argtypes = create_sparse_batch_stream.argtypes + [ctypes.c_uint64, ctypes.c_uint64]
    fetch.restype = SparseBatchPtr
    fetch.argtypes = [ctypes.c_void_p]
    count.restype = ctypes.c_uint64
    count.argtypes = [ctypes.c_char_p]
    error.restype = ctypes.c_char_p
    error.argtypes = []
    return create, fetch, count, error


def training_data_record_count(filename):
    _, _, count, error = _prp_api()
    result = count(os.fsencode(filename))
    if not result:
        raise RuntimeError(error().decode('utf-8', errors='replace'))
    return result


class PrpSparseBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, batch_size, num_workers=1,
                 filtered=False, random_fen_skipping=0, device='cpu', seed=42, epoch=0):
        create, fetch, _, self.error = _prp_api()
        super().__init__(feature_set, create, destroy_sparse_batch_stream, fetch,
                         destroy_sparse_batch, filename, False, num_workers,
                         batch_size, filtered, random_fen_skipping, device,
                         stream_args=(seed, epoch))
        if not self.stream:
            raise RuntimeError(self.error().decode('utf-8', errors='replace'))

    def __next__(self):
        batch = self.fetch_next(self.stream)
        if not batch:
            error = self.error()
            if error:
                raise RuntimeError(error.decode('utf-8', errors='replace'))
            raise StopIteration
        try:
            return batch.contents.get_tensors(self.device)
        finally:
            self.destroy_part(batch)


class PrpBatchDataset(torch.utils.data.IterableDataset):
    """One complete source traversal per iteration; no cyclic batch padding.

    Length is intentionally unspecified: filtering changes the number of yielded
    batches. Record counting and reads are delegated to the storage backend.
    """
    def __init__(self, feature_set, filename, batch_size, num_workers=1,
                 filtered=False, random_fen_skipping=0, device='cpu', seed=42):
        super().__init__()
        if batch_size < 1 or num_workers < 1 or random_fen_skipping < 0:
            raise ValueError('PRP requires positive batch size/workers and nonnegative skipping.')
        self.feature_set = feature_set
        self.filename = filename
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.filtered = filtered
        self.random_fen_skipping = random_fen_skipping
        self.device = device
        self.seed = seed % (1 << 64)
        self.record_count = training_data_record_count(filename)
        self.next_epoch = 0
        self.epoch_complete = True

    def __iter__(self):
        # Generator body starts when the loader consumes its first batch, not
        # when an unused iterator is constructed during trainer setup.
        if torch.utils.data.get_worker_info() is not None:
            raise RuntimeError('Use native --num-workers, not Python DataLoader workers, with PRP.')
        epoch = self.next_epoch
        self.next_epoch += 1
        self.epoch_complete = False
        provider = PrpSparseBatchProvider(
            self.feature_set, self.filename, self.batch_size, self.num_workers,
            self.filtered, self.random_fen_skipping, self.device, self.seed, epoch)
        try:
            yield from provider
            self.epoch_complete = True
        finally:
            provider.close()

    def state_dict(self):
        return dict(version=1, next_epoch=self.next_epoch, epoch_complete=self.epoch_complete, seed=self.seed,
                    record_count=self.record_count, batch_size=self.batch_size,
                    feature_set=self.feature_set, filtered=self.filtered,
                    random_fen_skipping=self.random_fen_skipping)

    def load_state_dict(self, state):
        if not state.get('epoch_complete', False):
            raise ValueError('PRP resumption currently requires a completed-epoch checkpoint.')
        expected = self.state_dict()
        for key in expected:
            if key not in ('next_epoch', 'epoch_complete') and state.get(key) != expected[key]:
                raise ValueError('PRP checkpoint does not match current loader setting: ' + key)
        self.next_epoch = int(state['next_epoch'])
        if self.next_epoch < 0:
            raise ValueError('Invalid PRP checkpoint epoch.')
        self.epoch_complete = True
