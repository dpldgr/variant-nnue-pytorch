import numpy as np
import ctypes
import torch
import os
import sys
import glob
from torch.utils.data import Dataset

from pathlib import Path
import variant

# Resolve next to this module, never an unrelated DLL in the current directory.
_root = Path(__file__).resolve().parent
_names = (('training_data_loader.dll', 'libtraining_data_loader.dll') if sys.platform == 'win32'
          else ('libtraining_data_loader.dylib',) if sys.platform == 'darwin'
          else ('libtraining_data_loader.so',))
_candidates = [_root / name for name in _names]
_dllpath = next((path for path in _candidates if path.is_file()), None)
if _dllpath is None:
    raise RuntimeError('Generic data-loader library is missing. Install the matching prebuilt library or build it once.')
dll = ctypes.cdll.LoadLibrary(str(_dllpath))
try:
    _version = dll.get_loader_abi_version
except AttributeError as exc:
    raise RuntimeError('This is a variant-specific DLL; install the generic data-loader library.') from exc
_version.restype = ctypes.c_uint
_version.argtypes = []
if _version() != 2:
    raise RuntimeError('Incompatible generic data-loader ABI version; rebuild or install the ABI 2 library.')

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

def _bind(name, result, args):
    function = getattr(dll, name)
    function.restype, function.argtypes = result, args
    return function

_error = _bind('get_prp_last_error', ctypes.c_char_p, [])
_create_context = _bind('create_variant_context', ctypes.c_void_p, [ctypes.c_uint] * 6)
_destroy_context = _bind('destroy_variant_context', None, [ctypes.c_void_p])
_count = _bind('generic_record_count', ctypes.c_uint64, [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint])
_create = _bind('create_generic_batch_stream', ctypes.c_void_p,
    [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
     ctypes.c_bool, ctypes.c_bool, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64,
     ctypes.c_uint, ctypes.c_bool])
_create_buffered = _bind('create_buffered_prp_batch_stream', ctypes.c_void_p,
    [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
     ctypes.c_bool, ctypes.c_int, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint,
     ctypes.c_uint64, ctypes.c_uint])
_fetch = _bind('fetch_next_generic_batch', SparseBatchPtr, [ctypes.c_void_p])
_destroy_stream = _bind('destroy_sparse_batch_stream', None, [ctypes.c_void_p])
_destroy_batch = _bind('destroy_sparse_batch', None, [SparseBatchPtr])


class NativeVariant:
    def __init__(self, config):
        self.handle = _create_context(config.FILES, config.RANKS, config.PIECE_TYPES,
                                      config.PIECE_COUNT, config.KING_SQUARES, config.USE_POCKETS)
        if not self.handle:
            raise RuntimeError(_error().decode('utf-8', errors='replace'))

    def close(self):
        if getattr(self, 'handle', None):
            _destroy_context(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def validate_record_bytes(value):
    if type(value) is not int or not 7 <= value <= 144:
        raise ValueError('record bytes must be an integer in 7..144')
    return value


def prp_settings(prp_mode=None, prp_block_size=None, prp_buffer_blocks=None):
    settings = dict(prp_mode='buffered' if prp_mode is None else prp_mode,
                    prp_block_size=16384 if prp_block_size is None else prp_block_size,
                    prp_buffer_blocks=64 if prp_buffer_blocks is None else prp_buffer_blocks)
    if settings['prp_mode'] not in ('buffered', 'global'):
        raise ValueError('PRP mode must be buffered or global')
    for name, limit in (('prp_block_size', 1 << 64), ('prp_buffer_blocks', 1 << 32)):
        value = settings[name]
        if type(value) is not int or not 0 < value < limit:
            raise ValueError(name.replace('_', '-') + ' must be a positive integer below ' + str(limit))
    return settings


def _checked_next(provider):
    batch = provider.fetch_next(provider.stream)
    if not batch:
        error = provider.error()
        if error:
            raise RuntimeError(error.decode('utf-8', errors='replace'))
        raise StopIteration
    try:
        if batch.contents.num_inputs != provider.expected_inputs:
            raise RuntimeError('Native/Python feature dimensions disagree')
        return batch.contents.get_tensors(provider.device)
    finally:
        provider.destroy_part(batch)


class GenericBatchProvider(TrainingDataProvider):
    def __init__(self, feature_set, filename, batch_size, cyclic, num_workers, filtered,
                 random_fen_skipping, device, record_bytes, config, seed=42, epoch=0, shuffle=False,
                 prp_mode=None, prp_block_size=None, prp_buffer_blocks=None):
        validate_record_bytes(record_bytes)
        settings = prp_settings(prp_mode, prp_block_size, prp_buffer_blocks) if shuffle else None
        self.config = config or variant.default_config()
        self.native = NativeVariant(self.config)
        self.device, self.error = device, _error
        self.fetch_next, self.destroy_part, self.destroy_stream = _fetch, _destroy_batch, _destroy_stream
        if shuffle and settings['prp_mode'] == 'buffered':
            self.stream = _create_buffered(self.native.handle, feature_set.encode('utf-8'), num_workers,
                                          os.fsencode(filename), batch_size, filtered, random_fen_skipping,
                                          seed % (1 << 64), epoch, record_bytes,
                                          settings['prp_block_size'], settings['prp_buffer_blocks'])
        else:
            self.stream = _create(self.native.handle, feature_set.encode('utf-8'), num_workers,
                                  os.fsencode(filename), batch_size, cyclic, filtered, random_fen_skipping,
                                  seed % (1 << 64), epoch, record_bytes, shuffle)
        if not self.stream:
            message = _error().decode('utf-8', errors='replace')
            self.native.close()
            raise RuntimeError(message)
        import features
        self.expected_inputs = features.get_feature_set_from_name(feature_set, self.config).num_features

    def __next__(self):
        return _checked_next(self)

    def close(self):
        super().close()
        if hasattr(self, 'native'):
            self.native.close()


class SparseBatchProvider(GenericBatchProvider):
    def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1,
                 filtered=False, random_fen_skipping=0, device='cpu', record_bytes=72, config=None):
        super().__init__(feature_set, filename, batch_size, cyclic, num_workers, filtered,
                         random_fen_skipping, device, record_bytes, config)


class SparseBatchDataset(torch.utils.data.IterableDataset):
    def __init__(self, feature_set, filename, batch_size, cyclic=True, num_workers=1,
                 filtered=False, random_fen_skipping=0, device='cpu', record_bytes=72, config=None):
        super().__init__()
        self.config = config or variant.default_config()
        self.args = dict(feature_set=feature_set, filename=filename, batch_size=batch_size,
                         cyclic=cyclic, num_workers=num_workers, filtered=filtered,
                         random_fen_skipping=random_fen_skipping, device=device,
                         record_bytes=validate_record_bytes(record_bytes), config=self.config)

    def __iter__(self):
        return SparseBatchProvider(**self.args)

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


def training_data_record_count(filename, record_bytes=72, config=None):
    validate_record_bytes(record_bytes)
    native = NativeVariant(config or variant.default_config())
    try:
        result = _count(native.handle, os.fsencode(filename), record_bytes)
        if not result:
            raise RuntimeError(_error().decode('utf-8', errors='replace'))
        return result
    finally:
        native.close()


class PrpSparseBatchProvider(GenericBatchProvider):
    def __init__(self, feature_set, filename, batch_size, num_workers=1, filtered=False,
                 random_fen_skipping=0, device='cpu', seed=42, epoch=0, record_bytes=72, config=None,
                 prp_mode=None, prp_block_size=None, prp_buffer_blocks=None):
        super().__init__(feature_set, filename, batch_size, False, num_workers, filtered,
                         random_fen_skipping, device, record_bytes, config, seed, epoch, True,
                         prp_mode, prp_block_size, prp_buffer_blocks)


class PrpBatchDataset(torch.utils.data.IterableDataset):
    """One complete source traversal per iteration; no cyclic batch padding.

    Length is intentionally unspecified: filtering changes the number of yielded
    batches. Record counting and reads are delegated to the storage backend.
    """
    def __init__(self, feature_set, filename, batch_size, num_workers=1,
                 filtered=False, random_fen_skipping=0, device='cpu', seed=42, record_bytes=72, config=None,
                 prp_mode=None, prp_block_size=None, prp_buffer_blocks=None):
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
        self.record_bytes = validate_record_bytes(record_bytes)
        self.config = config or variant.default_config()
        self._prp_overrides = dict(prp_mode=prp_mode, prp_block_size=prp_block_size,
                                  prp_buffer_blocks=prp_buffer_blocks)
        self.prp_settings = prp_settings(**self._prp_overrides)
        self.record_count = training_data_record_count(filename, record_bytes, self.config)
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
            self.filtered, self.random_fen_skipping, self.device, self.seed, epoch, self.record_bytes,
            self.config, **self.prp_settings)
        try:
            yield from provider
            self.epoch_complete = True
        finally:
            provider.close()

    def state_dict(self):
        return dict(version=2, next_epoch=self.next_epoch, epoch_complete=self.epoch_complete, seed=self.seed,
                    record_count=self.record_count, batch_size=self.batch_size,
                    feature_set=self.feature_set, filtered=self.filtered,
                    random_fen_skipping=self.random_fen_skipping, record_bytes=self.record_bytes,
                    variant_config=self.config.to_dict(), **self.prp_settings)

    def load_state_dict(self, state):
        version = state.get('version', 1)
        if version not in (1, 2):
            raise ValueError('Unsupported PRP checkpoint version')
        if not state.get('epoch_complete', False):
            raise ValueError('PRP resumption currently requires a completed-epoch checkpoint.')
        if 'variant_config' in state and not self.config.compatible_with(variant.VariantConfig.from_dict(state['variant_config'])):
            raise ValueError('PRP checkpoint does not match current variant configuration')
        expected = self.state_dict()
        if version == 2 and not all(key in state for key in self._prp_overrides):
            raise ValueError('PRP checkpoint is missing its read strategy settings')
        saved_prp = (prp_settings(prp_mode='global') if version == 1 else
                     prp_settings(**{key: state[key] for key in self._prp_overrides}))
        resumed_prp = prp_settings(**{key: saved_prp[key] if value is None else value
                                     for key, value in self._prp_overrides.items()})
        for key in expected:
            # A lossless conversion can change record width without changing
            # logical positions or permutation state. Legacy states omit it.
            if key not in ('version', 'next_epoch', 'epoch_complete', 'record_bytes', 'variant_config',
                           *self._prp_overrides) and state.get(key) != expected[key]:
                raise ValueError('PRP checkpoint does not match current loader setting: ' + key)
        self.next_epoch = int(state['next_epoch'])
        if self.next_epoch < 0:
            raise ValueError('Invalid PRP checkpoint epoch.')
        self.prp_settings = resumed_prp
        self.epoch_complete = True
