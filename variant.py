"""Shared runtime variant configuration; no generated Python or C++ constants."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path

FIELDS = ('RANKS', 'FILES', 'KING_SQUARES', 'PIECE_TYPES', 'PIECE_COUNT', 'USE_POCKETS')


@dataclass(frozen=True)
class VariantConfig:
    RANKS: int
    FILES: int
    KING_SQUARES: int
    PIECE_TYPES: int
    PIECE_COUNT: int
    USE_POCKETS: bool
    _piece_values: tuple

    @classmethod
    def from_dict(cls, value):
        expected = set(FIELDS) | {'PIECE_VALUES'}
        if not isinstance(value, dict) or set(value) != expected:
            raise ValueError('Variant JSON must contain exactly: ' + ', '.join(sorted(expected)))
        for name in FIELDS[:-1]:
            if type(value[name]) is not int or value[name] < 1:
                raise ValueError(name + ' must be a positive integer')
        if type(value['USE_POCKETS']) is not bool:
            raise ValueError('USE_POCKETS must be a Boolean')
        squares = value['RANKS'] * value['FILES']
        if squares > 65535 or value['PIECE_TYPES'] > 32766 or value['PIECE_COUNT'] > 65535:
            raise ValueError('Variant exceeds native square/type/count capacity')
        if value['KING_SQUARES'] > squares:
            raise ValueError('KING_SQUARES exceeds board size')
        values = value['PIECE_VALUES']
        if not isinstance(values, dict):
            raise ValueError('PIECE_VALUES must be an object')
        normalized = {}
        for key, number in values.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise ValueError('PIECE_VALUES keys must be integer indices') from exc
            if str(index) != str(key) or not 1 <= index <= value['PIECE_TYPES'] or type(number) is not int:
                raise ValueError('Invalid PIECE_VALUES index or value')
            normalized[index] = number
        return cls(*(value[k] for k in FIELDS), tuple(sorted(normalized.items())))

    @property
    def SQUARES(self): return self.RANKS * self.FILES
    @property
    def PIECES(self): return 2 * self.PIECE_TYPES
    @property
    def POCKETS(self): return 2 * self.FILES if self.USE_POCKETS else 0
    @property
    def PIECE_VALUES(self): return dict(self._piece_values)

    def to_dict(self):
        result = {key: getattr(self, key) for key in FIELDS}
        result['PIECE_VALUES'] = {str(k): v for k, v in self._piece_values}
        return result

    def compatible_with(self, other):
        return all(getattr(self, key) == getattr(other, key) for key in FIELDS)

    @property
    def fingerprint(self):
        # Initialization values are recorded, but do not define feature identity.
        payload = {key: getattr(self, key) for key in FIELDS}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_config(path):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value: raise ValueError('Duplicate JSON key: ' + key)
            value[key] = item
        return value
    with open(path, encoding='utf-8') as stream:
        return VariantConfig.from_dict(json.load(stream, object_pairs_hook=unique))


@lru_cache(maxsize=1)
def default_config():
    return load_config(Path(__file__).with_name('variant.json'))


def resolve_config(path=None, saved=None):
    stored = VariantConfig.from_dict(saved) if saved is not None else None
    current = load_config(path) if path is not None else (stored or default_config())
    if stored is not None and not current.compatible_with(stored):
        raise ValueError('Variant configuration does not match saved model/checkpoint')
    return current


def __getattr__(name):
    # Compatibility for external scripts importing the old generated constants.
    if name in (*FIELDS, 'SQUARES', 'PIECES', 'POCKETS', 'PIECE_VALUES'):
        return getattr(default_config(), name)
    raise AttributeError(name)


def saved_config(path):
    """Read metadata before model dimensions are constructed; no DLL dependency."""
    if path is None or not str(path).endswith(('.ckpt', '.pt')):
        return None
    import torch
    value = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(value, dict):
        return value.get('variant_config')
    config = getattr(getattr(value, 'feature_set', None), 'config', None)
    return config.to_dict() if config is not None else None
