"""Training-only score targets; NNUE inference and output scaling are unchanged.

The DTM curves are independent of the current signed-16-bit score lookup table.
They accept unbounded distances, while the table adapts the existing record
encoding: zero is a draw and signed (32000 - DTM) encodes a decisive result.
"""
import argparse
import math

import torch
from torch import nn


NAMES = ('sigmoid', 'tanh', 'a-law', 'mu-law',
         'linear-invlog', 'linear-exp', 'linear-recip')
DEFAULT_SCALE = 64.0
MATE_VALUE = 32000
MU = 255.0
A = 87.6
VERSION = 1


def validate(name, scale):
    if name not in NAMES:
        raise ValueError('Unknown score target: ' + str(name))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Score target scale must be positive and finite.')


def positive_scale(text):
    try:
        value = float(text)
        validate('sigmoid', value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return value


def add_argparse_args(parser):
    # None distinguishes an omitted option from an explicit checkpoint override.
    parser.add_argument('--score-target', choices=NAMES, default=None,
                        help='Score target mapping. Default: sigmoid for a new run; inherit when resuming.')
    parser.add_argument('--score-target-scale', type=positive_scale, default=None,
                        help='DTM scale T in plies (default: 64); inherit when resuming. Not used by sigmoid.')


def winning_probability(dtm, name, scale=DEFAULT_SCALE):
    """Report curves, with the same initial slope -0.2/T and win floor 0.6.

    Defined separately from the record representation so other encodings can
    use the same curves. In particular, this function needs no maximum DTM.
    """
    validate(name, scale)
    if name == 'sigmoid':
        raise ValueError('Sigmoid maps evaluation scores, not DTM.')
    x = torch.as_tensor(dtm, dtype=torch.float64).clamp_min(0) / scale
    if name == 'tanh':
        return 1 - 0.4 * torch.tanh(x / 2)
    if name in ('mu-law', 'a-law'):
        s = 2 * MU / math.log1p(MU) if name == 'mu-law' else 2 * A / (1 + math.log(A))
        # Equivalent to x/(x+s), also defined at x=0 and x=infinity.
        z = 1 / (1 + s / x)
        if name == 'mu-law':
            compressed = torch.log1p(MU * z) / math.log1p(MU)
        else:
            logarithmic = (1 + torch.log((A * z).clamp_min(torch.finfo(z.dtype).tiny))) / (1 + math.log(A))
            compressed = torch.where(z <= 1 / A, A * z / (1 + math.log(A)), logarithmic)
        return 1 - 0.4 * compressed

    beyond = x.clamp_min(1)
    if name == 'linear-invlog':
        tail = 0.5 / (1 + torch.log(beyond))
    elif name == 'linear-exp':
        tail = 0.5 * torch.exp(-(beyond - 1))
    else:  # linear-recip
        tail = 0.5 / beyond
    residual = torch.where(x <= 1, 1 - x / 2, tail)
    return 0.6 + 0.4 * residual


def make_lookup_table(name, scale=DEFAULT_SCALE):
    validate(name, scale)
    if name == 'sigmoid':
        return None
    scores = torch.arange(-32768, 32768, dtype=torch.int64)
    dtm = (MATE_VALUE - scores.abs()).clamp_min(0)
    winning = winning_probability(dtm, name, scale).float()
    return torch.where(scores > 0, winning,
                       torch.where(scores < 0, 1 - winning, 0.5))


class ScoreTarget(nn.Module):
    """A device-resident 256 KiB LUT for the current packed score format.

    The default sigmoid retains its original expression, including support for
    noninteger scores. Other modes require the integer scores supplied by the
    native loader. Scores beyond +/-MATE_VALUE are treated as terminal results.
    """
    def __init__(self, name='sigmoid', scale=DEFAULT_SCALE):
        super().__init__()
        scale = float(scale)
        validate(name, scale)
        self.name = name
        self.scale = scale
        self.register_buffer('table', make_lookup_table(name, scale), persistent=False)

    def forward(self, scores):
        if self.name == 'sigmoid':
            return (scores / 410).sigmoid()
        indices = scores.to(dtype=torch.int64).reshape(-1) + 32768
        # index_select rejects both negative and too-large indices, without a
        # per-batch CPU/device synchronization to validate the loader's scores.
        return self.table.index_select(0, indices).reshape(scores.shape)

    def config(self):
        return dict(version=VERSION, name=self.name, scale=self.scale)
