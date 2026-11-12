import chess
import torch
import feature_block
from collections import OrderedDict
from feature_block import *

import variant

class Layout:
  def __init__(self, config):
    self.config = config
    self.NUM_SQ = config.SQUARES
    self.NUM_KSQ = config.KING_SQUARES
    self.NUM_PT = config.PIECES
    self.NUM_PLANES = (self.NUM_SQ * self.NUM_PT + 1)

def orient(is_white_pov: bool, sq: int, layout):
  return sq % layout.config.FILES + (layout.config.RANKS - 1 - (sq // layout.config.FILES)) * layout.config.FILES if not is_white_pov else sq

def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, piece_type: int, color: bool, layout):
  p_idx = (piece_type - 1) * 2 + (color != is_white_pov)
  return 1 + orient(is_white_pov, sq, layout) + p_idx * layout.NUM_SQ + king_sq * layout.NUM_PLANES

def halfka_psqts(layout):
  values = [0] * (layout.NUM_PLANES * layout.NUM_SQ)

  for ksq in range(layout.NUM_KSQ):
    for s in range(layout.NUM_SQ):
      for pt, val in layout.config.PIECE_VALUES.items():
        idxw = halfka_idx(True, ksq, s, pt, chess.WHITE, layout)
        idxb = halfka_idx(True, ksq, s, pt, chess.BLACK, layout)
        values[idxw] = val
        values[idxb] = -val

  return values

class Features(FeatureBlock):
  def __init__(self, config=None):
    layout = self.layout = Layout(config or variant.default_config())
    super(Features, self).__init__('HalfKA', 0x5f134cb8, OrderedDict([('HalfKA', layout.NUM_PLANES * layout.NUM_SQ)]))

  def get_active_features(self, board: chess.Board):
    layout = self.layout
    def piece_features(turn):
      indices = torch.zeros(layout.NUM_PLANES * layout.NUM_SQ)
      for sq, p in board.piece_map().items():
        indices[halfka_idx(turn, orient(turn, board.king(turn), layout), sq, p.piece_type, p.color, layout)] = 1.0
      return indices
    return (piece_features(chess.WHITE), piece_features(chess.BLACK))

  def get_initial_psqt_features(self):
    layout = self.layout
    return halfka_psqts(layout)

class FactorizedFeatures(FeatureBlock):
  def __init__(self, config=None):
    layout = self.layout = Layout(config or variant.default_config())
    super(FactorizedFeatures, self).__init__('HalfKA^', 0x5f134cb8, OrderedDict([('HalfKA', layout.NUM_PLANES * layout.NUM_SQ), ('A', layout.NUM_SQ * layout.NUM_PT)]))

  def get_active_features(self, board: chess.Board):
    layout = self.layout
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def get_feature_factors(self, idx):
    layout = self.layout
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    a_idx = idx % layout.NUM_PLANES - 1

    return [idx, self.get_factor_base_feature('A') + a_idx]

  def get_initial_psqt_features(self):
    layout = self.layout
    return halfka_psqts(layout) + [0] * (layout.NUM_SQ * layout.NUM_PT)

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
