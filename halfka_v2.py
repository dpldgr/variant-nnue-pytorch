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
    self.NUM_PT_REAL = config.PIECES - (self.NUM_KSQ != 1)
    self.NUM_PT_VIRTUAL = config.PIECES
    self.NUM_PLANES_REAL = self.NUM_SQ * self.NUM_PT_REAL + (self.NUM_PT_REAL - (self.NUM_KSQ != 1)) * config.POCKETS
    self.NUM_PLANES_VIRTUAL = self.NUM_SQ * self.NUM_PT_VIRTUAL + (self.NUM_PT_REAL - (self.NUM_KSQ != 1)) * config.POCKETS
    self.NUM_INPUTS = self.NUM_PLANES_REAL * self.NUM_KSQ

def orient(is_white_pov: bool, sq: int, layout):
  return sq % layout.config.FILES + (layout.config.RANKS - 1 - (sq // layout.config.FILES)) * layout.config.FILES if not is_white_pov else sq

def halfka_idx(is_white_pov: bool, king_sq: int, sq: int, piece_type: int, color: bool, layout):
  p_idx = (piece_type - 1) * 2 + (color != is_white_pov)
  if layout.NUM_PT_REAL % 2 and p_idx == layout.NUM_PT_REAL:
    # merge kings into one plane
    p_idx -= 1
  return orient(is_white_pov, sq, layout) + p_idx * layout.NUM_SQ + king_sq * layout.NUM_PLANES_REAL

def halfka_hand_idx(is_white_pov: bool, king_sq: int, handCount: int, piece_type: int, color: bool, layout):
  p_idx = (piece_type - 1) * 2 + (color != is_white_pov)
  return handCount + p_idx * layout.config.POCKETS + layout.NUM_SQ * layout.NUM_PT_REAL + king_sq * layout.NUM_PLANES_REAL

def map_king(sq: int, layout):
  # palace squares for Xiangi/Janggi
  if layout.NUM_KSQ == 9 and layout.NUM_KSQ != layout.NUM_SQ:
    if sq > layout.config.FILES * ((layout.config.RANKS + 1) // 2):
      # in order to allow unambiguously detecting opposing kings, just return value out of range
      return sq
    # map accessible king squares skipping the gaps
    return (sq - 6 * (sq // layout.config.FILES) - 3) % layout.NUM_KSQ
  return sq % layout.NUM_KSQ

def halfka_psqts(layout):
  values = [0] * (layout.NUM_PLANES_REAL * layout.NUM_KSQ)

  for ksq in range(layout.NUM_KSQ):
    for s in range(layout.NUM_SQ):
      for pt, val in layout.config.PIECE_VALUES.items():
        idxw = halfka_idx(True, ksq, s, pt, chess.WHITE, layout)
        idxb = halfka_idx(True, ksq, s, pt, chess.BLACK, layout)
        values[idxw] = val
        values[idxb] = -val
    for i in range(layout.config.POCKETS):
      for pt, val in layout.config.PIECE_VALUES.items():
        idxw = halfka_hand_idx(True, ksq, i, pt, chess.WHITE, layout)
        idxb = halfka_hand_idx(True, ksq, i, pt, chess.BLACK, layout)
        values[idxw] = val
        values[idxb] = -val

  return values

class Features(FeatureBlock):
  def __init__(self, config=None):
    layout = self.layout = Layout(config or variant.default_config())
    super(Features, self).__init__('HalfKAv2', 0x5f234cb8, OrderedDict([('HalfKAv2', layout.NUM_PLANES_REAL * layout.NUM_KSQ)]))

  def get_active_features(self, board: chess.Board):
    layout = self.layout
    def piece_features(turn):
      indices = torch.zeros(layout.NUM_PLANES_REAL * layout.NUM_KSQ)
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
    super(FactorizedFeatures, self).__init__('HalfKAv2^', 0x5f234cb8, OrderedDict([('HalfKAv2', layout.NUM_PLANES_REAL * layout.NUM_KSQ), ('A', layout.NUM_PLANES_VIRTUAL)]))

  def get_active_features(self, board: chess.Board):
    layout = self.layout
    raise Exception('Not supported yet, you must use the c++ data loader for factorizer support during training')

  def get_feature_factors(self, idx):
    layout = self.layout
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    a_idx = idx % layout.NUM_PLANES_REAL
    k_idx = idx // layout.NUM_PLANES_REAL

    if layout.NUM_PT_VIRTUAL != layout.NUM_PT_REAL:
      if a_idx // layout.NUM_SQ == layout.NUM_PT_REAL - 1 and k_idx != map_king(a_idx % layout.NUM_SQ, layout):
        # is king piece, but not ours
        a_idx += layout.NUM_SQ
      elif a_idx >= layout.NUM_SQ * layout.NUM_PT_REAL:
        # pockets
        a_idx += layout.NUM_SQ

    return [idx, self.get_factor_base_feature('A') + a_idx]

  def get_initial_psqt_features(self):
    layout = self.layout
    return halfka_psqts(layout) + [0] * layout.NUM_PLANES_VIRTUAL

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
