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
    self.NUM_PT = config.PIECES - 2
    self.NUM_PLANES = (self.NUM_SQ * self.NUM_PT + 1)

def orient(is_white_pov: bool, sq: int, layout):
  return (layout.config.FILES - 1 - sq % layout.config.FILES) + (layout.config.RANKS - 1 - (sq // layout.config.FILES)) * layout.config.FILES if not is_white_pov else sq

def halfkp_idx(is_white_pov: bool, king_sq: int, sq: int, piece_type: int, color: bool, layout):
  p_idx = (piece_type - 1) * 2 + (color != is_white_pov)
  return 1 + orient(is_white_pov, sq, layout) + p_idx * layout.NUM_SQ + king_sq * layout.NUM_PLANES

class Features(FeatureBlock):
  def __init__(self, config=None):
    layout = self.layout = Layout(config or variant.default_config())
    super(Features, self).__init__('HalfKP', 0x5d69d5b8, OrderedDict([('HalfKP', layout.NUM_PLANES * layout.NUM_SQ)]))

  def get_active_features(self, board: chess.Board):
    layout = self.layout
    def piece_features(turn):
      indices = torch.zeros(layout.NUM_PLANES * layout.NUM_SQ)
      for sq, p in board.piece_map().items():
        if p.piece_type == chess.KING:
          continue
        indices[halfkp_idx(turn, orient(turn, board.king(turn), layout), sq, p.piece_type, p.color, layout)] = 1.0
      return indices
    return (piece_features(chess.WHITE), piece_features(chess.BLACK))

  def get_initial_psqt_features(self):
    layout = self.layout
    raise Exception('Not supported yet. See HalfKA')

class FactorizedFeatures(FeatureBlock):
  def __init__(self, config=None):
    layout = self.layout = Layout(config or variant.default_config())
    super(FactorizedFeatures, self).__init__('HalfKP^', 0x5d69d5b8, OrderedDict([('HalfKP', layout.NUM_PLANES * layout.NUM_SQ), ('HalfK', layout.NUM_SQ), ('P', layout.NUM_SQ * layout.NUM_PT )]))
    self.base = Features(layout.config)

  def get_active_features(self, board: chess.Board):
    layout = self.layout
    white, black = self.base.get_active_features(board)
    def piece_features(base, color):
      indices = torch.zeros(layout.NUM_SQ * (layout.NUM_PT + 1))
      piece_count = 0
      # P feature
      for sq, p in board.piece_map().items():
        if p.piece_type == chess.KING:
          continue
        piece_count += 1
        p_idx = (p.piece_type - 1) * 2 + (p.color != color)
        indices[(p_idx + 1) * layout.NUM_SQ + orient(color, sq, layout)] = 1.0
      # HalfK feature
      indices[orient(color, board.king(color), layout)] = piece_count
      return torch.cat((base, indices))
    return (piece_features(white, chess.WHITE), piece_features(black, chess.BLACK))

  def get_feature_factors(self, idx):
    layout = self.layout
    if idx >= self.num_real_features:
      raise Exception('Feature must be real')

    k_idx = idx // layout.NUM_PLANES
    p_idx = idx % layout.NUM_PLANES - 1

    return [idx, self.get_factor_base_feature('HalfK') + k_idx, self.get_factor_base_feature('P') + p_idx]

  def get_initial_psqt_features(self):
    layout = self.layout
    raise Exception('Not supported yet. See HalfKA^')

'''
This is used by the features module for discovery of feature blocks.
'''
def get_feature_block_clss():
  return [Features, FactorizedFeatures]
