#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace training_data {
// Immutable numeric counterpart of the agreed JSON fields. No JSON in hot loops.
struct VariantConfig {
    unsigned FILES, RANKS, PIECE_TYPES, PIECE_COUNT, KING_SQUARES;
    bool USE_POCKETS;
    unsigned squares, king_type, hand_capacity, colored_types, row_size;
    std::vector<unsigned> identity, flip, rotate, king_map;

    VariantConfig(unsigned files = 8, unsigned ranks = 8, unsigned types = 6, unsigned pieces = 32,
                  unsigned kings = 64, unsigned pockets = 0)
        : FILES(files), RANKS(ranks), PIECE_TYPES(types), PIECE_COUNT(pieces), KING_SQUARES(kings),
          USE_POCKETS(pockets != 0) {
        if (!files || !ranks || std::uint64_t(files) * ranks > 65535 || !types || types > 32766 || !pieces ||
            pieces > 65535 || !kings || kings > std::uint64_t(files) * ranks || pockets > 1)
            throw std::invalid_argument("Invalid variant dimensions, PIECE_COUNT or KING_SQUARES");
        squares = files * ranks;
        king_type = types - unsigned(kings > 1);
        hand_capacity = pockets ? 2 * files : 0;
        colored_types = 2 * (king_type + 1);
        row_size = squares + colored_types;
        identity.resize(squares + 1);
        flip.resize(squares + 1);
        rotate.resize(squares + 1);
        king_map.resize(squares + 1);
        for (unsigned sq = 0; sq < squares; ++sq) {
            identity[sq] = sq;
            flip[sq] = (ranks - 1 - sq / files) * files + sq % files;
            rotate[sq] = squares - 1 - sq;
            int mapped = (kings == 9 && kings != squares) ? (int(sq) - 6 * int(sq / files) - 3) % int(kings)
                                                          : int(sq % kings);
            king_map[sq] = mapped < 0 ? unsigned(kings) : unsigned(mapped);
        }
        // Existing no-reference-piece encoding uses the board end sentinel.
        identity[squares] = flip[squares] = rotate[squares] = king_map[squares] = 0;
    }
};
using VariantPtr = std::shared_ptr<const VariantConfig>;
inline VariantPtr default_variant() {
    static auto value = std::make_shared<const VariantConfig>();
    return value;
}
} // namespace training_data
