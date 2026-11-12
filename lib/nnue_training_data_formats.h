/*

Copyright 2020 Tomasz Sobczyk

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software
and associated documentation files (the "Software"),
to deal in the Software without restriction,
including without limitation the rights to use, copy,
modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall
be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

*/

// SPDX-License-Identifier: MIT
// Runtime-sized value-training records. Geometry and type IDs are stream-local.
#pragma once
#include "variant_config.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace chess {
enum class Color : std::uint8_t { White, Black };
using Square = unsigned;
using Piece = std::uint16_t; // interleaved dense type/color, not a wire encoding
constexpr Piece NoPiece = 65535;
inline unsigned type_of(Piece pc) { return pc >> 1; }
inline Color color_of(Piece pc) { return Color(pc & 1); }
inline Piece make_piece(unsigned type, Color color) { return Piece(2 * type + unsigned(color)); }
struct Position {
    const training_data::VariantConfig *config = nullptr;
    Piece *board = nullptr;
    std::uint16_t *hands = nullptr;
    std::array<unsigned, 2> kings{};
    unsigned count = 0;
    Color turn = Color::White;
    void clear() {
        std::fill_n(board, config->squares, NoPiece);
        std::fill_n(hands, config->colored_types, 0);
        kings = {config->squares, config->squares};
        count = 0;
        turn = Color::White;
    }
    void place(Piece pc, unsigned sq) {
        if (sq >= config->squares)
            throw std::runtime_error("Piece square outside variant board");
        if (board[sq] != NoPiece)
            throw std::runtime_error("Overlapping training pieces");
        board[sq] = pc;
        ++count;
        if (type_of(pc) == config->king_type)
            kings[unsigned(color_of(pc))] = sq;
    }
    Piece pieceAt(unsigned sq) const { return board[sq]; }
    unsigned kingSquare(Color c) const { return kings[unsigned(c)]; }
    unsigned pieceCount() const { return count; }
    Color sideToMove() const { return turn; }
    unsigned getHandCount(Piece pc) const { return hands[pc]; }
};
} // namespace chess
namespace bin {
struct TrainingDataEntry {
    chess::Position pos;
    std::uint16_t move = 0, ply = 1;
    std::int16_t score = 0, result = 0;
};
// Each feature worker owns one slab. Record views never allocate or own buffers.
class BatchStorage {
    training_data::VariantPtr config;
    std::vector<std::uint16_t> data;

  public:
    BatchStorage(training_data::VariantPtr cfg, unsigned count) : config(std::move(cfg)) {
        if (config)
            data.resize(std::size_t(count) * config->row_size);
    }
    TrainingDataEntry entry(unsigned i) {
        TrainingDataEntry e;
        if (config) {
            e.pos.config = config.get();
            e.pos.board = data.data() + std::size_t(i) * config->row_size;
            e.pos.hands = e.pos.board + config->squares;
        }
        return e;
    }
};
struct BitReader {
    const std::uint8_t *data;
    unsigned at = 0, limit = 512;
    unsigned get(unsigned n) {
        if (at + n > limit)
            throw std::runtime_error("Truncated PackedSfen position");
        unsigned value = 0;
        for (unsigned i = 0; i < n; ++i, ++at)
            value |= unsigned((data[at / 8] >> (at % 8)) & 1) << i;
        return value;
    }
};
inline void validate_entry(const TrainingDataEntry &e) {
    const auto &c = *e.pos.config;
    unsigned total = e.pos.count;
    for (unsigned pc = 0; pc < c.colored_types; ++pc) {
        unsigned n = e.pos.hands[pc];
        if (n && (!c.USE_POCKETS || n > c.hand_capacity || (pc >> 1) >= c.king_type))
            throw std::runtime_error("Pocket count exceeds configured feature capacity");
        total += n;
    }
    if (total > c.PIECE_COUNT)
        throw std::runtime_error("Record exceeds configured PIECE_COUNT");
    if (c.KING_SQUARES > 1 && (e.pos.kings[0] >= c.squares || e.pos.kings[1] >= c.squares))
        throw std::runtime_error("Record requires two NNUE Kings");
    if (e.result < -1 || e.result > 1)
        throw std::runtime_error("Invalid training outcome");
}
inline void decode_packed72(const std::uint8_t *bytes, TrainingDataEntry &e) {
    const auto &c = *e.pos.config;
    BitReader in{bytes};
    e.pos.clear();
    e.pos.turn = chess::Color(in.get(1));
    for (unsigned color = 0; color < 2; ++color) {
        unsigned sq = in.get(7);
        if (c.KING_SQUARES == 1) {
            if (sq != c.squares)
                throw std::runtime_error("Invalid absent NNUE King square");
        } else
            e.pos.place(chess::make_piece(c.king_type, chess::Color(color)), sq);
    }
    for (int rank = int(c.RANKS) - 1; rank >= 0; --rank)
        for (unsigned file = 0; file < c.FILES; ++file) {
            unsigned sq = unsigned(rank) * c.FILES + file;
            if (e.pos.board[sq] != chess::NoPiece)
                continue;
            if (in.get(1)) {
                unsigned type = in.get(4), color = in.get(1);
                if (type >= c.PIECE_TYPES || type == c.king_type)
                    throw std::runtime_error("Invalid PackedSfen piece type");
                e.pos.place(chess::make_piece(type, chess::Color(color)), sq);
            }
        }
    for (unsigned color = 0; color < 2; ++color)
        for (unsigned type = 0; type < c.PIECE_TYPES; ++type)
            e.pos.hands[chess::make_piece(type, chess::Color(color))] = std::uint16_t(in.get(5));
    in.get(4);
    if (in.get(1))
        in.get(7); // rights and EP are not training features
    in.get(6);
    in.get(16);
    in.get(1); // clocks
    unsigned score = unsigned(bytes[64]) | (unsigned(bytes[65]) << 8);
    e.score = std::int16_t(score < 32768 ? int(score) : int(score) - 65536);
    e.move = std::uint16_t(unsigned(bytes[66]) | (unsigned(bytes[67]) << 8));
    e.ply = std::uint16_t(unsigned(bytes[68]) | (unsigned(bytes[69]) << 8));
    e.result = bytes[70] < 128 ? bytes[70] : int(bytes[70]) - 256;
    validate_entry(e);
}
} // namespace bin
