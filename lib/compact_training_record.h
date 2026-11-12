// SPDX-License-Identifier: MIT
// Standalone adaptive training-record codec. No engine or trainer dependencies.
// Wire schema: geometry, dense non-King type order and record width are external.
#ifndef COMPACT_TRAINING_RECORD_H
#define COMPACT_TRAINING_RECORD_H
#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace CompactTrainingRecord {
constexpr unsigned MinBytes = 7, MaxBytes = 144;
struct Support { std::uint16_t square, type; std::uint8_t color; };
struct Entry {
    std::array<std::uint16_t, 2> kings{};
    std::vector<Support> pieces;
    std::uint8_t turn = 0;
    std::int16_t score = 0;
    std::int8_t result = 0;
};

class Codec {
    static unsigned width(unsigned values) {
        unsigned bits = 0;
        for (unsigned x = values ? values - 1 : 0; x; x >>= 1) ++bits;
        return bits;
    }
    static bool fail(std::string& error, const char* text) { error = text; return false; }
    struct Reader {
        const std::uint8_t* data;
        unsigned at = 0, limit;
        bool ok = true;
        std::uint32_t get(unsigned bits) {
            if (bits > 32 || at + bits > limit) { ok = false; return 0; }
            std::uint32_t value = 0;
            for (unsigned i = 0; i < bits; ++i, ++at)
                value |= std::uint32_t((data[at / 8] >> (at % 8)) & 1) << i;
            return value;
        }
        bool padding() {
            while (ok && at < limit) if (get(1)) return false;
            return ok;
        }
    };
    struct Writer {
        std::uint8_t* data;
        unsigned at = 0;
        void put(std::uint32_t value, unsigned bits) {
            for (unsigned i = 0; i < bits; ++i, ++at)
                data[at / 8] |= std::uint8_t(((value >> i) & 1) << (at % 8));
        }
    };
public:
    bool reset(unsigned cells, unsigned types, unsigned bytes, std::string& error) {
        m_bytes = 0;
        if (bytes < MinBytes || bytes > MaxBytes) return fail(error, "record bytes must be 7..144");
        if (cells < 2 || cells > 65535 || types > 65534)
            return fail(error, "compact schema exceeds coordinate or type capacity");
        m_cells = cells; m_types = types;
        // Minimum widths preserve the existing 8-byte sparse wire format.
        m_square = std::max(6u, width(cells));
        m_type = std::max(3u, width(types));
        m_bitmap_type = width(types + 1); // King is dense code `types`.
        m_capacity = 0;
        if (2 * m_square + 20 > bytes * 8)
            return fail(error, "record cannot fit the compact schema");
        for (unsigned n = 1; n <= bytes * 8; ++n) {
            unsigned bits = 2 * m_square + n * (m_square + m_type + 1)
                          + width(n + 1) + 20;
            if (bits > bytes * 8) break;
            m_capacity = n;
        }
        m_count = width(m_capacity + 1);
        m_used.resize((cells + 63) / 64);
        m_codes.resize(cells);
        m_bytes = bytes;
        return true;
    }
    unsigned record_bytes() const { return m_bytes; }
    unsigned sparse_capacity() const { return m_capacity; }
    bool can_fit(unsigned supports) const {
        return m_bytes && supports <= m_cells - 2 &&
               (supports <= m_capacity || m_cells + (supports + 2) * (m_bitmap_type + 1) + 20 <= m_bytes * 8);
    }
    bool encode(const Entry& entry, std::uint8_t* output, std::string& error) {
        if (!can_fit(unsigned(entry.pieces.size()))) return fail(error, "position does not fit selected record bytes");
        if (!validate(entry, error)) return false;
        std::fill(output, output + m_bytes, 0);
        Writer out{output};
        bool bitmap = entry.pieces.size() > m_capacity;
        if (bitmap) {
            output[m_bytes - 1] |= 128;
            for (unsigned sq = 0; sq < m_cells; ++sq) out.put(used(sq), 1);
            for (unsigned sq = 0; sq < m_cells; ++sq) if (used(sq)) {
                out.put(m_codes[sq] >> 1, m_bitmap_type);
                out.put(m_codes[sq] & 1, 1);
            }
        } else {
            out.put(entry.kings[0], m_square); out.put(entry.kings[1], m_square);
            for (unsigned i = 0; i < m_capacity; ++i) {
                bool present = i < entry.pieces.size();
                out.put(present ? entry.pieces[i].square : 0, m_square);
                out.put(present ? entry.pieces[i].type : 0, m_type);
                out.put(present ? entry.pieces[i].color : 0, 1);
            }
        }
        out.put(entry.turn, 1);
        if (!bitmap) out.put(unsigned(entry.pieces.size()), m_count);
        out.put(std::uint16_t(entry.score), 16);
        out.put(unsigned(entry.result + 1), 2);
        return true;
    }
    bool decode(const std::uint8_t* input, Entry& entry, std::string& error) {
        if (!m_bytes) return fail(error, "compact codec has no valid schema");
        Reader in{input, 0, m_bytes * 8 - 1};
        bool bitmap = (input[m_bytes - 1] & 128) != 0;
        entry.pieces.clear();
        if (bitmap) {
            if (m_cells + 19 > in.limit) return fail(error, "bitmap does not fit record bytes");
            std::fill(m_used.begin(), m_used.end(), 0);
            unsigned occupied = 0;
            for (unsigned sq = 0; sq < m_cells; ++sq) if (in.get(1)) {
                m_used[sq / 64] |= std::uint64_t(1) << (sq % 64); ++occupied;
            }
            if (occupied < 2 || m_cells + occupied * (m_bitmap_type + 1) + 19 > in.limit)
                return fail(error, "bitmap piece count does not fit record bytes");
            bool kings[2] = {false, false};
            for (unsigned sq = 0; sq < m_cells; ++sq) if (used(sq)) {
                unsigned type = in.get(m_bitmap_type), color = in.get(1);
                if (type > m_types) return fail(error, "invalid compact piece type");
                if (type == m_types) {
                    if (kings[color]) return fail(error, "duplicate compact King");
                    kings[color] = true; entry.kings[color] = std::uint16_t(sq);
                } else entry.pieces.push_back({std::uint16_t(sq), std::uint16_t(type), std::uint8_t(color)});
            }
            if (!kings[0] || !kings[1]) return fail(error, "compact record requires two Kings");
        } else {
            entry.kings = {std::uint16_t(in.get(m_square)), std::uint16_t(in.get(m_square))};
            entry.pieces.resize(m_capacity);
            for (auto& p : entry.pieces) {
                p.square = std::uint16_t(in.get(m_square));
                p.type = std::uint16_t(in.get(m_type)); p.color = std::uint8_t(in.get(1));
            }
        }
        entry.turn = std::uint8_t(in.get(1));
        if (!bitmap) {
            unsigned count = in.get(m_count);
            if (count > m_capacity) return fail(error, "invalid compact support count");
            for (unsigned i = count; i < m_capacity; ++i)
                if (entry.pieces[i].square || entry.pieces[i].type || entry.pieces[i].color)
                    return fail(error, "nonzero unused compact piece");
            entry.pieces.resize(count);
        }
        unsigned score = in.get(16), result = in.get(2);
        entry.score = std::int16_t(score < 32768 ? int(score) : int(score) - 65536);
        if (result == 3) return fail(error, "invalid compact result");
        entry.result = std::int8_t(int(result) - 1);
        if (!in.padding()) return fail(error, "nonzero compact padding or truncated record");
        return validate(entry, error);
    }
private:
    bool used(unsigned sq) const { return (m_used[sq / 64] >> (sq % 64)) & 1; }
    bool mark(unsigned sq, unsigned type, unsigned color) {
        if (sq >= m_cells || used(sq)) return false;
        m_used[sq / 64] |= std::uint64_t(1) << (sq % 64);
        m_codes[sq] = (type << 1) | color;
        return true;
    }
    bool validate(const Entry& e, std::string& error) {
        if (e.turn > 1 || e.result < -1 || e.result > 1 || e.pieces.size() > m_cells - 2)
            return fail(error, "invalid compact metadata");
        std::fill(m_used.begin(), m_used.end(), 0);
        if (!mark(e.kings[0], m_types, 0) || !mark(e.kings[1], m_types, 1))
            return fail(error, "invalid or overlapping compact King");
        for (const auto& p : e.pieces)
            if (p.type >= m_types || p.color > 1 || !mark(p.square, p.type, p.color))
                return fail(error, "invalid or overlapping compact piece");
        return true;
    }
    unsigned m_cells = 0, m_types = 0, m_bytes = 0;
    unsigned m_square = 0, m_type = 0, m_bitmap_type = 0, m_capacity = 0, m_count = 0;
    std::vector<std::uint64_t> m_used;
    std::vector<std::uint32_t> m_codes;
};
} // namespace CompactTrainingRecord
#endif
