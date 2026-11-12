#pragma once

#include <array>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace training_data {

// Index ordering is independent of record size, file layout and decoding.
// Six-round unbalanced Feistel, restricted to [0, count) by cycle walking.
// This mixer is for statistical shuffling, not cryptographic security.
class RecordPermutation {
public:
    using Index = std::uint64_t;

    static Index mix(Index x) {
        x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
        return x ^ (x >> 31);
    }

    RecordPermutation(Index count, Index seed, Index epoch) : m_count(count) {
        if (!count) throw std::invalid_argument("Cannot permute an empty dataset");
        unsigned bits = 0;
        for (Index x = count - 1; x; x >>= 1) ++bits;
        m_left_bits = bits / 2;
        m_right_bits = bits - m_left_bits;
        Index key = mix(seed ^ mix(epoch + UINT64_C(0x9e3779b97f4a7c15)));
        for (unsigned i = 0; i < m_keys.size(); ++i)
            m_keys[i] = mix(key + (i + 1) * UINT64_C(0x9e3779b97f4a7c15));
    }

    Index operator()(Index index) const { return walk(index, false); }
    Index inverse(Index index) const { return walk(index, true); }

private:
    static Index mask(unsigned bits) {
        // Each half has at most 32 bits, including a full 64-bit domain.
        return bits ? (Index(1) << bits) - 1 : 0;
    }

    Index binary(Index x, bool reverse) const {
        Index left = x >> m_right_bits, right = x & mask(m_right_bits);
        unsigned a = m_left_bits, b = m_right_bits;
        for (unsigned i = 0; i < m_keys.size(); ++i) {
            if (reverse) {
                Index previous = right ^ (mix(left ^ m_keys[m_keys.size() - 1 - i]) & mask(b));
                right = left;
                left = previous;
            } else {
                Index next = left ^ (mix(right ^ m_keys[i]) & mask(a));
                left = right;
                right = next;
            }
            std::swap(a, b);
        }
        return (left << b) | right;
    }

    Index walk(Index index, bool reverse) const {
        if (index >= m_count) throw std::out_of_range("Record index outside permutation domain");
        do { index = binary(index, reverse); } while (index >= m_count);
        return index;
    }

    Index m_count;
    unsigned m_left_bits, m_right_bits;
    std::array<Index, 6> m_keys;
};

} // namespace training_data
