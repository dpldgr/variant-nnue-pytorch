// SPDX-License-Identifier: MIT
// DTX v1: header-described, fixed-stride distance training records.
// Shared verbatim by the exporter and generic Trainer. No engine dependencies.
#ifndef DTX_TRAINING_RECORD_H
#define DTX_TRAINING_RECORD_H
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

namespace DtxTrainingRecord {
constexpr unsigned PrefixBytes = 64, CountOffset = 40, MaxHeaderBytes = 1048576;
constexpr std::uint8_t Mate = 1, Capture = 2, Zeroing = 3;
constexpr char Magic[8] = {'D', 'T', 'X', 'T', 'R', 'N', '\r', '\n'};
inline unsigned width(unsigned values) {
    unsigned n = 0;
    for (unsigned v = values ? values - 1 : 0; v; v >>= 1)
        ++n;
    return n;
}
inline bool fail(std::string &error, const char *text) {
    error = text;
    return false;
}
inline std::uint64_t get(const std::uint8_t *p, unsigned bytes) {
    std::uint64_t value = 0;
    for (unsigned i = 0; i < bytes; ++i)
        value |= std::uint64_t(p[i]) << (8 * i);
    return value;
}
inline void put(std::uint8_t *p, std::uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i)
        p[i] = std::uint8_t(value >> (8 * i));
}
struct Header {
    unsigned files = 8, ranks = 8, maxPieces = 4;
    unsigned squareBits = 6, typeBits = 3, dtmBits = 11, recordBytes = 6;
    std::uint8_t metric = Mate;
    std::uint64_t records = 0;
    std::string variant;
    // King first, followed by dense non-King types; wire code zero means empty.
    std::vector<std::string> symbols;
    unsigned cells() const {
        return files * ranks;
    }
    unsigned types() const {
        return symbols.empty() ? 0 : unsigned(symbols.size() - 1);
    }
    unsigned slots() const {
        return maxPieces - 2;
    }
    unsigned bits() const {
        return 2 * squareBits + slots() * (squareBits + typeBits + 1) + 1 + dtmBits;
    }
    int draw() const {
        return -(1 << (dtmBits - 1));
    }
    int maxDistance() const {
        return (1 << (dtmBits - 1)) - 1;
    }
    void chooseWidths() {
        squareBits = std::max(1u, width(cells()));
        typeBits = std::max(1u, width(types() + 1));
        recordBytes = (bits() + 7) / 8;
    }
    bool validate(std::string &error) const {
        if (!files || files > 65535 || !ranks || ranks > 65535 || std::uint64_t(files) * ranks > 65535)
            return fail(error, "invalid DTX board dimensions");
        if (maxPieces < 2 || maxPieces > 65535 || maxPieces > cells())
            return fail(error, "invalid DTX piece capacity");
        if (symbols.empty() || symbols.size() > 65535 || variant.empty() || variant.size() > 65535 ||
            variant.find('\0') != std::string::npos)
            return fail(error, "invalid DTX variant or piece catalogue");
        std::unordered_set<std::string> seen;
        for (unsigned i = 0; i < symbols.size(); ++i) {
            if (symbols[i].empty() || symbols[i].size() > 65535 || symbols[i].find('\0') != std::string::npos)
                return fail(error, "invalid DTX piece symbol");
            if (!seen.insert(symbols[i]).second)
                return fail(error, "duplicate DTX piece symbol");
        }
        if (squareBits < std::max(1u, width(cells())) || squareBits > 16 ||
            typeBits < std::max(1u, width(types() + 1)) || typeBits > 16 || dtmBits < 1 || dtmBits > 15)
            return fail(error, "invalid DTX field widths");
        if (recordBytes < (bits() + 7) / 8 || recordBytes > 65535)
            return fail(error, "invalid DTX record stride");
        if (metric < Mate || metric > Zeroing)
            return fail(error, "unknown DTX distance metric");
        return true;
    }
    bool encode(std::vector<std::uint8_t> &bytes, std::string &error) const {
        if (!validate(error))
            return false;
        std::uint64_t catalogue = 0;
        for (const auto &s : symbols)
            catalogue += 2 + s.size();
        std::uint64_t total = (PrefixBytes + variant.size() + catalogue + 7) & ~std::uint64_t(7);
        if (total > MaxHeaderBytes)
            return fail(error, "DTX header exceeds size limit");
        bytes.assign(std::size_t(total), 0);
        std::memcpy(bytes.data(), Magic, 8);
        put(bytes.data() + 8, 1, 2);
        put(bytes.data() + 10, PrefixBytes, 2);
        put(bytes.data() + 12, total, 4);
        put(bytes.data() + 16, recordBytes, 4);
        put(bytes.data() + 20, files, 2);
        put(bytes.data() + 22, ranks, 2);
        put(bytes.data() + 24, types(), 2);
        put(bytes.data() + 26, maxPieces, 2);
        bytes[28] = std::uint8_t(squareBits);
        bytes[29] = std::uint8_t(typeBits);
        bytes[30] = std::uint8_t(dtmBits);
        bytes[31] = metric;
        put(bytes.data() + 32, variant.size(), 4);
        put(bytes.data() + 36, catalogue, 4);
        put(bytes.data() + CountOffset, records, 8);
        bytes[52] = 1;             // Signed two's complement, minimum=draw, zero=terminal loss.
        bytes[54] = bytes[55] = 1; // Colour and side-to-move widths; LSB-first bit order=0.
        unsigned at = PrefixBytes;
        std::memcpy(bytes.data() + at, variant.data(), variant.size());
        at += unsigned(variant.size());
        for (const auto &s : symbols) {
            put(bytes.data() + at, s.size(), 2);
            at += 2;
            std::memcpy(bytes.data() + at, s.data(), s.size());
            at += unsigned(s.size());
        }
        return true;
    }
    static bool prefix(const std::uint8_t *bytes, std::size_t size, unsigned &headerBytes,
                       std::string &error) {
        if (size < PrefixBytes)
            return fail(error, "truncated DTX header");
        if (std::memcmp(bytes, Magic, 8) || get(bytes + 8, 2) != 1 || get(bytes + 10, 2) != PrefixBytes)
            return fail(error, "invalid DTX magic or unsupported version");
        auto n = get(bytes + 12, 4);
        if (n < PrefixBytes || n > MaxHeaderBytes || n % 8)
            return fail(error, "invalid DTX header length");
        headerBytes = unsigned(n);
        return true;
    }
    static bool decode(const std::uint8_t *bytes, std::size_t size, Header &output, std::string &error) {
        unsigned length;
        if (!prefix(bytes, size, length, error))
            return false;
        if (size != length)
            return fail(error, "truncated or oversized DTX header");
        if (get(bytes + 48, 4) || bytes[52] != 1 || bytes[53] || bytes[54] != 1 || bytes[55] != 1 ||
            get(bytes + 56, 8))
            return fail(error, "unsupported DTX flags or field conventions");
        Header h;
        h.files = unsigned(get(bytes + 20, 2));
        h.ranks = unsigned(get(bytes + 22, 2));
        h.recordBytes = unsigned(get(bytes + 16, 4));
        h.maxPieces = unsigned(get(bytes + 26, 2));
        h.squareBits = bytes[28];
        h.typeBits = bytes[29];
        h.dtmBits = bytes[30];
        h.metric = bytes[31];
        h.records = get(bytes + CountOffset, 8);
        auto vn = get(bytes + 32, 4), cn = get(bytes + 36, 4);
        if (vn > length - PrefixBytes || cn > length - PrefixBytes - vn)
            return fail(error, "invalid DTX metadata lengths");
        unsigned at = PrefixBytes;
        h.variant.assign(reinterpret_cast<const char *>(bytes + at), std::size_t(vn));
        at += unsigned(vn);
        unsigned end = at + unsigned(cn), count = unsigned(get(bytes + 24, 2)) + 1;
        for (unsigned i = 0; i < count; ++i) {
            if (end - at < 2)
                return fail(error, "truncated DTX piece catalogue");
            unsigned n = unsigned(get(bytes + at, 2));
            at += 2;
            if (n > end - at)
                return fail(error, "truncated DTX piece symbol");
            h.symbols.emplace_back(reinterpret_cast<const char *>(bytes + at), n);
            at += n;
        }
        if (at != end)
            return fail(error, "extra DTX catalogue data");
        while (at < length)
            if (bytes[at++])
                return fail(error, "nonzero DTX header padding");
        if (!h.validate(error))
            return false;
        output = std::move(h);
        return true;
    }
    bool fileSize(unsigned headerBytes, std::uint64_t &result, std::string &error) const {
        if (!recordBytes)
            return fail(error, "invalid DTX record stride");
        if (records > (std::numeric_limits<std::uint64_t>::max() - headerBytes) / recordBytes)
            return fail(error, "DTX record count overflows file size");
        result = headerBytes + records * recordBytes;
        return true;
    }
};
struct Support {
    std::uint16_t square = 0, type = 0;
    std::uint8_t color = 0;
};
struct Entry {
    std::array<std::uint16_t, 2> kings{};
    std::vector<Support> pieces;
    std::uint8_t turn = 0;
    int dtm = 0;
    // Decoded compatibility labels only; these are NOT stored on disk.
    std::int16_t score = 0;
    std::int8_t result = 0;
};
struct Label {
    std::int16_t score;
    std::int8_t result;
};
class Codec {
    struct Reader {
        const std::uint8_t *data;
        unsigned at = 0;
        unsigned take(unsigned n) {
            unsigned value = 0, shift = 0;
            while (n) {
                unsigned k = std::min(n, 8 - at % 8);
                value |= unsigned((data[at / 8] >> (at % 8)) & ((1u << k) - 1)) << shift;
                at += k;
                shift += k;
                n -= k;
            }
            return value;
        }
    };
    struct Writer {
        std::uint8_t *data;
        unsigned at = 0;
        void write(unsigned value, unsigned n) {
            while (n) {
                unsigned k = std::min(n, 8 - at % 8);
                data[at / 8] |= std::uint8_t((value & ((1u << k) - 1)) << (at % 8));
                value >>= k;
                at += k;
                n -= k;
            }
        }
    };
    Header m_header;
    std::vector<std::uint64_t> m_used;
    std::vector<Label> m_labels;
    bool mark(unsigned sq) {
        if (sq >= m_header.cells() || (m_used[sq / 64] & (std::uint64_t(1) << (sq % 64))))
            return false;
        m_used[sq / 64] |= std::uint64_t(1) << (sq % 64);
        return true;
    }
    bool validate(const Entry &e, std::string &error) {
        if (e.turn > 1 || e.pieces.size() > m_header.slots() || e.dtm < m_header.draw() ||
            e.dtm > m_header.maxDistance())
            return fail(error, "invalid DTX position metadata or distance");
        std::fill(m_used.begin(), m_used.end(), 0);
        if (!mark(e.kings[0]) || !mark(e.kings[1]))
            return fail(error, "invalid or overlapping DTX Kings");
        for (auto p : e.pieces)
            if (p.type >= m_header.types() || p.color > 1 || !mark(p.square))
                return fail(error, "invalid or overlapping DTX piece");
        return true;
    }

  public:
    bool reset(const Header &h, std::string &error) {
        if (!h.validate(error))
            return false;
        m_header = h;
        m_used.assign((h.cells() + 63) / 64, 0);
        m_labels.resize(1u << h.dtmBits);
        // Built once per reader; record decoding only indexes these labels.
        for (unsigned code = 0; code < m_labels.size(); ++code) {
            int d = int(code);
            if (code & (1u << (h.dtmBits - 1)))
                d -= 1 << h.dtmBits;
            int result = d == h.draw() ? 0 : d > 0 ? 1 : -1;
            int distance = d < 0 ? -d : d;
            m_labels[code] = {std::int16_t(result * (32000 - distance)), std::int8_t(result)};
        }
        return true;
    }
    bool encode(const Entry &e, std::uint8_t *bytes, std::string &error) {
        if (m_labels.empty())
            return fail(error, "DTX codec is not initialized");
        if (!validate(e, error))
            return false;
        std::fill(bytes, bytes + m_header.recordBytes, 0);
        Writer out{bytes};
        out.write(e.kings[0], m_header.squareBits);
        out.write(e.kings[1], m_header.squareBits);
        for (unsigned i = 0; i < m_header.slots(); ++i) {
            bool present = i < e.pieces.size();
            out.write(present ? e.pieces[i].square : 0, m_header.squareBits);
            out.write(present ? e.pieces[i].type + 1 : 0, m_header.typeBits);
            out.write(present ? e.pieces[i].color : 0, 1);
        }
        out.write(e.turn, 1);
        out.write(unsigned(e.dtm) & ((1u << m_header.dtmBits) - 1), m_header.dtmBits);
        return true;
    }
    bool decode(const std::uint8_t *bytes, Entry &e, std::string &error) {
        if (m_labels.empty())
            return fail(error, "DTX codec is not initialized");
        Reader in{bytes};
        e.pieces.clear();
        e.kings = {std::uint16_t(in.take(m_header.squareBits)), std::uint16_t(in.take(m_header.squareBits))};
        bool empty = false;
        for (unsigned i = 0; i < m_header.slots(); ++i) {
            auto sq = in.take(m_header.squareBits), type = in.take(m_header.typeBits), color = in.take(1);
            if (!type) {
                if (sq || color)
                    return fail(error, "nonzero empty DTX slot");
                empty = true;
            } else {
                if (empty)
                    return fail(error, "noncanonical DTX slots");
                e.pieces.push_back({std::uint16_t(sq), std::uint16_t(type - 1), std::uint8_t(color)});
            }
        }
        e.turn = std::uint8_t(in.take(1));
        unsigned code = in.take(m_header.dtmBits);
        e.dtm = int(code);
        if (code & (1u << (m_header.dtmBits - 1)))
            e.dtm -= 1 << m_header.dtmBits;
        while (in.at < m_header.recordBytes * 8) {
            if (in.take(std::min(8u, m_header.recordBytes * 8 - in.at)))
                return fail(error, "nonzero DTX record padding");
        }
        if (!validate(e, error))
            return false;
        e.score = m_labels[code].score;
        e.result = m_labels[code].result;
        return true;
    }
};
} // namespace DtxTrainingRecord
#endif
