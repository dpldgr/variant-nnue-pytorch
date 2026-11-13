#include "../lib/dtx_training_record.h"
#include <iostream>
#include <stdexcept>

static void check(bool x, const char *message) {
    if (!x)
        throw std::runtime_error(message);
}
int main() {
    try {
        using namespace DtxTrainingRecord;
        Header h;
        h.variant = "metal-chess";
        h.symbols = {"K", "P", "Q", "R", "B", "N"};
        h.chooseWidths();
        check(h.bits() == 44 && h.recordBytes == 6, "default bit layout");
        h.records = UINT64_C(7000000000);
        std::string error;
        std::vector<std::uint8_t> bytes;
        check(h.encode(bytes, error), "header encode");
        Header parsed;
        check(Header::decode(bytes.data(), bytes.size(), parsed, error), "header decode");
        check(parsed.records == h.records && parsed.symbols == h.symbols, "header metadata");
        Codec codec;
        check(codec.reset(h, error), "codec init");
        Entry e, decoded;
        e.kings = {0, 63};
        e.pieces = {{1, 0, 0}, {32, 4, 1}};
        std::vector<std::uint8_t> raw(h.recordBytes);
        for (int d = h.draw(); d <= h.maxDistance(); ++d) {
            e.dtm = d;
            e.turn = unsigned(d) & 1;
            check(codec.encode(e, raw.data(), error) && codec.decode(raw.data(), decoded, error),
                  "label roundtrip");
            check(decoded.dtm == d && decoded.turn == e.turn && decoded.pieces[0].type == 0,
                  "distance/pawn decode");
            int outcome = d == h.draw() ? 0 : d > 0 ? 1 : -1;
            check(decoded.result == outcome && decoded.score == outcome * (32000 - std::abs(d)),
                  "label lookup");
        }
        e.dtm = 1024;
        check(!codec.encode(e, raw.data(), error), "out of range distance accepted");
        e.dtm = 0;
        check(codec.encode(e, raw.data(), error), "terminal encode");
        raw.back() |= 128;
        check(!codec.decode(raw.data(), decoded, error), "padding accepted");
        e.pieces[1].square = 0;
        check(!codec.encode(e, raw.data(), error), "overlap accepted");
        Codec empty;
        check(!empty.decode(raw.data(), decoded, error), "uninitialized codec accepted");
        for (unsigned size : {2, 3, 4, 5, 6}) {
            Header x = h;
            x.maxPieces = size;
            x.chooseWidths();
            check(x.bits() == 24 + 10 * (size - 2), "capacity layout");
        }
        for (unsigned b : {1, 7, 11, 15}) {
            Header x = h;
            x.dtmBits = b;
            x.chooseWidths();
            Codec c;
            check(c.reset(x, error), "variable widths");
            Entry v;
            v.kings = {0, 63};
            v.dtm = x.draw();
            raw.resize(x.recordBytes);
            check(c.encode(v, raw.data(), error) && c.decode(raw.data(), decoded, error) && !decoded.score &&
                      !decoded.result,
                  "draw sentinel");
        }
        Header wide = h;
        wide.files = wide.ranks = 16;
        wide.symbols = {"K"};
        for (unsigned i = 0; i < 301; ++i) wide.symbols.push_back("p" + std::to_string(i));
        wide.chooseWidths();
        Codec broad;
        check(wide.squareBits == 8 && wide.typeBits == 9 && broad.reset(wide, error), "wide schema");
        Entry large;
        large.kings = {0, 255};
        large.pieces = {{129, 0, 0}, {254, 300, 1}};
        large.dtm = -1023;
        raw.resize(wide.recordBytes);
        check(broad.encode(large, raw.data(), error) && broad.decode(raw.data(), decoded, error), "wide roundtrip");
        check(decoded.kings[1] == 255 && decoded.pieces[1].type == 300 && decoded.dtm == -1023, "wide values");
        bytes[56] = 1;
        check(!Header::decode(bytes.data(), bytes.size(), parsed, error), "reserved header accepted");
        std::cout << "DTX header, all 2048 labels, empty/Pawn slots, widths and malformed records pass\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
