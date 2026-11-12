#pragma once

#include "nnue_training_data_formats.h"
#include "compact_training_record.h"
#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace training_data {

// Storage backends resolve logical record indices. Future variable-size or
// compressed formats can provide an offset/block index behind this interface.
class IndexedTrainingDataSource {
public:
    using Index = std::uint64_t;
    virtual Index record_count() const = 0;
    virtual bin::TrainingDataEntry read_record(Index index) = 0;
    virtual ~IndexedTrainingDataSource() = default;
};

// Fixed-width layout assumptions belong only to this backend.
class BinIndexedTrainingDataSource : public IndexedTrainingDataSource {
public:
    explicit BinIndexedTrainingDataSource(const std::string& filename)
        : BinIndexedTrainingDataSource(filename, sizeof(bin::nodchip::PackedSfenValue), true) {}

    BinIndexedTrainingDataSource(const std::string& filename, unsigned record_bytes, bool compiled_packed = false)
        : m_file(filename, std::ios::binary | std::ios::ate), m_bytes(record_bytes),
          m_legacy(compiled_packed || record_bytes == 72) {
        if (record_bytes < 7 || record_bytes > 144)
            throw std::invalid_argument("record bytes must be 7..144");
        if (m_legacy && record_bytes != sizeof(bin::nodchip::PackedSfenValue))
            throw std::invalid_argument("72-byte PackedSfen requires the default DATA_SIZE=512 loader");
        if (!m_legacy) {
            std::string error;
            if (KING_SQUARES == 1 || !m_codec.reset(unsigned(chess::Square::NB), unsigned(chess::PieceType::King), record_bytes, error))
                throw std::runtime_error(error.empty() ? "Compact records require two NNUE Kings" : error);
        }
        if (!m_file) throw std::runtime_error("Cannot open training data: " + filename);
        auto size = m_file.tellg();
        if (size < 0) throw std::runtime_error("Cannot determine training data size: " + filename);
        auto bytes = static_cast<Index>(size);
        if (!bytes) throw std::runtime_error("Training dataset is empty: " + filename);
        if (bytes % m_bytes)
            throw std::runtime_error("Incomplete .bin record (check --record-bytes and the loader schema): " + filename);
        m_count = bytes / m_bytes;
    }

    Index record_count() const override { return m_count; }

    bin::TrainingDataEntry read_record(Index index) override {
        if (index >= m_count) throw std::out_of_range("Record index outside training dataset");
        if (index != m_next_index)
            m_file.seekg(static_cast<std::streamoff>(index * m_bytes));
        else if (index == 0)
            m_file.seekg(0);
        if (m_legacy) {
            bin::nodchip::PackedSfenValue packed;
            if (!m_file.read(reinterpret_cast<char*>(&packed), sizeof(packed)))
                throw std::runtime_error("Failed to read indexed training record");
            m_next_index = index + 1;
            return bin::packedSfenValueToTrainingDataEntry(packed);
        }
        if (!m_file.read(reinterpret_cast<char*>(m_buffer.data()), m_bytes))
            throw std::runtime_error("Failed to read indexed training record");
        m_next_index = index + 1;
        std::string error;
        if (!m_codec.decode(m_buffer.data(), m_entry, error))
            throw std::runtime_error("Invalid compact record: " + error);
        if (m_entry.pieces.size() + 2 > PIECE_COUNT)
            throw std::runtime_error("Compact record exceeds the compiled PIECE_COUNT");
        bin::TrainingDataEntry entry{};
        entry.pos.setSideToMove(static_cast<chess::Color>(m_entry.turn));
        for (unsigned color = 0; color < 2; ++color)
            entry.pos.place(chess::make_piece(chess::PieceType::King, static_cast<chess::Color>(color)),
                            static_cast<chess::Square>(m_entry.kings[color]));
        for (const auto& piece : m_entry.pieces)
            entry.pos.place(chess::make_piece(static_cast<chess::PieceType>(piece.type), static_cast<chess::Color>(piece.color)),
                            static_cast<chess::Square>(piece.square));
        entry.pos.setCastlingRights(chess::CastlingRights::None);
        entry.pos.setFullMove(1);
        entry.score = m_entry.score; entry.result = m_entry.result; entry.ply = 1;
        return entry;
    }

private:
    std::ifstream m_file;
    Index m_count, m_next_index = 0;
    unsigned m_bytes;
    bool m_legacy;
    std::array<std::uint8_t, 144> m_buffer;
    CompactTrainingRecord::Codec m_codec;
    CompactTrainingRecord::Entry m_entry;
};

inline std::unique_ptr<IndexedTrainingDataSource> open_indexed_training_data_source(const std::string& filename, unsigned record_bytes) {
    if (filename.size() >= 4 && filename.compare(filename.size() - 4, 4, ".bin") == 0)
        return std::make_unique<BinIndexedTrainingDataSource>(filename, record_bytes);
    throw std::invalid_argument("No indexed training-data backend for: " + filename);
}

// Existing callers retain the record layout compiled into the legacy library.
inline std::unique_ptr<IndexedTrainingDataSource> open_indexed_training_data_source(const std::string& filename) {
    if (filename.size() >= 4 && filename.compare(filename.size() - 4, 4, ".bin") == 0)
        return std::make_unique<BinIndexedTrainingDataSource>(filename);
    throw std::invalid_argument("No indexed training-data backend for: " + filename);
}

} // namespace training_data
