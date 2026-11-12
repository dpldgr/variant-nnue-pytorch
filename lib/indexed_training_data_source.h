#pragma once
#include "nnue_training_data_formats.h"
#include "compact_training_record.h"
#include <array>
#include <fstream>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace training_data {
class IndexedTrainingDataSource {
  public:
    using Index = std::uint64_t;
    virtual Index record_count() const = 0;
    virtual VariantPtr config() const { return nullptr; }
    // Convenience view is valid until the next read. Batch readers supply
    // their own stable row storage through read_into().
    virtual bin::TrainingDataEntry read_record(Index index) = 0;
    virtual void read_into(Index index, bin::TrainingDataEntry &entry) { entry = read_record(index); }
    virtual ~IndexedTrainingDataSource() = default;
};
class BinIndexedTrainingDataSource final : public IndexedTrainingDataSource {
  public:
    BinIndexedTrainingDataSource(const std::string &filename, unsigned bytes = 72,
                                 VariantPtr cfg = default_variant())
        : m_file(std::filesystem::u8path(filename), std::ios::binary | std::ios::ate),
          m_config(std::move(cfg)), m_bytes(bytes), m_single(m_config, 1) {
        if (bytes < 7 || bytes > 144)
            throw std::invalid_argument("record bytes must be 7..144");
        std::string error;
        if (bytes == 72) {
            if (m_config->squares > 128 || (m_config->squares == 128 && m_config->KING_SQUARES == 1) ||
                m_config->PIECE_TYPES - unsigned(m_config->KING_SQUARES > 1) > 16)
                throw std::invalid_argument("72-byte PackedSfen schema exceeds coordinate/type capacity");
        } else if (m_config->KING_SQUARES == 1 ||
                   !m_codec.reset(m_config->squares, m_config->king_type, bytes, error))
            throw std::runtime_error(error.empty() ? "Compact records require two NNUE Kings" : error);
        if (!m_file)
            throw std::runtime_error("Cannot open training data: " + filename);
        auto size = m_file.tellg();
        if (size < 0)
            throw std::runtime_error("Cannot determine training data size: " + filename);
        auto length = static_cast<Index>(size);
        if (!length)
            throw std::runtime_error("Training dataset is empty: " + filename);
        if (length % bytes)
            throw std::runtime_error(
                "Incomplete .bin record (check --record-bytes and variant configuration): " + filename);
        m_count = length / bytes;
    }
    Index record_count() const override { return m_count; }
    VariantPtr config() const override { return m_config; }
    bin::TrainingDataEntry read_record(Index i) override {
        auto entry = m_single.entry(0);
        read_into(i, entry);
        return entry;
    }
    void read_into(Index index, bin::TrainingDataEntry &entry) override {
        if (index >= m_count)
            throw std::out_of_range("Record index outside training dataset");
        if (index != m_next || index == 0)
            m_file.seekg(static_cast<std::streamoff>(index * m_bytes));
        if (!m_file.read(reinterpret_cast<char *>(m_buffer.data()), m_bytes))
            throw std::runtime_error("Failed to read indexed training record");
        m_next = index + 1;
        if (m_bytes == 72) {
            bin::decode_packed72(m_buffer.data(), entry);
            return;
        }
        std::string error;
        if (!m_codec.decode(m_buffer.data(), m_entry, error))
            throw std::runtime_error("Invalid compact record: " + error);
        entry.pos.clear();
        entry.pos.turn = chess::Color(m_entry.turn);
        for (unsigned c = 0; c < 2; ++c)
            entry.pos.place(chess::make_piece(m_config->king_type, chess::Color(c)), m_entry.kings[c]);
        for (const auto &p : m_entry.pieces)
            entry.pos.place(chess::make_piece(p.type, chess::Color(p.color)), p.square);
        entry.score = m_entry.score;
        entry.result = m_entry.result;
        entry.move = 0;
        entry.ply = 1;
        bin::validate_entry(entry);
    }

  private:
    std::ifstream m_file;
    VariantPtr m_config;
    unsigned m_bytes;
    Index m_count = 0, m_next = 0;
    bin::BatchStorage m_single;
    std::array<std::uint8_t, 144> m_buffer;
    CompactTrainingRecord::Codec m_codec;
    CompactTrainingRecord::Entry m_entry;
};
inline std::unique_ptr<IndexedTrainingDataSource>
open_indexed_training_data_source(const std::string &filename, unsigned bytes = 72,
                                  VariantPtr cfg = default_variant()) {
    if (filename.size() >= 4 && filename.compare(filename.size() - 4, 4, ".bin") == 0)
        return std::make_unique<BinIndexedTrainingDataSource>(filename, bytes, std::move(cfg));
    throw std::invalid_argument("No indexed training-data backend for: " + filename);
}
} // namespace training_data
