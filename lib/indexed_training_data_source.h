#pragma once
#include "nnue_training_data_formats.h"
#include "compact_training_record.h"
#include <array>
#include <fstream>
#include <filesystem>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>

namespace training_data {
class IndexedTrainingDataSource {
  public:
    using Index = std::uint64_t;
    struct Range {
        Index start, count;
    };
    virtual Index record_count() const = 0;
    virtual VariantPtr config() const { return nullptr; }
    // Convenience view is valid until the next read. Batch readers supply
    // their own stable row storage through read_into().
    virtual bin::TrainingDataEntry read_record(Index index) = 0;
    virtual void read_into(Index index, bin::TrainingDataEntry &entry) { entry = read_record(index); }
    // Storage owns the buffered representation. These indices are logical
    // records; permutations and future backends need not know byte offsets.
    virtual Index buffer_ranges(const std::vector<Range> &) {
        throw std::runtime_error("Training-data backend does not support bulk reads");
    }
    virtual void read_buffered_into(Index, bin::TrainingDataEntry &) {
        throw std::runtime_error("Training-data backend does not support buffered records");
    }
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
        decode_into(m_buffer.data(), entry);
    }
    Index buffer_ranges(const std::vector<Range> &ranges) override {
        Index count = 0;
        for (const auto &range : ranges) {
            if (range.start > m_count || range.count > m_count - range.start || range.count > m_count - count)
                throw std::out_of_range("Buffered range outside training dataset");
            count += range.count;
        }
        if (count > std::numeric_limits<std::size_t>::max() / m_bytes ||
            count > Index(std::numeric_limits<std::streamsize>::max()) / m_bytes)
            throw std::length_error("PRP buffer exceeds addressable memory");
        m_records.resize(std::size_t(count) * m_bytes);
        std::size_t offset = 0;
        for (const auto &range : ranges) {
            if (!range.count)
                continue;
            if (range.start != m_next || range.start == 0)
                m_file.seekg(static_cast<std::streamoff>(range.start * m_bytes));
            auto bytes = std::size_t(range.count) * m_bytes;
            if (!m_file.read(reinterpret_cast<char *>(m_records.data() + offset),
                             static_cast<std::streamsize>(bytes)))
                throw std::runtime_error("Failed to bulk-read training records");
            offset += bytes;
            m_next = range.start + range.count;
        }
        m_buffered_count = count;
        return count;
    }
    void read_buffered_into(Index index, bin::TrainingDataEntry &entry) override {
        if (index >= m_buffered_count)
            throw std::out_of_range("Record index outside PRP buffer");
        decode_into(m_records.data() + std::size_t(index) * m_bytes, entry);
    }

  private:
    void decode_into(const std::uint8_t *record, bin::TrainingDataEntry &entry) {
        if (m_bytes == 72) {
            bin::decode_packed72(record, entry);
            return;
        }
        std::string error;
        if (!m_codec.decode(record, m_entry, error))
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
    Index m_count = 0, m_next = 0, m_buffered_count = 0;
    bin::BatchStorage m_single;
    std::array<std::uint8_t, 144> m_buffer;
    std::vector<std::uint8_t> m_records;
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
