#pragma once
#include "nnue_training_data_formats.h"
#include "compact_training_record.h"
#include "dtx_training_record.h"
#include <cctype>
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
    virtual bool legacy_packed() const { return false; }
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
    bool legacy_packed() const override { return m_bytes == 72; }
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
// Header is read and validated once. All random/bulk offsets refer only to
// fixed-width payload records; permutations never see header bytes.
class DtxIndexedTrainingDataSource final : public IndexedTrainingDataSource {
  public:
    DtxIndexedTrainingDataSource(const std::string &filename, VariantPtr cfg)
        : m_file(std::filesystem::u8path(filename), std::ios::binary | std::ios::ate),
          m_config(std::move(cfg)), m_single(m_config, 1) {
        if (!m_file)
            throw std::runtime_error("Cannot open DTX data: " + filename);
        auto end = m_file.tellg();
        if (end < 0)
            throw std::runtime_error("Cannot determine DTX file size");
        std::array<std::uint8_t, DtxTrainingRecord::PrefixBytes> prefix{};
        m_file.seekg(0);
        if (!m_file.read(reinterpret_cast<char *>(prefix.data()), prefix.size()))
            throw std::runtime_error("Truncated DTX header");
        std::string error;
        unsigned length;
        if (!DtxTrainingRecord::Header::prefix(prefix.data(), prefix.size(), length, error))
            throw std::runtime_error(error);
        if (Index(end) < length)
            throw std::runtime_error("Truncated DTX header metadata");
        std::vector<std::uint8_t> bytes(length);
        m_file.seekg(0);
        if (!m_file.read(reinterpret_cast<char *>(bytes.data()), bytes.size()) ||
            !DtxTrainingRecord::Header::decode(bytes.data(), bytes.size(), m_header, error))
            throw std::runtime_error(error.empty() ? "Truncated DTX metadata" : error);
        if (m_header.metric != DtxTrainingRecord::Mate)
            throw std::runtime_error("This Trainer currently supports only mate-distance DTX labels");
        if (m_config->KING_SQUARES == 1 || m_header.files != m_config->FILES ||
            m_header.ranks != m_config->RANKS || m_header.types() != m_config->king_type)
            throw std::runtime_error(
                "DTX header does not match the configured variant dimensions or type count");
        Index expected;
        if (!m_header.fileSize(length, expected, error) || expected != Index(end))
            throw std::runtime_error(error.empty() ? "DTX payload length does not match header record count"
                                                   : error);
        if (!m_header.records)
            throw std::runtime_error("Training dataset is empty: " + filename);
        if (!m_codec.reset(m_header, error))
            throw std::runtime_error(error);
        m_offset = length;
        m_count = m_header.records;
        m_bytes = m_header.recordBytes;
        m_buffer.resize(m_bytes);
        m_entry.pieces.reserve(m_header.slots());
    }
    Index record_count() const override { return m_count; }
    VariantPtr config() const override { return m_config; }
    bin::TrainingDataEntry read_record(Index index) override {
        auto entry = m_single.entry(0);
        read_into(index, entry);
        return entry;
    }
    void read_into(Index index, bin::TrainingDataEntry &entry) override {
        if (index >= m_count)
            throw std::out_of_range("Record index outside DTX dataset");
        if (index != m_next || !index)
            m_file.seekg(std::streamoff(m_offset + index * m_bytes));
        if (!m_file.read(reinterpret_cast<char *>(m_buffer.data()), m_bytes))
            throw std::runtime_error("Failed to read indexed DTX record");
        m_next = index + 1;
        decode_into(m_buffer.data(), entry);
    }
    Index buffer_ranges(const std::vector<Range> &ranges) override {
        Index count = 0;
        for (const auto &r : ranges) {
            if (r.start > m_count || r.count > m_count - r.start || r.count > m_count - count)
                throw std::out_of_range("Buffered range outside DTX dataset");
            count += r.count;
        }
        if (count > std::numeric_limits<std::size_t>::max() / m_bytes ||
            count > Index(std::numeric_limits<std::streamsize>::max()) / m_bytes)
            throw std::length_error("DTX buffer exceeds addressable memory");
        m_records.resize(std::size_t(count) * m_bytes);
        std::size_t offset = 0;
        for (const auto &r : ranges) {
            if (!r.count)
                continue;
            if (r.start != m_next || !r.start)
                m_file.seekg(std::streamoff(m_offset + r.start * m_bytes));
            auto bytes = std::size_t(r.count) * m_bytes;
            if (!m_file.read(reinterpret_cast<char *>(m_records.data() + offset), std::streamsize(bytes)))
                throw std::runtime_error("Failed to bulk-read DTX records");
            offset += bytes;
            m_next = r.start + r.count;
        }
        m_buffered = count;
        return count;
    }
    void read_buffered_into(Index index, bin::TrainingDataEntry &entry) override {
        if (index >= m_buffered)
            throw std::out_of_range("Record index outside DTX buffer");
        decode_into(m_records.data() + std::size_t(index) * m_bytes, entry);
    }

  private:
    void decode_into(const std::uint8_t *bytes, bin::TrainingDataEntry &e) {
        std::string error;
        if (!m_codec.decode(bytes, m_entry, error))
            throw std::runtime_error("Invalid DTX record: " + error);
        e.pos.clear();
        e.pos.turn = chess::Color(m_entry.turn);
        for (unsigned color = 0; color < 2; ++color)
            e.pos.place(chess::make_piece(m_config->king_type, chess::Color(color)), m_entry.kings[color]);
        for (auto p : m_entry.pieces)
            e.pos.place(chess::make_piece(p.type, chess::Color(p.color)), p.square);
        e.score = m_entry.score;
        e.result = m_entry.result;
        e.move = 0;
        e.ply = 1;
        bin::validate_entry(e);
    }
    std::ifstream m_file;
    VariantPtr m_config;
    bin::BatchStorage m_single;
    DtxTrainingRecord::Header m_header;
    DtxTrainingRecord::Codec m_codec;
    DtxTrainingRecord::Entry m_entry;
    Index m_offset = 0, m_count = 0, m_next = 0, m_buffered = 0;
    unsigned m_bytes = 0;
    std::vector<std::uint8_t> m_buffer, m_records;
};

inline std::unique_ptr<IndexedTrainingDataSource>
open_indexed_training_data_source(const std::string &filename, unsigned bytes = 72,
                                  VariantPtr cfg = default_variant()) {
    std::string ext = std::filesystem::path(filename).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (ext == ".dtx")
        return std::make_unique<DtxIndexedTrainingDataSource>(filename, std::move(cfg));
    if (filename.size() >= 4 && filename.compare(filename.size() - 4, 4, ".bin") == 0)
        return std::make_unique<BinIndexedTrainingDataSource>(filename, bytes, std::move(cfg));
    throw std::invalid_argument("No indexed training-data backend for: " + filename);
}
} // namespace training_data
