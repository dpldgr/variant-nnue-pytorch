#pragma once

#include "nnue_training_data_formats.h"
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
        : m_file(filename, std::ios::binary | std::ios::ate) {
        if (!m_file) throw std::runtime_error("Cannot open training data: " + filename);
        auto size = m_file.tellg();
        if (size < 0) throw std::runtime_error("Cannot determine training data size: " + filename);
        auto bytes = static_cast<Index>(size);
        if (!bytes) throw std::runtime_error("Training dataset is empty: " + filename);
        if (bytes % sizeof(bin::nodchip::PackedSfenValue))
            throw std::runtime_error("Incomplete .bin record (check the loader's DATA_SIZE): " + filename);
        m_count = bytes / sizeof(bin::nodchip::PackedSfenValue);
    }

    Index record_count() const override { return m_count; }

    bin::TrainingDataEntry read_record(Index index) override {
        if (index >= m_count) throw std::out_of_range("Record index outside training dataset");
        bin::nodchip::PackedSfenValue packed;
        m_file.seekg(static_cast<std::streamoff>(index * sizeof(packed)));
        if (!m_file.read(reinterpret_cast<char*>(&packed), sizeof(packed)))
            throw std::runtime_error("Failed to read indexed training record");
        return bin::packedSfenValueToTrainingDataEntry(packed);
    }

private:
    std::ifstream m_file;
    Index m_count;
};

inline std::unique_ptr<IndexedTrainingDataSource> open_indexed_training_data_source(const std::string& filename) {
    if (filename.size() >= 4 && filename.compare(filename.size() - 4, 4, ".bin") == 0)
        return std::make_unique<BinIndexedTrainingDataSource>(filename);
    throw std::invalid_argument("No indexed training-data backend for: " + filename);
}

} // namespace training_data
