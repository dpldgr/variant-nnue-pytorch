#pragma once

#include <functional>
#include "indexed_training_data_source.h"
#include "nnue_training_data_stream.h"
#include "record_permutation.h"

namespace training_data {

// One finite traversal. Epochs are explicit, so a final partial batch cannot
// consume records from the following epoch, even with filtering enabled.
class PrpSfenInputStream : public BasicSfenInputStream {
public:
    PrpSfenInputStream(std::unique_ptr<IndexedTrainingDataSource> source,
                      std::uint64_t seed, std::uint64_t epoch,
                      std::function<bool(const bin::TrainingDataEntry&)> skip = nullptr)
        : m_source(std::move(source)), m_count(m_source->record_count()),
          m_permutation(m_count, seed, epoch), m_skip(std::move(skip)) {}

    std::optional<bin::TrainingDataEntry> next() override {
        while (m_position < m_count) {
            auto entry = m_source->read_record(m_permutation(m_position++));
            if (!m_skip || !m_skip(entry)) return entry;
        }
        return std::nullopt;
    }

    bool eof() const override { return m_position == m_count; }

private:
    std::unique_ptr<IndexedTrainingDataSource> m_source;
    const std::uint64_t m_count;
    RecordPermutation m_permutation;
    std::uint64_t m_position = 0;
    std::function<bool(const bin::TrainingDataEntry&)> m_skip;
};

class IndexedSequentialInputStream : public BasicSfenInputStream {
public:
    IndexedSequentialInputStream(std::unique_ptr<IndexedTrainingDataSource> source, bool cyclic,
                                std::function<bool(const bin::TrainingDataEntry&)> skip)
        : m_source(std::move(source)), m_count(m_source->record_count()),
          m_cyclic(cyclic), m_skip(std::move(skip)) {}
    std::optional<bin::TrainingDataEntry> next() override {
        std::uint64_t visited = 0;
        while (m_cyclic || m_position < m_count) {
            if (m_position == m_count) m_position = 0;
            auto entry = m_source->read_record(m_position++);
            if (!m_skip || !m_skip(entry)) return entry;
            if (++visited == m_count) return std::nullopt;
        }
        return std::nullopt;
    }
    bool eof() const override { return !m_cyclic && m_position == m_count; }
private:
    std::unique_ptr<IndexedTrainingDataSource> m_source;
    const std::uint64_t m_count;
    std::uint64_t m_position = 0;
    bool m_cyclic;
    std::function<bool(const bin::TrainingDataEntry&)> m_skip;
};

} // namespace training_data
