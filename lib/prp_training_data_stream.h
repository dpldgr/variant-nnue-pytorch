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

} // namespace training_data
