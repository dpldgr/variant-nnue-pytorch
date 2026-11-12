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
    PrpSfenInputStream(std::unique_ptr<IndexedTrainingDataSource> source, std::uint64_t seed,
                       std::uint64_t epoch,
                       std::function<bool(const bin::TrainingDataEntry &)> skip = nullptr)
        : m_source(std::move(source)), m_count(m_source->record_count()), m_permutation(m_count, seed, epoch),
          m_skip(std::move(skip)) {}

    std::optional<bin::TrainingDataEntry> next() override {
        while (m_position < m_count) {
            auto entry = m_source->read_record(m_permutation(m_position++));
            if (!m_skip || !m_skip(entry))
                return entry;
        }
        return std::nullopt;
    }

    bool next_into(bin::TrainingDataEntry &entry) override {
        while (m_position < m_count) {
            m_source->read_into(m_permutation(m_position++), entry);
            if (!m_skip || !m_skip(entry))
                return true;
        }
        return false;
    }
    VariantPtr config() const override { return m_source->config(); }
    bool eof() const override { return m_position == m_count; }

  private:
    std::unique_ptr<IndexedTrainingDataSource> m_source;
    const std::uint64_t m_count;
    RecordPermutation m_permutation;
    std::uint64_t m_position = 0;
    std::function<bool(const bin::TrainingDataEntry &)> m_skip;
};

class IndexedSequentialInputStream : public BasicSfenInputStream {
  public:
    IndexedSequentialInputStream(std::unique_ptr<IndexedTrainingDataSource> source, bool cyclic,
                                 std::function<bool(const bin::TrainingDataEntry &)> skip,
                                 bool legacy_cycle = false)
        : m_source(std::move(source)), m_count(m_source->record_count()), m_cyclic(cyclic),
          m_legacy_cycle(legacy_cycle), m_skip(std::move(skip)) {}
    std::optional<bin::TrainingDataEntry> next() override {
        bin::TrainingDataEntry entry;
        if (!advance<false>(entry))
            return std::nullopt;
        return entry;
    }
    bool next_into(bin::TrainingDataEntry &entry) override { return advance<true>(entry); }
    VariantPtr config() const override { return m_source->config(); }
    bool eof() const override { return !m_cyclic && m_position == m_count; }

  private:
    template <bool Into> bool advance(bin::TrainingDataEntry &entry) {
        std::uint64_t visited = 0;
        bool wrapped = false;
        for (;;) {
            if (m_position == m_count) {
                if (!m_cyclic || (m_legacy_cycle && wrapped))
                    return false;
                m_position = 0;
                wrapped = true;
            }
            if constexpr (Into)
                m_source->read_into(m_position++, entry);
            else
                entry = m_source->read_record(m_position++);
            if (!m_skip || !m_skip(entry))
                return true;
            // Legacy PackedSfen permits the remaining tail plus one reopened
            // file; the compact sequential backend permits one full traversal.
            if (!m_legacy_cycle && ++visited == m_count)
                return false;
        }
    }
    std::unique_ptr<IndexedTrainingDataSource> m_source;
    const std::uint64_t m_count;
    std::uint64_t m_position = 0;
    bool m_cyclic, m_legacy_cycle;
    std::function<bool(const bin::TrainingDataEntry &)> m_skip;
};
} // namespace training_data
