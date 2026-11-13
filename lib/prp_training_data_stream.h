#pragma once

#include <functional>
#include <algorithm>
#include <vector>
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

// Permute blocks on disk, bulk-read each group, then permute every real record
// across the group in memory. fill() naturally joins batches across groups.
class BufferedPrpSfenInputStream : public BasicSfenInputStream {
  public:
    BufferedPrpSfenInputStream(std::unique_ptr<IndexedTrainingDataSource> source, std::uint64_t seed,
                               std::uint64_t epoch, std::uint64_t block_size = 16384,
                               unsigned buffer_blocks = 64,
                               std::function<bool(const bin::TrainingDataEntry &)> skip = nullptr)
        : m_source(std::move(source)), m_count(m_source->record_count()), m_block_size(block_size),
          m_buffer_blocks(buffer_blocks), m_seed(seed), m_epoch(epoch),
          m_block_count(block_size && m_count ? 1 + (m_count - 1) / block_size : 0),
          m_outer(m_block_count, RecordPermutation::mix(seed ^ UINT64_C(0x626c6f636b707270)), epoch),
          m_single(m_source->config(), 1), m_skip(std::move(skip)) {
        if (!block_size || !buffer_blocks)
            throw std::invalid_argument("PRP block size and buffer block count must be positive");
    }
    std::optional<bin::TrainingDataEntry> next() override {
        auto entry = m_single.entry(0);
        if (!next_into(entry))
            return std::nullopt;
        return entry;
    }
    bool next_into(bin::TrainingDataEntry &entry) override {
        while (m_consumed < m_count) {
            if (m_buffer_position == m_buffer_count)
                load_group();
            m_source->read_buffered_into((*m_inner)(m_buffer_position++), entry);
            ++m_consumed;
            if (!m_skip || !m_skip(entry))
                return true;
        }
        return false;
    }
    VariantPtr config() const override { return m_source->config(); }
    bool eof() const override { return m_consumed == m_count; }

  private:
    void load_group() {
        std::vector<IndexedTrainingDataSource::Range> ranges;
        auto blocks = std::min<std::uint64_t>(m_buffer_blocks, m_block_count - m_next_block);
        ranges.reserve(std::size_t(blocks));
        std::uint64_t expected = 0;
        for (std::uint64_t i = 0; i < blocks; ++i) {
            auto start = m_outer(m_next_block++) * m_block_size;
            auto count = std::min(m_block_size, m_count - start);
            ranges.push_back({start, count});
            expected += count;
        }
        // Read the selected ranges in file order, keeping group membership.
        std::sort(ranges.begin(), ranges.end(),
                  [](const auto &a, const auto &b) { return a.start < b.start; });
        m_buffer_count = m_source->buffer_ranges(ranges);
        if (!expected || m_buffer_count != expected)
            throw std::runtime_error("Bulk reader returned an incorrect record count");
        auto key =
            RecordPermutation::mix(m_seed ^ UINT64_C(0x7265636f72647072) ^ RecordPermutation::mix(m_group++));
        m_inner.emplace(m_buffer_count, key, m_epoch);
        m_buffer_position = 0;
    }
    std::unique_ptr<IndexedTrainingDataSource> m_source;
    const std::uint64_t m_count, m_block_size;
    const unsigned m_buffer_blocks;
    const std::uint64_t m_seed, m_epoch, m_block_count;
    RecordPermutation m_outer;
    bin::BatchStorage m_single;
    std::optional<RecordPermutation> m_inner;
    std::uint64_t m_next_block = 0, m_group = 0, m_consumed = 0;
    std::uint64_t m_buffer_count = 0, m_buffer_position = 0;
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
