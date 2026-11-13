#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <iterator>
#include <future>
#include <mutex>
#include <thread>
#include <deque>
#include <random>
#include <map>
#include <exception>
#include <condition_variable>
#include <atomic>
#include <climits>
#include <stdexcept>

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"
#include "lib/prp_training_data_stream.h"

#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL
#endif

using namespace bin;
using namespace chess;

// Compile-time feature-family selection, runtime variant configuration.
enum class Family { KP, KPF, KA, KAF, V2, V2F };
template <Family F> struct RuntimeFeatures {
    training_data::VariantPtr config;
    int INPUTS, MAX_ACTIVE_FEATURES;
    unsigned planes, base_inputs, virtual_board, pt_real;
    std::vector<unsigned> offsets;
    explicit RuntimeFeatures(training_data::VariantPtr cfg) : config(std::move(cfg)) {
        const auto &c = *config;
        constexpr bool kp = F == Family::KP || F == Family::KPF;
        constexpr bool v2 = F == Family::V2 || F == Family::V2F;
        constexpr bool factor = F == Family::KPF || F == Family::KAF || F == Family::V2F;
        pt_real = 2 * c.PIECE_TYPES - (kp ? 2u : (v2 && c.KING_SQUARES > 1 ? 1u : 0u));
        std::uint64_t plane =
            std::uint64_t(c.squares) * pt_real +
            (v2 ? std::uint64_t(c.hand_capacity) * (pt_real - unsigned(c.KING_SQUARES > 1)) : 1);
        std::uint64_t base = plane * (v2 ? c.KING_SQUARES : c.squares);
        std::uint64_t extra = kp ? std::uint64_t(c.squares) * (pt_real + 1)
                                 : std::uint64_t(c.squares) * 2 * c.PIECE_TYPES +
                                       (v2 ? std::uint64_t(c.hand_capacity) *
                                                 (2 * c.PIECE_TYPES - 2 * unsigned(c.KING_SQUARES > 1))
                                           : 0);
        if (plane > INT_MAX || base + (factor ? extra : 0) > INT_MAX)
            throw std::invalid_argument("Variant feature dimensions exceed int32 feature IDs");
        planes = unsigned(plane);
        base_inputs = unsigned(base);
        virtual_board = c.squares * 2 * c.PIECE_TYPES;
        INPUTS = int(base + (factor ? extra : 0));
        MAX_ACTIVE_FEATURES = int(c.PIECE_COUNT * (factor ? 2 : 1) + (F == Family::KPF ? 1 : 0));
        offsets.resize(2 * c.PIECE_TYPES);
        for (unsigned p = 0; p < offsets.size(); ++p)
            offsets[p] = ((v2 && pt_real % 2 && p == pt_real) ? p - 1 : p) * c.squares;
    }
    std::pair<int, int> fill_features_sparse(const TrainingDataEntry &e, int *ids, float *values,
                                             Color color) const {
        const auto &c = *config;
        const auto &pos = e.pos;
        const unsigned pov = unsigned(color);
        constexpr bool kp = F == Family::KP || F == Family::KPF;
        constexpr bool v2 = F == Family::V2 || F == Family::V2F;
        const auto &orientation = pov ? (kp ? c.rotate : c.flip) : c.identity;
        unsigned king = orientation[pos.kingSquare(color)];
        unsigned mapped = c.king_map[king];
        if (mapped >= c.KING_SQUARES)
            throw std::runtime_error("King outside configured reference-square mapping");
        unsigned anchor = mapped * planes;
        int j = 0;
        for (unsigned sq = 0; sq < c.squares; ++sq) {
            auto p = pos.board[sq];
            if (p == chess::NoPiece || (kp && chess::type_of(p) == c.king_type))
                continue;
            unsigned plane = offsets[unsigned(p) ^ pov];
            ids[j] = int((v2 ? 0u : 1u) + orientation[sq] + plane + anchor);
            values[j++] = 1.0f;
        }
        if constexpr (v2) {
            for (unsigned type = 0; type < c.king_type; ++type)
                for (unsigned owner = 0; owner < 2; ++owner) {
                    unsigned pc = 2 * type + owner;
                    for (unsigned n = 0; n < pos.hands[pc]; ++n) {
                        ids[j] = int(n + (pc ^ pov) * c.hand_capacity + c.squares * pt_real + anchor);
                        values[j++] = 1.0f;
                    }
                }
        }
        if constexpr (F == Family::KPF) {
            ids[j] = int(base_inputs + king);
            values[j] = float(j);
            ++j;
        }
        if constexpr (F == Family::KPF || F == Family::KAF || F == Family::V2F) {
            unsigned extra = base_inputs + (kp ? c.squares : 0);
            for (unsigned sq = 0; sq < c.squares; ++sq) {
                auto p = pos.board[sq];
                if (p == chess::NoPiece || (kp && chess::type_of(p) == c.king_type))
                    continue;
                ids[j] = int(extra + (unsigned(p) ^ pov) * c.squares + orientation[sq]);
                values[j++] = 1.0f;
            }
            if constexpr (v2) {
                for (unsigned type = 0; type < c.king_type; ++type)
                    for (unsigned owner = 0; owner < 2; ++owner) {
                        unsigned pc = 2 * type + owner;
                        for (unsigned n = 0; n < pos.hands[pc]; ++n) {
                            ids[j] = int(extra + virtual_board + n + (pc ^ pov) * c.hand_capacity);
                            values[j++] = 1.0f;
                        }
                    }
            }
        }
        return {j, INPUTS};
    }
};

struct SparseBatch {
    static constexpr bool IS_BATCH = true;

    template <typename FeatureSetT>
    SparseBatch(const FeatureSetT &fs, const std::vector<TrainingDataEntry> &entries) {
        num_inputs = fs.INPUTS;
        size = entries.size();
        const std::size_t slots = std::size_t(size) * fs.MAX_ACTIVE_FEATURES;
        // Keep constructor failures exception-safe without changing batch ABI.
        auto own_is_white = std::unique_ptr<float[]>(new float[size]);
        auto own_outcome = std::unique_ptr<float[]>(new float[size]);
        auto own_score = std::unique_ptr<float[]>(new float[size]);
        auto own_white = std::unique_ptr<int[]>(new int[slots]);
        auto own_black = std::unique_ptr<int[]>(new int[slots]);
        auto own_white_values = std::unique_ptr<float[]>(new float[slots]);
        auto own_black_values = std::unique_ptr<float[]>(new float[slots]);
        auto own_psqt = std::unique_ptr<int[]>(new int[size]);
        auto own_stack = std::unique_ptr<int[]>(new int[size]);
        is_white = own_is_white.get();
        outcome = own_outcome.get();
        score = own_score.get();
        white = own_white.get();
        black = own_black.get();
        white_values = own_white_values.get();
        black_values = own_black_values.get();
        psqt_indices = own_psqt.get();
        layer_stack_indices = own_stack.get();

        num_active_white_features = 0;
        num_active_black_features = 0;
        max_active_features = fs.MAX_ACTIVE_FEATURES;

        for (std::size_t i = 0; i < slots; ++i)
            white[i] = -1;
        for (std::size_t i = 0; i < slots; ++i)
            black[i] = -1;
        for (std::size_t i = 0; i < slots; ++i)
            white_values[i] = 0.0f;
        for (std::size_t i = 0; i < slots; ++i)
            black_values[i] = 0.0f;

        for (int i = 0; i < entries.size(); ++i) {
            fill_entry(fs, i, entries[i]);
        }
        own_is_white.release();
        own_outcome.release();
        own_score.release();
        own_white.release();
        own_black.release();
        own_white_values.release();
        own_black_values.release();
        own_psqt.release();
        own_stack.release();
    }

    int num_inputs;
    int size;

    float *is_white;
    float *outcome;
    float *score;
    int num_active_white_features;
    int num_active_black_features;
    int max_active_features;
    int *white;
    int *black;
    float *white_values;
    float *black_values;
    int *psqt_indices;
    int *layer_stack_indices;

    ~SparseBatch() {
        delete[] is_white;
        delete[] outcome;
        delete[] score;
        delete[] white;
        delete[] black;
        delete[] white_values;
        delete[] black_values;
        delete[] psqt_indices;
        delete[] layer_stack_indices;
    }

  private:
    template <typename FeatureSetT>
    void fill_entry(const FeatureSetT &fs, int i, const TrainingDataEntry &e) {
        is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        psqt_indices[i] = (e.pos.pieceCount() - 1) * 8 / fs.config->PIECE_COUNT;
        layer_stack_indices[i] = psqt_indices[i];
        fill_features(fs, i, e);
    }

    template <typename FeatureSetT>
    void fill_features(const FeatureSetT &fs, int i, const TrainingDataEntry &e) {
        const int offset = i * fs.MAX_ACTIVE_FEATURES;
        num_active_white_features +=
            fs.fill_features_sparse(e, white + offset, white_values + offset, Color::White).first;
        num_active_black_features +=
            fs.fill_features_sparse(e, black + offset, black_values + offset, Color::Black).first;
    }
};

struct AnyStream {
    virtual ~AnyStream() = default;
};

template <typename StorageT> struct Stream : AnyStream {
    using StorageType = StorageT;

    explicit Stream(std::unique_ptr<training_data::BasicSfenInputStream> stream)
        : m_stream(std::move(stream)) {}

    virtual StorageT *next() = 0;

  protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

template <typename FeatureSetT, typename StorageT> struct FeaturedBatchStream : Stream<StorageT> {
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(int concurrency, int batch_size,
                        std::unique_ptr<training_data::BasicSfenInputStream> input, FeatureSet fs)
        : BaseType(std::move(input)), m_features(std::move(fs)), m_batch_size(batch_size),
          m_concurrency(concurrency) {
        m_stop_flag.store(false);

        auto worker = [this]() {
            try {
                std::vector<TrainingDataEntry> entries;
                entries.reserve(m_batch_size);
                bin::BatchStorage storage(this->m_stream->config(), unsigned(m_batch_size));
                while (!m_stop_flag.load()) {
                    entries.clear();

                    {
                        std::unique_lock lock(m_stream_mutex);
                        BaseType::m_stream->fill(entries, m_batch_size, &storage);
                        if (entries.empty()) {
                            break;
                        }
                    }

                    auto batch = new StorageT(m_features, entries);

                    {
                        std::unique_lock lock(m_batch_mutex);
                        m_batches_not_full.wait(lock, [this]() {
                            return m_batches.size() < m_concurrency + 1 || m_stop_flag.load();
                        });

                        m_batches.emplace_back(batch);

                        lock.unlock();
                        m_batches_any.notify_one();
                    }
                }
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(m_batch_mutex);
                    m_error = std::current_exception();
                    m_stop_flag.store(true);
                }
                m_batches_not_full.notify_all();
            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads =
            std::max(1, concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread));

        try {
            for (int i = 0; i < num_feature_threads; ++i) {
                m_workers.emplace_back(worker);

                // This cannot be done in the thread worker. We need
                // to have a guarantee that this is incremented, but if
                // we did it in the worker there's no guarantee
                // that it executed.
                m_num_workers.fetch_add(1);
            }
        } catch (...) {
            m_stop_flag.store(true);
            m_batches_not_full.notify_all();
            for (auto &worker : m_workers)
                worker.join();
            for (auto *batch : m_batches)
                delete batch;
            throw;
        }
    }

    StorageT *next() override {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock,
                           [this]() { return m_error || !m_batches.empty() || m_num_workers.load() == 0; });

        if (m_error)
            std::rethrow_exception(m_error);
        if (!m_batches.empty()) {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FeaturedBatchStream() {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto &worker : m_workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }

        for (auto &batch : m_batches) {
            delete batch;
        }
    }

  private:
    std::exception_ptr m_error;
    FeatureSet m_features;
    int m_batch_size;
    int m_concurrency;
    std::deque<StorageT *> m_batches;
    std::mutex m_batch_mutex;
    std::mutex m_stream_mutex;
    std::condition_variable m_batches_not_full;
    std::condition_variable m_batches_any;
    std::atomic_bool m_stop_flag;
    std::atomic_int m_num_workers{0};

    std::vector<std::thread> m_workers;
};

// PRP-only queue: keep the legacy sequential pipeline unchanged. Jobs receive
// sequence numbers when read; features can finish in parallel, but consumption
// remains ordered. Reserving a bounded number of jobs before reading also
// guarantees that a slow first job can always publish without a full-queue deadlock.
template <typename FeatureSetT, typename StorageT> struct PrpFeaturedBatchStream : Stream<StorageT> {
    PrpFeaturedBatchStream(int concurrency, int batch_size,
                           std::unique_ptr<training_data::BasicSfenInputStream> input, FeatureSetT fs)
        : Stream<StorageT>(std::move(input)), m_features(std::move(fs)), m_batch_size(batch_size),
          m_capacity(static_cast<std::size_t>(concurrency) + 1),
          m_active(std::max(1, concurrency - std::max(1, concurrency / 2))) {
        const int worker_count = m_active;
        try {
            for (int i = 0; i < worker_count; ++i)
                m_workers.emplace_back([this] { work(); });
        } catch (...) {
            stop();
            for (auto &worker : m_workers)
                worker.join();
            throw;
        }
    }

    StorageT *next() override {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_changed.wait(lock, [this] { return m_error || m_ready.count(m_output) || m_active == 0; });
        if (m_error)
            std::rethrow_exception(m_error);
        auto found = m_ready.find(m_output);
        if (found == m_ready.end())
            return nullptr;
        auto result = found->second.release();
        m_ready.erase(found);
        ++m_output;
        --m_outstanding;
        lock.unlock();
        m_changed.notify_all();
        return result;
    }

    ~PrpFeaturedBatchStream() override {
        stop();
        for (auto &worker : m_workers)
            worker.join();
    }

  private:
    void stop() {
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_stop = true;
        }
        m_changed.notify_all();
    }

    void work() {
        try {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);
            bin::BatchStorage storage(this->m_stream->config(), unsigned(m_batch_size));
            for (;;) {
                entries.clear();
                std::uint64_t id;
                {
                    std::unique_lock<std::mutex> read_lock(m_read_mutex);
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_changed.wait(lock,
                                       [this] { return m_stop || m_eof || m_outstanding < m_capacity; });
                        if (m_stop || m_eof)
                            break;
                        ++m_outstanding;
                        id = m_input++;
                    }
                    this->m_stream->fill(entries, m_batch_size, &storage);
                    if (entries.empty()) {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        --m_outstanding;
                        m_eof = true;
                        m_changed.notify_all();
                        break;
                    }
                }
                auto batch = std::make_unique<StorageT>(m_features, entries);
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    if (m_stop)
                        break;
                    m_ready.emplace(id, std::move(batch));
                }
                m_changed.notify_all();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_error)
                m_error = std::current_exception();
            m_stop = true;
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            --m_active;
        }
        m_changed.notify_all();
    }

    FeatureSetT m_features;
    int m_batch_size;
    std::size_t m_capacity, m_outstanding = 0;
    int m_active;
    bool m_stop = false, m_eof = false;
    std::uint64_t m_input = 0, m_output = 0;
    std::exception_ptr m_error;
    std::mutex m_mutex, m_read_mutex;
    std::condition_variable m_changed;
    std::map<std::uint64_t, std::unique_ptr<StorageT>> m_ready;
    std::vector<std::thread> m_workers;
};

std::function<bool(const TrainingDataEntry &)> make_skip_predicate(bool filtered, int random_fen_skipping) {
    if (filtered || random_fen_skipping) {
        return [random_fen_skipping, prob = double(random_fen_skipping) / (random_fen_skipping + 1),
                filtered](const TrainingDataEntry &e) {
            auto do_skip = [&]() {
                std::bernoulli_distribution distrib(prob);
                auto &prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_filter = [&]() { return false; };

            static thread_local std::mt19937 gen(std::random_device{}());
            return (random_fen_skipping && do_skip()) || (filtered && do_filter());
        };
    }

    return nullptr;
}

// Unlike the legacy reader's thread-local RNG, PRP filtering belongs to the
// traversal and is reproducible regardless of which feature worker reads next.
std::function<bool(const TrainingDataEntry &)>
make_prp_skip_predicate(bool filtered, int random_fen_skipping, std::uint64_t seed, std::uint64_t epoch) {
    if (random_fen_skipping < 0)
        throw std::invalid_argument("random_fen_skipping must be nonnegative");
    auto filter = make_skip_predicate(filtered, 0);
    if (!random_fen_skipping)
        return filter;
    return [filter, rng = std::mt19937_64(training_data::RecordPermutation::mix(seed ^ epoch)),
            probability = double(random_fen_skipping) /
                          (random_fen_skipping + 1.0)](const TrainingDataEntry &entry) mutable {
        return std::bernoulli_distribution(probability)(rng) || (filter && filter(entry));
    };
}

static thread_local std::string prp_last_error;

using ConfigHandle = training_data::VariantPtr;
using namespace training_data;
template <Family F>
Stream<SparseBatch> *make_stream(VariantPtr cfg, int concurrency, const char *filename, int batch_size,
                                 bool cyclic, bool filtered, int skipping, std::uint64_t seed,
                                 std::uint64_t epoch, unsigned bytes, bool shuffle,
                                 std::uint64_t block_size = 0, unsigned buffer_blocks = 0) {
    RuntimeFeatures<F> features(cfg);
    if (std::uint64_t(batch_size) * features.MAX_ACTIVE_FEATURES > INT_MAX)
        throw std::invalid_argument("Batch feature offsets exceed int32 capacity");
    auto source = open_indexed_training_data_source(filename, bytes, cfg);
    auto skip = (shuffle || bytes != 72) ? make_prp_skip_predicate(filtered, skipping, seed, epoch)
                                         : make_skip_predicate(filtered, skipping);
    std::unique_ptr<BasicSfenInputStream> input;
    if (shuffle && block_size)
        input = std::make_unique<BufferedPrpSfenInputStream>(std::move(source), seed, epoch, block_size,
                                                             buffer_blocks, skip);
    else if (shuffle)
        input = std::make_unique<PrpSfenInputStream>(std::move(source), seed, epoch, skip);
    else
        input = std::make_unique<IndexedSequentialInputStream>(std::move(source), cyclic, skip, bytes == 72);
    if (shuffle || bytes != 72)
        return new PrpFeaturedBatchStream<RuntimeFeatures<F>, SparseBatch>(
            concurrency, batch_size, std::move(input), std::move(features));
    return new FeaturedBatchStream<RuntimeFeatures<F>, SparseBatch>(concurrency, batch_size, std::move(input),
                                                                    std::move(features));
}
extern "C" {
EXPORT unsigned CDECL get_loader_abi_version() { return 2; }
EXPORT const char *CDECL get_prp_last_error() { return prp_last_error.c_str(); }
EXPORT ConfigHandle *CDECL create_variant_context(unsigned files, unsigned ranks, unsigned types,
                                                  unsigned pieces, unsigned kings, unsigned pockets) {
    prp_last_error.clear();
    try {
        return new ConfigHandle(
            std::make_shared<const VariantConfig>(files, ranks, types, pieces, kings, pockets));
    } catch (const std::exception &e) {
        prp_last_error = e.what();
        return nullptr;
    }
}
EXPORT void CDECL destroy_variant_context(ConfigHandle *h) { delete h; }
EXPORT std::uint64_t CDECL generic_record_count(ConfigHandle *h, const char *filename, unsigned bytes) {
    prp_last_error.clear();
    try {
        if (!h || !filename)
            throw std::invalid_argument("Missing variant context or filename");
        return open_indexed_training_data_source(filename, bytes, *h)->record_count();
    } catch (const std::exception &e) {
        prp_last_error = e.what();
        return 0;
    }
}
static Stream<SparseBatch> *create_stream_impl(ConfigHandle *h, const char *feature, int concurrency,
                                               const char *filename, int batch_size, bool cyclic,
                                               bool filtered, int skipping, std::uint64_t seed,
                                               std::uint64_t epoch, unsigned bytes, bool shuffle,
                                               std::uint64_t block_size, unsigned buffer_blocks) {
    prp_last_error.clear();
    try {
        if (!h || !feature || !filename || concurrency < 1 || batch_size < 1 || skipping < 0 ||
            (shuffle && cyclic))
            throw std::invalid_argument("Invalid generic reader settings");
        std::string_view name(feature);
        if (name == "HalfKP")
            return make_stream<Family::KP>(*h, concurrency, filename, batch_size, cyclic, filtered, skipping,
                                           seed, epoch, bytes, shuffle, block_size, buffer_blocks);
        if (name == "HalfKP^")
            return make_stream<Family::KPF>(*h, concurrency, filename, batch_size, cyclic, filtered, skipping,
                                            seed, epoch, bytes, shuffle, block_size, buffer_blocks);
        if (name == "HalfKA")
            return make_stream<Family::KA>(*h, concurrency, filename, batch_size, cyclic, filtered, skipping,
                                           seed, epoch, bytes, shuffle, block_size, buffer_blocks);
        if (name == "HalfKA^")
            return make_stream<Family::KAF>(*h, concurrency, filename, batch_size, cyclic, filtered, skipping,
                                            seed, epoch, bytes, shuffle, block_size, buffer_blocks);
        if (name == "HalfKAv2")
            return make_stream<Family::V2>(*h, concurrency, filename, batch_size, cyclic, filtered, skipping,
                                           seed, epoch, bytes, shuffle, block_size, buffer_blocks);
        if (name == "HalfKAv2^")
            return make_stream<Family::V2F>(*h, concurrency, filename, batch_size, cyclic, filtered, skipping,
                                            seed, epoch, bytes, shuffle, block_size, buffer_blocks);
        throw std::invalid_argument("Unknown feature_set: " + std::string(name));
    } catch (const std::exception &e) {
        prp_last_error = e.what();
        return nullptr;
    }
}
EXPORT Stream<SparseBatch> *CDECL create_generic_batch_stream(
    ConfigHandle *h, const char *feature, int concurrency, const char *filename, int batch_size, bool cyclic,
    bool filtered, int skipping, std::uint64_t seed, std::uint64_t epoch, unsigned bytes, bool shuffle) {
    return create_stream_impl(h, feature, concurrency, filename, batch_size, cyclic, filtered, skipping, seed,
                              epoch, bytes, shuffle, 0, 0);
}
EXPORT Stream<SparseBatch> *CDECL create_buffered_prp_batch_stream(
    ConfigHandle *h, const char *feature, int concurrency, const char *filename, int batch_size,
    bool filtered, int skipping, std::uint64_t seed, std::uint64_t epoch, unsigned bytes,
    std::uint64_t block_size, unsigned buffer_blocks) {
    if (!block_size || !buffer_blocks) {
        prp_last_error = "PRP block size and buffer block count must be positive";
        return nullptr;
    }
    return create_stream_impl(h, feature, concurrency, filename, batch_size, false, filtered, skipping, seed,
                              epoch, bytes, true, block_size, buffer_blocks);
}
EXPORT SparseBatch *CDECL fetch_next_generic_batch(Stream<SparseBatch> *stream) {
    prp_last_error.clear();
    try {
        if (!stream)
            throw std::invalid_argument("Missing batch stream");
        return stream->next();
    } catch (const std::exception &e) {
        prp_last_error = e.what();
        return nullptr;
    }
}
EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch> *stream) { delete stream; }
EXPORT void CDECL destroy_sparse_batch(SparseBatch *batch) { delete batch; }
}
