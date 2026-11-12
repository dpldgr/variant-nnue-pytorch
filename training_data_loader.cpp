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

#include "lib/nnue_training_data_formats.h"
#include "lib/nnue_training_data_stream.h"
#include "lib/rng.h"
#include "lib/prp_training_data_stream.h"

#if defined (__x86_64__)
#define EXPORT
#define CDECL
#else
#if defined (_MSC_VER)
#define EXPORT __declspec(dllexport)
#define CDECL __cdecl
#else
#define EXPORT
#define CDECL __attribute__ ((__cdecl__))
#endif
#endif

using namespace bin;
using namespace chess;

static constexpr int MAX_PIECES = PIECE_COUNT;
static constexpr int MAX_HAND_PIECES = POCKETS ? 2 * static_cast<int>(File::FILE_NB) : 0;

static Square orient(Color color, Square sq)
{
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        // IMPORTANT: for now we use rotate180 instead of rank flip
        //            for compatibility with the stockfish master branch.
        //            Note that this is inconsistent with nodchip/master.
        return flip_horizontally(flip_vertically(sq));
    }
}

static Square orient_flip(Color color, Square sq)
{
    if (sq == Square::NB)
        // map missing king to zero
        return Square::MIN;
    if (color == Color::White)
    {
        return sq;
    }
    else
    {
        return flip_vertically(sq);
    }
}

static int map_king(Square sq)
{
    // palace squares for Xiangi/Janggi
    // map accessible king squares skipping the gaps
    if (Square::KNB == Square(9) && Square::KNB != Square::NB)
        return (int(sq) - 6 * (int(sq) / int(File::FILE_NB)) - 3) % int(Square::KNB);

    return int(sq) % int(Square::KNB);
}

struct HalfKP {
    static constexpr int NUM_SQ = static_cast<int>(Square::NB);
    static constexpr int NUM_PT = static_cast<int>(PieceType::MaxPiece) * 2;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = MAX_PIECES;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        return 1 + static_cast<int>(orient(color, sq)) + p_idx * NUM_SQ + map_king(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto ksq = pos.kingSquare(color);

        // We order the features so that the resulting sparse
        // tensor is coalesced.
        int j = 0;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None || type_of(p) == PieceType::King)
                continue;
            values[j] = 1.0f;
            features[j] = feature_index(color, orient(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKPFactorized {
    // Factorized features
    static constexpr int K_INPUTS = HalfKP::NUM_SQ;
    static constexpr int PIECE_INPUTS = HalfKP::NUM_SQ * HalfKP::NUM_PT;
    static constexpr int INPUTS = HalfKP::INPUTS + K_INPUTS + PIECE_INPUTS;

    static constexpr int MAX_K_FEATURES = 1;
    static constexpr int MAX_PIECE_FEATURES = MAX_PIECES;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKP::MAX_ACTIVE_FEATURES + MAX_K_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto [start_j, offset] = HalfKP::fill_features_sparse(e, features, values, color);
        int j = start_j;
        auto& pos = e.pos;
        {
            // king square factor
            auto ksq = pos.kingSquare(color);
            features[j] = offset + static_cast<int>(orient(color, ksq));
            values[j] = static_cast<float>(start_j);
            ++j;
        }
        offset += K_INPUTS;

        // We order the features so that the resulting sparse
        // tensor is coalesced. Note that we can just sort
        // the parts where values are all 1.0f and leave the
        // halfk feature where it was.
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None || type_of(p) == PieceType::King)
                continue;
            auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKP::NUM_SQ) + static_cast<int>(orient(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKA {
    static constexpr int NUM_SQ = static_cast<int>(Square::NB);
    static constexpr int NUM_PT = (static_cast<int>(PieceType::MaxPiece) + 1) * 2;
    static constexpr int NUM_PLANES = (NUM_SQ * NUM_PT + 1);
    static constexpr int INPUTS = NUM_PLANES * NUM_SQ;

    static constexpr int MAX_ACTIVE_FEATURES = MAX_PIECES;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        return 1 + static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + map_king(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None)
                continue;
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAFactorized {
    // Factorized features
    static constexpr int PIECE_INPUTS = HalfKA::NUM_SQ * HalfKA::NUM_PT;
    static constexpr int INPUTS = HalfKA::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = MAX_PIECES;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKA::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKA::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;

        int j = start_j;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None)
                continue;
            auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKA::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        return { j, INPUTS };
    }
};

struct HalfKAv2 {
    static constexpr int NUM_KSQ = static_cast<int>(Square::KNB);
    static constexpr int NUM_SQ = static_cast<int>(Square::NB);
    static constexpr int NUM_PT = (static_cast<int>(PieceType::MaxPiece) + 1) * 2 - (NUM_KSQ > 1);
    static constexpr int NUM_PLANES = NUM_SQ * NUM_PT + MAX_HAND_PIECES * (NUM_PT - (NUM_KSQ > 1));
    static constexpr int INPUTS = NUM_PLANES * NUM_KSQ;

    static constexpr int MAX_ACTIVE_FEATURES = MAX_PIECES;

    static int feature_index(Color color, Square ksq, Square sq, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        if (NUM_PT % 2 && p_idx == NUM_PT)
            --p_idx; // pack the opposite king into the same NUM_SQ * NUM_SQ
        return static_cast<int>(orient_flip(color, sq)) + p_idx * NUM_SQ + map_king(ksq) * NUM_PLANES;
    }

    static int feature_index(Color color, Square ksq, int handCount, Piece p)
    {
        auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
        return handCount + p_idx * MAX_HAND_PIECES + NUM_SQ * NUM_PT + map_king(ksq) * NUM_PLANES;
    }

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        auto& pos = e.pos;
        auto ksq = pos.kingSquare(color);

        int j = 0;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None)
                continue;
            values[j] = 1.0f;
            features[j] = feature_index(color, orient_flip(color, ksq), sq, p);
            ++j;
        }

        for (PieceType pt = PieceType::Pawn; pt < PieceType::King; ++pt)
            for (Color c : { Color::White, Color::Black })
                for (int i = 0; i < pos.getHandCount(make_piece(pt, c)); i++)
                {
                    values[j] = 1.0f;
                    features[j] = feature_index(color, orient_flip(color, ksq), i, make_piece(pt, c));
                    ++j;
                }

        return { j, INPUTS };
    }
};

struct HalfKAv2Factorized {
    // Factorized features
    static constexpr int NUM_PT = (static_cast<int>(PieceType::MaxPiece) + 1) * 2;
    static constexpr int PIECE_INPUTS = HalfKAv2::NUM_SQ * NUM_PT + MAX_HAND_PIECES * (NUM_PT - 2 * (HalfKAv2::NUM_KSQ > 1));
    static constexpr int INPUTS = HalfKAv2::INPUTS + PIECE_INPUTS;

    static constexpr int MAX_PIECE_FEATURES = MAX_PIECES;
    static constexpr int MAX_ACTIVE_FEATURES = HalfKAv2::MAX_ACTIVE_FEATURES + MAX_PIECE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        const auto [start_j, offset] = HalfKAv2::fill_features_sparse(e, features, values, color);
        auto& pos = e.pos;

        int j = start_j;
        for(Square sq = Square::MIN; sq <= Square::MAX; ++sq)
        {
            auto p = pos.pieceAt(sq);
            if (p == Piece::None)
                continue;
            auto p_idx = static_cast<int>(type_of(p)) * 2 + (color_of(p) != color);
            values[j] = 1.0f;
            features[j] = offset + (p_idx * HalfKAv2::NUM_SQ) + static_cast<int>(orient_flip(color, sq));
            ++j;
        }

        for (PieceType pt = PieceType::Pawn; pt < PieceType::King; ++pt)
            for (Color c : { Color::White, Color::Black })
                for (int i = 0; i < pos.getHandCount(make_piece(pt, c)); i++)
                {
                    values[j] = 1.0f;
                    auto p_idx = static_cast<int>(pt) * 2 + (c != color);
                    features[j] = offset + i + p_idx * MAX_HAND_PIECES + HalfKAv2::NUM_SQ * NUM_PT;
                    ++j;
                }

        return { j, INPUTS };
    }
};

template <typename T, typename... Ts>
struct FeatureSet
{
    static_assert(sizeof...(Ts) == 0, "Currently only one feature subset supported.");

    static constexpr int INPUTS = T::INPUTS;
    static constexpr int MAX_ACTIVE_FEATURES = T::MAX_ACTIVE_FEATURES;

    static std::pair<int, int> fill_features_sparse(const TrainingDataEntry& e, int* features, float* values, Color color)
    {
        return T::fill_features_sparse(e, features, values, color);
    }
};

struct SparseBatch
{
    static constexpr bool IS_BATCH = true;

    template <typename... Ts>
    SparseBatch(FeatureSet<Ts...>, const std::vector<TrainingDataEntry>& entries)
    {
        num_inputs = FeatureSet<Ts...>::INPUTS;
        size = entries.size();
        is_white = new float[size];
        outcome = new float[size];
        score = new float[size];
        white = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black = new int[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        white_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        black_values = new float[size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES];
        psqt_indices = new int[size];
        layer_stack_indices = new int[size];

        num_active_white_features = 0;
        num_active_black_features = 0;
        max_active_features = FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;

        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black[i] = -1;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            white_values[i] = 0.0f;
        for (std::size_t i = 0; i < size * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES; ++i)
            black_values[i] = 0.0f;

        for(int i = 0; i < entries.size(); ++i)
        {
            fill_entry(FeatureSet<Ts...>{}, i, entries[i]);
        }
    }

    int num_inputs;
    int size;

    float* is_white;
    float* outcome;
    float* score;
    int num_active_white_features;
    int num_active_black_features;
    int max_active_features;
    int* white;
    int* black;
    float* white_values;
    float* black_values;
    int* psqt_indices;
    int* layer_stack_indices;

    ~SparseBatch()
    {
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

    template <typename... Ts>
    void fill_entry(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        is_white[i] = static_cast<float>(e.pos.sideToMove() == Color::White);
        outcome[i] = (e.result + 1.0f) / 2.0f;
        score[i] = e.score;
        psqt_indices[i] = (e.pos.pieceCount() - 1) * 8 / MAX_PIECES;
        layer_stack_indices[i] = psqt_indices[i];
        fill_features(FeatureSet<Ts...>{}, i, e);
    }

    template <typename... Ts>
    void fill_features(FeatureSet<Ts...>, int i, const TrainingDataEntry& e)
    {
        const int offset = i * FeatureSet<Ts...>::MAX_ACTIVE_FEATURES;
        num_active_white_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, white + offset, white_values + offset, Color::White)
            .first;
        num_active_black_features +=
            FeatureSet<Ts...>::fill_features_sparse(e, black + offset, black_values + offset, Color::Black)
            .first;
    }
};

struct AnyStream
{
    virtual ~AnyStream() = default;
};

template <typename StorageT>
struct Stream : AnyStream
{
    using StorageType = StorageT;

    explicit Stream(std::unique_ptr<training_data::BasicSfenInputStream> stream) :
        m_stream(std::move(stream)) {}

    Stream(int concurrency, const char* filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        m_stream(training_data::open_sfen_input_file_parallel(concurrency, filename, cyclic, skipPredicate))
    {
    }

    virtual StorageT* next() = 0;

protected:
    std::unique_ptr<training_data::BasicSfenInputStream> m_stream;
};

template <typename StorageT>
struct AsyncStream : Stream<StorageT>
{
    using BaseType = Stream<StorageT>;

    AsyncStream(int concurrency, const char* filename, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(1, filename, cyclic, skipPredicate)
    {
    }

    ~AsyncStream()
    {
        if (m_next.valid())
        {
            delete m_next.get();
        }
    }

protected:
    std::future<StorageT*> m_next;
};

template <typename FeatureSetT, typename StorageT>
struct FeaturedBatchStream : Stream<StorageT>
{
    static_assert(StorageT::IS_BATCH);

    using FeatureSet = FeatureSetT;
    using BaseType = Stream<StorageT>;

    static constexpr int num_feature_threads_per_reading_thread = 2;

    FeaturedBatchStream(int concurrency, const char* filename, int batch_size, bool cyclic, std::function<bool(const TrainingDataEntry&)> skipPredicate) :
        BaseType(
            std::max(
                1,
                concurrency / num_feature_threads_per_reading_thread
            ),
            filename,
            cyclic,
            skipPredicate
        ),
        m_concurrency(concurrency),
        m_batch_size(batch_size)
    {
        m_stop_flag.store(false);

        auto worker = [this]()
        {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);

            while(!m_stop_flag.load())
            {
                entries.clear();

                {
                    std::unique_lock lock(m_stream_mutex);
                    BaseType::m_stream->fill(entries, m_batch_size);
                    if (entries.empty())
                    {
                        break;
                    }
                }

                auto batch = new StorageT(FeatureSet{}, entries);

                {
                    std::unique_lock lock(m_batch_mutex);
                    m_batches_not_full.wait(lock, [this]() { return m_batches.size() < m_concurrency + 1 || m_stop_flag.load(); });

                    m_batches.emplace_back(batch);

                    lock.unlock();
                    m_batches_any.notify_one();
                }

            }
            m_num_workers.fetch_sub(1);
            m_batches_any.notify_one();
        };

        const int num_feature_threads = std::max(
            1,
            concurrency - std::max(1, concurrency / num_feature_threads_per_reading_thread)
        );

        for (int i = 0; i < num_feature_threads; ++i)
        {
            m_workers.emplace_back(worker);

            // This cannot be done in the thread worker. We need
            // to have a guarantee that this is incremented, but if
            // we did it in the worker there's no guarantee
            // that it executed.
            m_num_workers.fetch_add(1);
        }
    }

    StorageT* next() override
    {
        std::unique_lock lock(m_batch_mutex);
        m_batches_any.wait(lock, [this]() { return !m_batches.empty() || m_num_workers.load() == 0; });

        if (!m_batches.empty())
        {
            auto batch = m_batches.front();
            m_batches.pop_front();

            lock.unlock();
            m_batches_not_full.notify_one();

            return batch;
        }
        return nullptr;
    }

    ~FeaturedBatchStream()
    {
        m_stop_flag.store(true);
        m_batches_not_full.notify_all();

        for (auto& worker : m_workers)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }

        for (auto& batch : m_batches)
        {
            delete batch;
        }
    }

private:
    int m_batch_size;
    int m_concurrency;
    std::deque<StorageT*> m_batches;
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
template <typename FeatureSetT, typename StorageT>
struct PrpFeaturedBatchStream : Stream<StorageT>
{
    PrpFeaturedBatchStream(int concurrency, int batch_size,
                          std::unique_ptr<training_data::BasicSfenInputStream> input)
        : Stream<StorageT>(std::move(input)), m_batch_size(batch_size),
          m_capacity(static_cast<std::size_t>(concurrency) + 1),
          m_active(std::max(1, concurrency - std::max(1, concurrency / 2))) {
        const int worker_count = m_active;
        try {
            for (int i = 0; i < worker_count; ++i)
                m_workers.emplace_back([this] { work(); });
        } catch (...) {
            stop();
            for (auto& worker : m_workers) worker.join();
            throw;
        }
    }

    StorageT* next() override {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_changed.wait(lock, [this] {
            return m_error || m_ready.count(m_output) || m_active == 0;
        });
        if (m_error) std::rethrow_exception(m_error);
        auto found = m_ready.find(m_output);
        if (found == m_ready.end()) return nullptr;
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
        for (auto& worker : m_workers) worker.join();
    }

private:
    void stop() {
        { std::lock_guard<std::mutex> lock(m_mutex); m_stop = true; }
        m_changed.notify_all();
    }

    void work() {
        try {
            std::vector<TrainingDataEntry> entries;
            entries.reserve(m_batch_size);
            for (;;) {
                entries.clear();
                std::uint64_t id;
                {
                    std::unique_lock<std::mutex> read_lock(m_read_mutex);
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        m_changed.wait(lock, [this] {
                            return m_stop || m_eof || m_outstanding < m_capacity;
                        });
                        if (m_stop || m_eof) break;
                        ++m_outstanding;
                        id = m_input++;
                    }
                    this->m_stream->fill(entries, m_batch_size);
                    if (entries.empty()) {
                        std::lock_guard<std::mutex> lock(m_mutex);
                        --m_outstanding;
                        m_eof = true;
                        m_changed.notify_all();
                        break;
                    }
                }
                auto batch = std::make_unique<StorageT>(FeatureSetT{}, entries);
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    if (m_stop) break;
                    m_ready.emplace(id, std::move(batch));
                }
                m_changed.notify_all();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(m_mutex);
            if (!m_error) m_error = std::current_exception();
            m_stop = true;
        }
        {
            std::lock_guard<std::mutex> lock(m_mutex);
            --m_active;
        }
        m_changed.notify_all();
    }

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

std::function<bool(const TrainingDataEntry&)> make_skip_predicate(bool filtered, int random_fen_skipping)
{
    if (filtered || random_fen_skipping)
    {
        return [
            random_fen_skipping,
            prob = double(random_fen_skipping) / (random_fen_skipping + 1),
            filtered
            ](const TrainingDataEntry& e){

            auto do_skip = [&]() {
                std::bernoulli_distribution distrib(prob);
                auto& prng = rng::get_thread_local_rng();
                return distrib(prng);
            };

            auto do_filter = [&]() {
                return false;
            };

            static thread_local std::mt19937 gen(std::random_device{}());
            return (random_fen_skipping && do_skip()) || (filtered && do_filter());
        };
    }

    return nullptr;
}

// Unlike the legacy reader's thread-local RNG, PRP filtering belongs to the
// traversal and is reproducible regardless of which feature worker reads next.
std::function<bool(const TrainingDataEntry&)> make_prp_skip_predicate(
    bool filtered, int random_fen_skipping, std::uint64_t seed, std::uint64_t epoch)
{
    if (random_fen_skipping < 0)
        throw std::invalid_argument("random_fen_skipping must be nonnegative");
    auto filter = make_skip_predicate(filtered, 0);
    if (!random_fen_skipping) return filter;
    return [filter, rng = std::mt19937_64(training_data::RecordPermutation::mix(seed ^ epoch)),
            probability = double(random_fen_skipping) / (random_fen_skipping + 1.0)]
           (const TrainingDataEntry& entry) mutable {
        return std::bernoulli_distribution(probability)(rng) || (filter && filter(entry));
    };
}

static thread_local std::string prp_last_error;

template <typename FeatureSetT>
Stream<SparseBatch>* make_prp_stream(int concurrency, const char* filename, int batch_size,
                                    bool filtered, int skipping, std::uint64_t seed, std::uint64_t epoch)
{
    auto input = std::make_unique<training_data::PrpSfenInputStream>(
        training_data::open_indexed_training_data_source(filename), seed, epoch,
        make_prp_skip_predicate(filtered, skipping, seed, epoch));
    return new PrpFeaturedBatchStream<FeatureSetT, SparseBatch>(concurrency, batch_size, std::move(input));
}

template <typename FeatureSetT>
Stream<SparseBatch>* make_sized_stream(int concurrency, const char* filename, int batch_size,
    bool cyclic, bool filtered, int skipping, std::uint64_t seed, std::uint64_t epoch,
    unsigned record_bytes, bool shuffle)
{
    auto source = training_data::open_indexed_training_data_source(filename, record_bytes);
    auto skip = make_prp_skip_predicate(filtered, skipping, seed, epoch);
    std::unique_ptr<training_data::BasicSfenInputStream> input;
    if (shuffle) input = std::make_unique<training_data::PrpSfenInputStream>(std::move(source), seed, epoch, skip);
    else input = std::make_unique<training_data::IndexedSequentialInputStream>(std::move(source), cyclic, skip);
    return new PrpFeaturedBatchStream<FeatureSetT, SparseBatch>(concurrency, batch_size, std::move(input));
}

extern "C" {

    // Additional entry points in the SAME library. The legacy ABI is unchanged.
    EXPORT const char* CDECL get_prp_last_error() { return prp_last_error.c_str(); }

    EXPORT std::uint64_t CDECL training_data_record_count_sized(const char* filename, unsigned record_bytes) {
        prp_last_error.clear();
        try { return training_data::open_indexed_training_data_source(filename, record_bytes)->record_count(); }
        catch (const std::exception& e) { prp_last_error = e.what(); return 0; }
    }

    EXPORT Stream<SparseBatch>* CDECL create_sized_sparse_batch_stream(
        const char* feature_set_c, int concurrency, const char* filename, int batch_size,
        bool cyclic, bool filtered, int skipping, std::uint64_t seed, std::uint64_t epoch,
        unsigned record_bytes, bool shuffle)
    {
        prp_last_error.clear();
        try {
            if ((shuffle && cyclic) || concurrency < 1 || batch_size < 1)
                throw std::invalid_argument("Invalid sized-reader batch/workers/epoch settings");
            std::string_view feature_set(feature_set_c);
            if (feature_set == "HalfKP") return make_sized_stream<FeatureSet<HalfKP>>(concurrency, filename, batch_size, cyclic, filtered, skipping, seed, epoch, record_bytes, shuffle);
            if (feature_set == "HalfKP^") return make_sized_stream<FeatureSet<HalfKPFactorized>>(concurrency, filename, batch_size, cyclic, filtered, skipping, seed, epoch, record_bytes, shuffle);
            if (feature_set == "HalfKA") return make_sized_stream<FeatureSet<HalfKA>>(concurrency, filename, batch_size, cyclic, filtered, skipping, seed, epoch, record_bytes, shuffle);
            if (feature_set == "HalfKA^") return make_sized_stream<FeatureSet<HalfKAFactorized>>(concurrency, filename, batch_size, cyclic, filtered, skipping, seed, epoch, record_bytes, shuffle);
            if (feature_set == "HalfKAv2") return make_sized_stream<FeatureSet<HalfKAv2>>(concurrency, filename, batch_size, cyclic, filtered, skipping, seed, epoch, record_bytes, shuffle);
            if (feature_set == "HalfKAv2^") return make_sized_stream<FeatureSet<HalfKAv2Factorized>>(concurrency, filename, batch_size, cyclic, filtered, skipping, seed, epoch, record_bytes, shuffle);
            throw std::invalid_argument("Unknown feature_set: " + std::string(feature_set));
        } catch (const std::exception& e) { prp_last_error = e.what(); return nullptr; }
    }

    EXPORT std::uint64_t CDECL training_data_record_count(const char* filename) {
        prp_last_error.clear();
        try {
            return training_data::open_indexed_training_data_source(filename)->record_count();
        } catch (const std::exception& e) { prp_last_error = e.what(); return 0; }
    }

    EXPORT Stream<SparseBatch>* CDECL create_prp_sparse_batch_stream(
        const char* feature_set_c, int concurrency, const char* filename, int batch_size,
        bool cyclic, bool filtered, int skipping, std::uint64_t seed, std::uint64_t epoch)
    {
        prp_last_error.clear();
        try {
            if (cyclic || concurrency < 1 || batch_size < 1)
                throw std::invalid_argument("PRP requires a finite epoch, positive workers and positive batch size");
            std::string_view feature_set(feature_set_c);
            if (feature_set == "HalfKP") return make_prp_stream<FeatureSet<HalfKP>>(concurrency, filename, batch_size, filtered, skipping, seed, epoch);
            if (feature_set == "HalfKP^") return make_prp_stream<FeatureSet<HalfKPFactorized>>(concurrency, filename, batch_size, filtered, skipping, seed, epoch);
            if (feature_set == "HalfKA") return make_prp_stream<FeatureSet<HalfKA>>(concurrency, filename, batch_size, filtered, skipping, seed, epoch);
            if (feature_set == "HalfKA^") return make_prp_stream<FeatureSet<HalfKAFactorized>>(concurrency, filename, batch_size, filtered, skipping, seed, epoch);
            if (feature_set == "HalfKAv2") return make_prp_stream<FeatureSet<HalfKAv2>>(concurrency, filename, batch_size, filtered, skipping, seed, epoch);
            if (feature_set == "HalfKAv2^") return make_prp_stream<FeatureSet<HalfKAv2Factorized>>(concurrency, filename, batch_size, filtered, skipping, seed, epoch);
            throw std::invalid_argument("Unknown feature_set: " + std::string(feature_set));
        } catch (const std::exception& e) { prp_last_error = e.what(); return nullptr; }
    }

    EXPORT SparseBatch* CDECL fetch_next_prp_sparse_batch(Stream<SparseBatch>* stream) {
        prp_last_error.clear();
        try { return stream->next(); }
        catch (const std::exception& e) { prp_last_error = e.what(); return nullptr; }
    }

    EXPORT Stream<SparseBatch>* CDECL create_sparse_batch_stream(const char* feature_set_c, int concurrency, const char* filename, int batch_size, bool cyclic, bool filtered, int random_fen_skipping)
    {
        auto skipPredicate = make_skip_predicate(filtered, random_fen_skipping);

        std::string_view feature_set(feature_set_c);
        if (feature_set == "HalfKP")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKP>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKP^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKPFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKA")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKA>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKA^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAFactorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKAv2")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAv2>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        else if (feature_set == "HalfKAv2^")
        {
            return new FeaturedBatchStream<FeatureSet<HalfKAv2Factorized>, SparseBatch>(concurrency, filename, batch_size, cyclic, skipPredicate);
        }
        fprintf(stderr, "Unknown feature_set %s\n", feature_set_c);
        return nullptr;
    }

    EXPORT void CDECL destroy_sparse_batch_stream(Stream<SparseBatch>* stream)
    {
        delete stream;
    }

    EXPORT SparseBatch* CDECL fetch_next_sparse_batch(Stream<SparseBatch>* stream)
    {
        return stream->next();
    }

    EXPORT void CDECL destroy_sparse_batch(SparseBatch* e)
    {
        delete e;
    }

}

/* benches */ //*
#include <chrono>

int main()
{
    auto stream = create_sparse_batch_stream("HalfKP", 4, "10m_d3_q_2.bin", 8192, true, false, 0);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i)
    {
        if (i % 100 == 0) std::cout << i << '\n';
        destroy_sparse_batch(stream->next());
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << (t1 - t0).count() / 1e9 << "s\n";
}
//*/
