#include "lib/prp_training_data_stream.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace training_data;
using Index = std::uint64_t;

static void check(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

// No file, offsets or fixed record size: exercise the same reader through a
// different storage implementation, including filtering and partial batches.
class MemorySource : public IndexedTrainingDataSource {
public:
    explicit MemorySource(Index count) : m_count(count) {}
    Index record_count() const override { return m_count; }
    bin::TrainingDataEntry read_record(Index index) override {
        check(index < m_count, "backend received out-of-range index");
        visited.push_back(index);
        bin::TrainingDataEntry entry{};
        entry.score = static_cast<std::int16_t>(index);
        return entry;
    }
    std::vector<Index> visited;
    Index buffer_ranges(const std::vector<Range>& ranges) override {
        buffered.clear();
        for (const auto& range : ranges) {
            check(range.start < m_count && range.count <= m_count-range.start,"invalid bulk range");
            for (Index i=0;i<range.count;++i) buffered.push_back(range.start+i);
        }
        loads.push_back(buffered.size());
        return buffered.size();
    }
    void read_buffered_into(Index index, bin::TrainingDataEntry& entry) override {
        entry=read_record(buffered.at(index));
    }
    std::vector<Index> buffered,loads;
private:
    Index m_count;
};

int main() {
    try {
        bool rejected = false;
        try { RecordPermutation bad(0, 42, 0); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "empty domain accepted");

        std::vector<Index> sizes;
        for (Index n = 1; n <= 257; ++n) sizes.push_back(n);
        for (Index n : {4095, 4096, 4097, 65535, 65536, 65537}) sizes.push_back(n);
        for (Index n : sizes) {
            for (Index epoch = 0; epoch < 3; ++epoch) {
                RecordPermutation p(n, 42, epoch);
                std::vector<bool> seen(n, false);
                for (Index i = 0; i < n; ++i) {
                    Index j = p(i);
                    check(j < n && !seen[j], "duplicate or out-of-range permutation output");
                    seen[j] = true;
                    check(p.inverse(j) == i && p(p.inverse(i)) == i, "inverse mismatch");
                }
                check(std::all_of(seen.begin(), seen.end(), [](bool x) { return x; }), "missing record");
            }
        }
        for (Index n : {(Index(1)<<32)+1, (Index(1)<<40)+1,
                         (Index(1)<<63)+1, std::numeric_limits<Index>::max()}) {
            RecordPermutation p(n, 12345, 123);
            for (Index i = 0; i < 10000; ++i) {
                Index x = i < 3 ? (i == 0 ? 0 : i == 1 ? n - 1 : n / 2) : RecordPermutation::mix(i) % n;
                check(p.inverse(p(x)) == x && p(p.inverse(x)) == x, "large-domain inverse mismatch");
            }
        }

        for (bool skip : {false, true}) {
            auto source = std::make_unique<MemorySource>(257);
            auto* observed = source.get();
            PrpSfenInputStream stream(std::move(source), 42, 7,
                [skip](const bin::TrainingDataEntry& e) { return skip && e.score % 3 == 0; });
            RecordPermutation p(257, 42, 7);
            std::vector<int> expected, actual;
            for (Index i = 0; i < 257; ++i) {
                auto id = p(i);
                if (!skip || id % 3) expected.push_back(static_cast<int>(id));
            }
            for (;;) {
                std::vector<bin::TrainingDataEntry> batch;
                stream.fill(batch, 64);
                if (batch.empty()) break;
                for (const auto& entry : batch) actual.push_back(entry.score);
            }
            check(actual == expected, "reader reordered, repeated or lost records");
            check(observed->visited.size() == 257, "filtering did not inspect each source record once");
            check(stream.eof() && !stream.next(), "epoch did not stop");
        }
        PrpSfenInputStream all_skipped(std::make_unique<MemorySource>(17), 42, 0,
            [](const bin::TrainingDataEntry&) { return true; });
        check(!all_skipped.next() && all_skipped.eof(), "all-skipped epoch did not finish");
        for (Index count : {1,2,7,64,65,257})
            for (Index block : {1,7,64,512})
                for (unsigned groups : {1,3,64})
                    for (bool skip : {false,true}) {
                        auto source=std::make_unique<MemorySource>(count);
                        auto* observed=source.get();
                        BufferedPrpSfenInputStream stream(std::move(source),42,7,block,groups,
                            [skip](const bin::TrainingDataEntry& e){return skip && e.score%3==0;});
                        std::vector<int> actual;
                        std::vector<std::size_t> sizes;
                        for (;;) {
                            std::vector<bin::TrainingDataEntry> batch;
                            stream.fill(batch,17);
                            if(batch.empty()) break;
                            sizes.push_back(batch.size());
                            for(const auto& e:batch) actual.push_back(e.score);
                        }
                        std::sort(actual.begin(),actual.end());
                        std::vector<int> expected;
                        for(Index i=0;i<count;++i) if(!skip || i%3) expected.push_back(int(i));
                        check(actual==expected,"buffered PRP dropped or repeated a record");
                        check(observed->visited.size()==count,"buffered filtering missed a record");
                        check(std::all_of(observed->loads.begin(),observed->loads.end(),
                            [&](Index n){return n>0 && n<=block*groups;}),"buffer size exceeded");
                        for(std::size_t i=0;i+1<sizes.size();++i)
                            check(sizes[i]==17,"short training batch at a buffer boundary");
                        check(stream.eof() && !stream.next(),"buffered epoch did not finish");
                    }
        int legacy_calls=0;
        IndexedSequentialInputStream legacy(std::make_unique<MemorySource>(3),true,
            [&](const bin::TrainingDataEntry&){return ++legacy_calls<6;},true);
        auto legacy_value=legacy.next();
        check(legacy_value && legacy_value->score==2 && legacy_calls==6,"legacy cyclic skip/reopen semantics changed");
        IndexedSequentialInputStream compact(std::make_unique<MemorySource>(3),true,
            [](const bin::TrainingDataEntry&){return true;});
        check(!compact.next(),"compact all-skipped traversal did not finish");
        std::cout << "PRP permutation, inverse, indexed backend and finite-epoch tests passed\n";
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
