#pragma once
#include "nnue_training_data_formats.h"
#include <optional>
namespace training_data {
struct BasicSfenInputStream {
    virtual std::optional<bin::TrainingDataEntry> next() = 0;
    virtual bool next_into(bin::TrainingDataEntry &entry) {
        auto value = next();
        if (!value)
            return false;
        entry = *value;
        return true;
    }
    virtual VariantPtr config() const { return nullptr; }
    virtual void fill(std::vector<bin::TrainingDataEntry> &vec, std::size_t n,
                      bin::BatchStorage *storage = nullptr) {
        for (std::size_t i = 0; i < n; ++i) {
            auto entry = storage ? storage->entry(unsigned(i)) : bin::TrainingDataEntry{};
            if (!next_into(entry))
                break;
            vec.emplace_back(entry);
        }
    }
    virtual bool eof() const = 0;
    virtual ~BasicSfenInputStream() = default;
};
} // namespace training_data
