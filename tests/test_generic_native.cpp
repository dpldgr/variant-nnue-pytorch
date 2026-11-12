// Native ABI/ownership/error checks; also run standalone under ASan/UBSan.
#include "../training_data_loader.cpp"
#include <chrono>
#include <filesystem>

static void check(bool value, const char *message) {
    if (!value)
        throw std::runtime_error(message);
}
struct TestFile {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("fsf-generic-" +
         std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) + ".bin");
    ~TestFile() {
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};
int main() {
    try {
        TestFile file;
        CompactTrainingRecord::Codec codec;
        CompactTrainingRecord::Entry input;
        input.kings = {0, 63};
        input.pieces = {{1, 4, 0}};
        input.score = 31999;
        input.result = 1;
        std::uint8_t bytes[8];
        std::string error;
        check(codec.reset(64, 5, 8, error) && codec.encode(input, bytes, error), "fixture encode failed");
        std::ofstream out(file.path, std::ios::binary);
        out.write(reinterpret_cast<char *>(bytes), 8);
        out.close();
        auto name = file.path.u8string();
        for (unsigned iteration = 0; iteration < 30; ++iteration) {
            auto *ctx = create_variant_context(8, 8, 6, 32, 64, 0);
            check(ctx, "context failed");
            auto *stream = create_generic_batch_stream(ctx, "HalfKAv2^", 4, name.c_str(), 2, false, false, 0,
                                                       17, 3, 8, true);
            check(stream, "stream failed");
            destroy_variant_context(ctx);
            unsigned count = 0;
            while (auto *batch = fetch_next_generic_batch(stream)) {
                check(batch->size == 1 && batch->num_inputs == 45824 && batch->max_active_features == 64,
                      "batch dimensions changed");
                check(batch->score[0] == 31999 && batch->outcome[0] == 1, "targets changed");
                check(batch->num_active_white_features == 6 && batch->num_active_black_features == 6,
                      "features changed");
                ++count;
                destroy_sparse_batch(batch);
            }
            check(count == 1 && !*get_prp_last_error(), "finite epoch failed");
            destroy_sparse_batch_stream(stream);
        }
        // Forces feature construction to throw after allocating its arrays.
        auto *ctx = create_variant_context(8, 8, 6, 32, 9, 0);
        auto *stream = create_generic_batch_stream(ctx, "HalfKAv2^", 4, name.c_str(), 2, false, false, 0, 17,
                                                   3, 8, true);
        check(stream, "error stream creation failed");
        check(!fetch_next_generic_batch(stream) && *get_prp_last_error(), "worker error did not reach ABI");
        destroy_variant_context(ctx);
        destroy_sparse_batch_stream(stream);
        check(!create_variant_context(0, 8, 6, 32, 64, 0), "invalid context accepted");
        std::cout << "Generic native ABI, context lifetime, batch ownership and worker errors passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
