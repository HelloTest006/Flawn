#pragma once
#include <string>
#include <cstdint>

namespace flawn::tune {

struct HardwareProfile {
    std::string cpu_brand;
    int logical_cores = 1;
    bool has_avx2 = false;
    bool has_fma3 = false;
    bool has_avx512 = false;
    size_t l1d_cache_bytes = 32768;
    size_t l2_cache_bytes = 262144;
    int best_simd_bits = 256;
    int best_unroll_factor = 8;
    int best_prefetch_dist = 256;
    size_t best_parallel_grain = 65536;
    double peak_gflops = 0.0;
};

HardwareProfile run_hardware_autotune(const std::string& project_root, bool verbose = true);

} // namespace flawn::tune
