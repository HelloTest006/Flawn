#include "flawn_autotune.hpp"
#include "flawn_runtime.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include <cstring>
#include <immintrin.h>

#if defined(__GNUC__) || defined(__clang__)
#include <cpuid.h>
#endif

namespace flawn::tune {

static std::string detect_cpu_brand() {
#if defined(__GNUC__) || defined(__clang__)
    char brand[0x40] = {0};
    unsigned int eax, ebx, ecx, edx;
    __get_cpuid(0x80000000, &eax, &ebx, &ecx, &edx);
    if (eax >= 0x80000004) {
        unsigned int* p = reinterpret_cast<unsigned int*>(brand);
        __get_cpuid(0x80000002, &p[0], &p[1], &p[2], &p[3]);
        __get_cpuid(0x80000003, &p[4], &p[5], &p[6], &p[7]);
        __get_cpuid(0x80000004, &p[8], &p[9], &p[10], &p[11]);
        std::string s(brand);
        while (!s.empty() && s.front() == ' ') s.erase(s.begin());
        return s;
    }
#endif
    return "x86_64 SIMD Processor";
}

// Micro-kernel candidate 1: 2-way unrolled AVX2 FMA
static float kernel_unroll2(const float* FLAWN_RESTRICT a, const float* FLAWN_RESTRICT b, size_t n, int prefetch) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        _mm_prefetch(reinterpret_cast<const char*>(a + i + prefetch), _MM_HINT_T0);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    acc0 = _mm256_add_ps(acc0, acc1);
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc0);
    return tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
}

// Micro-kernel candidate 2: 4-way unrolled AVX2 FMA
static float kernel_unroll4(const float* FLAWN_RESTRICT a, const float* FLAWN_RESTRICT b, size_t n, int prefetch) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 32 <= n; i += 32) {
        _mm_prefetch(reinterpret_cast<const char*>(a + i + prefetch), _MM_HINT_T0);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i +  0), _mm256_loadu_ps(b + i +  0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i +  8), _mm256_loadu_ps(b + i +  8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc0);
    return tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
}

// Micro-kernel candidate 3: 8-way unrolled AVX2 FMA (64 floats/iter)
static float kernel_unroll8(const float* FLAWN_RESTRICT a, const float* FLAWN_RESTRICT b, size_t n, int prefetch) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();
    __m256 acc4 = _mm256_setzero_ps();
    __m256 acc5 = _mm256_setzero_ps();
    __m256 acc6 = _mm256_setzero_ps();
    __m256 acc7 = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 64 <= n; i += 64) {
        _mm_prefetch(reinterpret_cast<const char*>(a + i + prefetch), _MM_HINT_T0);
        _mm_prefetch(reinterpret_cast<const char*>(b + i + prefetch), _MM_HINT_T0);
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i +  0), _mm256_loadu_ps(b + i +  0), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i +  8), _mm256_loadu_ps(b + i +  8), acc1);
        acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 16), _mm256_loadu_ps(b + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 24), _mm256_loadu_ps(b + i + 24), acc3);
        acc4 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 32), _mm256_loadu_ps(b + i + 32), acc4);
        acc5 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 40), _mm256_loadu_ps(b + i + 40), acc5);
        acc6 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 48), _mm256_loadu_ps(b + i + 48), acc6);
        acc7 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 56), _mm256_loadu_ps(b + i + 56), acc7);
    }
    acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    acc4 = _mm256_add_ps(_mm256_add_ps(acc4, acc5), _mm256_add_ps(acc6, acc7));
    acc0 = _mm256_add_ps(acc0, acc4);
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc0);
    return tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
}

HardwareProfile run_hardware_autotune(const std::string& project_root, bool verbose) {
    HardwareProfile prof;
    prof.cpu_brand = detect_cpu_brand();
    prof.logical_cores = static_cast<int>(std::thread::hardware_concurrency());
    if (prof.logical_cores <= 0) prof.logical_cores = 4;

#if defined(__GNUC__) || defined(__clang__)
    __builtin_cpu_init();
    prof.has_avx2 = __builtin_cpu_supports("avx2");
    prof.has_fma3 = __builtin_cpu_supports("fma");
    prof.has_avx512 = __builtin_cpu_supports("avx512f");
#else
    prof.has_avx2 = true;
    prof.has_fma3 = true;
#endif
    prof.best_simd_bits = prof.has_avx512 ? 512 : (prof.has_avx2 ? 256 : 128);

    if (verbose) {
        std::cout << "================================================================\n";
        std::cout << "  FLAWN HARDWARE AUTO-TUNER (Empirical ISA & Cache Sweep)\n";
        std::cout << "================================================================\n";
        std::cout << "  CPU Model      : " << prof.cpu_brand << "\n";
        std::cout << "  Logical Cores  : " << prof.logical_cores << "\n";
        std::cout << "  SIMD Features  : "
                  << (prof.has_avx2 ? "AVX2 " : "")
                  << (prof.has_fma3 ? "FMA3 " : "")
                  << (prof.has_avx512 ? "AVX-512" : "(256-bit optimal)") << "\n";
        std::cout << "----------------------------------------------------------------\n";
    }

    // Allocate 64-byte aligned arena buffers for empirical micro-kernel sweep
    constexpr size_t N = 1 << 20; // 1,048,576 floats (4 MB per vector)
    constexpr int ITERS = 60;
    rt::SlabArena arena(16 * 1024 * 1024);
    float* a = arena.alloc_array<float>(N);
    float* b = arena.alloc_array<float>(N);
    for (size_t i = 0; i < N; ++i) {
        a[i] = 1.001f;
        b[i] = 0.999f;
    }

    struct Candidate {
        int unroll;
        int prefetch;
        double elapsed_ms;
        double gflops;
    };
    std::vector<Candidate> results;
    volatile float sink = 0.0f;

    int unrolls[] = {2, 4, 8};
    int prefetches[] = {128, 256, 512};

    for (int u : unrolls) {
        for (int pf : prefetches) {
            // Warmup
            if (u == 2) sink = kernel_unroll2(a, b, N, pf);
            else if (u == 4) sink = kernel_unroll4(a, b, N, pf);
            else sink = kernel_unroll8(a, b, N, pf);

            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < ITERS; ++it) {
                if (u == 2) sink += kernel_unroll2(a, b, N, pf);
                else if (u == 4) sink += kernel_unroll4(a, b, N, pf);
                else sink += kernel_unroll8(a, b, N, pf);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            // 2 FLOPs (mul + add) per element * N * ITERS
            double total_flops = 2.0 * static_cast<double>(N) * ITERS;
            double gflops = (total_flops / (ms / 1000.0)) / 1e9;
            results.push_back({u, pf, ms, gflops});
        }
    }
    (void)sink;

    Candidate best = results.front();
    for (const auto& c : results) {
        if (verbose) {
            std::cout << "  [Probe] Unroll=" << c.unroll << "x, Prefetch=" << c.prefetch
                      << "B -> " << c.elapsed_ms << " ms (" << c.gflops << " GFLOP/s)\n";
        }
        if (c.gflops > best.gflops) {
            best = c;
        }
    }

    prof.best_unroll_factor = best.unroll;
    prof.best_prefetch_dist = best.prefetch;
    prof.peak_gflops = best.gflops;

    // Persist optimal parameters to include/flawn_tune_profile.hpp
    std::string header_path = project_root + "/include/flawn_tune_profile.hpp";
    std::ofstream ofs(header_path, std::ios::binary);
    if (ofs) {
        ofs << "#pragma once\n";
        ofs << "// Auto-generated by `flawn tune` for: " << prof.cpu_brand << "\n";
        ofs << "#define FLAWN_TUNE_SIMD_BITS " << prof.best_simd_bits << "\n";
        ofs << "#define FLAWN_TUNE_UNROLL_FACTOR " << prof.best_unroll_factor << "\n";
        ofs << "#define FLAWN_TUNE_PREFETCH_DIST " << prof.best_prefetch_dist << "\n";
        ofs << "#define FLAWN_TUNE_L1_TILE_BYTES " << prof.l1d_cache_bytes << "\n";
        ofs << "#define FLAWN_TUNE_L2_TILE_BYTES " << prof.l2_cache_bytes << "\n";
        ofs << "#define FLAWN_TUNE_PARALLEL_GRAIN " << prof.best_parallel_grain << "\n";
        ofs << "#define FLAWN_TUNE_THREADS " << prof.logical_cores << "\n";
    }

    if (verbose) {
        std::cout << "----------------------------------------------------------------\n";
        std::cout << "  WINNING HARDWARE PROFILE SAVED -> include/flawn_tune_profile.hpp\n";
        std::cout << "  Optimal Unroll : " << prof.best_unroll_factor << "x AVX2 FMA Accumulators\n";
        std::cout << "  Optimal Prefetch: " << prof.best_prefetch_dist << " Bytes\n";
        std::cout << "  Single-Core Peak: " << prof.peak_gflops << " GFLOP/s\n";
        std::cout << "================================================================\n";
    }
    return prof;
}

} // namespace flawn::tune
