// topk_ab: byte A/B + timing for GGML_OP_TOP_K on one GPU with GLM-5.3-Flash-like DSA pool scores.
//   topk_ab <out.bin|-> <ncols> <nrows> <k> <pattern> [seed]
//   patterns: 0 uniform random | 1 GLM-like (ReLU zero runs, positive tail, trailing -inf pools) | 2 heavy ties (8 values)
//             3 k-th key -inf (300 finite, rest -inf) | 4 exactly k-1 greater + one threshold tie in the last column
//             5 all equal (rank == k) | 6 GLM-like with NaN / +inf sprinkled | 7 zero runs straddling 256/1024 bounds
//   TOPK_AB_BENCH=<reps> prints the median us of the op (graph of 1 node). Prints the fnv1a hash of the index list.
//   The library under test is chosen with LD_LIBRARY_PATH; env switches (GGML_CUDA_TOPK_SCAN, ...) apply as usual.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

static uint64_t S = 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return (uint32_t) (S >> 16); }
static float urand() { return (rnd() & 0xffffff) / 16777216.0f; }

static void fill_row(float * r, int64_t n, int k, int pat) {
    const float NINF = -std::numeric_limits<float>::infinity();
    switch (pat) {
        case 0: for (int64_t i = 0; i < n; ++i) r[i] = urand()*2.0f - 1.0f; break;
        case 1: case 6: {
            const int64_t n_tail = std::max<int64_t>(2, n/50);          // unusable pools at the end
            int64_t i = 0;
            while (i < n - n_tail) {                                     // alternating zero runs and positive runs
                const int64_t z = rnd() % 40, p = 1 + rnd() % 60;
                for (int64_t j = 0; j < z && i < n - n_tail; ++j) r[i++] = 0.0f;
                for (int64_t j = 0; j < p && i < n - n_tail; ++j) r[i++] = expf(urand()*4.0f - 3.0f);
            }
            for (; i < n; ++i) r[i] = NINF;
            if (pat == 6) {
                for (int j = 0; j < 8; ++j) r[rnd() % n] = std::numeric_limits<float>::quiet_NaN();
                for (int j = 0; j < 4; ++j) r[rnd() % n] = std::numeric_limits<float>::infinity();
            }
        } break;
        case 2: { const float v[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 3.0f, 0.25f};
                  for (int64_t i = 0; i < n; ++i) r[i] = v[rnd() % 8]; } break;
        case 3: for (int64_t i = 0; i < n; ++i) r[i] = i < 300 ? urand() + 0.1f : NINF; break;
        case 4: {
            for (int64_t i = 0; i < n; ++i) r[i] = 0.0f;
            for (int j = 0; j < k - 1; ++j) r[rnd() % (n - 1)] = 2.0f + urand();   // ~k-1 distinct greater keys
            r[n - 1] = 1.0f;                                                        // the threshold, once, at the end
        } break;
        case 5: for (int64_t i = 0; i < n; ++i) r[i] = 0.0f; break;
        case 7: for (int64_t i = 0; i < n; ++i) r[i] = ((i + 128) / 256) % 3 == 0 ? 0.0f : urand(); break;
        default: fprintf(stderr, "bad pattern\n"); exit(2);
    }
}

int main(int argc, char ** argv) {
    if (argc < 6) { fprintf(stderr, "usage: topk_ab <out|-> <ncols> <nrows> <k> <pattern> [seed]\n"); return 2; }
    const char * out = argv[1];
    const int64_t nc = atoll(argv[2]), nr = atoll(argv[3]);
    const int k = atoi(argv[4]), pat = atoi(argv[5]);
    if (argc > 6) { S ^= (uint64_t) atoll(argv[6]) * 0x2545F4914F6CDD1Dull; for (int i = 0; i < 4; ++i) rnd(); }

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    ggml_init_params ip = { ggml_tensor_overhead()*8 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nc, nr);
    ggml_tensor * y = ggml_top_k(ctx, x, k);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> h(nc*nr);
    for (int64_t r = 0; r < nr; ++r) fill_row(h.data() + r*nc, nc, k, pat);
    ggml_backend_tensor_set(x, h.data(), 0, h.size()*sizeof(float));

    ggml_backend_graph_compute(backend, gf);
    std::vector<int32_t> idx(k*nr);
    ggml_backend_tensor_get(y, idx.data(), 0, idx.size()*sizeof(int32_t));
    uint64_t f = 1469598103934665603ull;
    for (int32_t v : idx) { const uint8_t * p = (const uint8_t *) &v; for (int b = 0; b < 4; ++b) { f ^= p[b]; f *= 1099511628211ull; } }

    // sanity: every index in range and the selected values are the k largest by the ordered-key relation
    int bad = 0;
    for (int64_t r = 0; r < nr && !bad; ++r) {
        for (int c = 0; c < k; ++c) { const int32_t v = idx[r*k + c]; if (v < 0 || v >= nc) { bad = 1; break; } }
    }
    // Output-order contract of the HIP radix / fused paths (ncols > 1024, k > 32, k <= ncols), which the sparse attention
    // consumes in list order: keys are compared as ordered uints (the float bits with the sign-magnitude flip); first the
    // keys strictly greater than the k-th key, in the radix gather's order - block b of bpr = min(ceil(ncols/1024), 64)
    // owns the 256-column chunks starting at b*256 + j*bpr*256 - then the first (k - n_greater) keys equal to the k-th
    // key in column order. CPU model of that order, compared position by position.
    const char * contract = "";
    if (nc > 1024 && k > 32 && k <= nc) {
        const int64_t bpr = std::min<int64_t>((nc + 1023) / 1024, 64);
        auto ord = [](float v) { uint32_t b; memcpy(&b, &v, 4); return b ^ ((uint32_t) (-(int32_t) (b >> 31)) | 0x80000000u); };
        std::vector<uint32_t> key(nc), srt(nc);
        std::vector<int32_t> exp; exp.reserve(k);
        int64_t mism = 0;
        for (int64_t r = 0; r < nr; ++r) {
            for (int64_t c = 0; c < nc; ++c) key[c] = srt[c] = ord(h[r*nc + c]);
            std::nth_element(srt.begin(), srt.begin() + (k - 1), srt.end(), std::greater<uint32_t>());
            const uint32_t kth = srt[k - 1];
            exp.clear();
            for (int64_t b = 0; b < bpr; ++b)
                for (int64_t base = b*256; base < nc; base += bpr*256)
                    for (int64_t c = base; c < std::min(base + 256, nc); ++c)
                        if (key[c] > kth) exp.push_back((int32_t) c);
            for (int64_t c = 0; c < nc && (int) exp.size() < k; ++c)
                if (key[c] == kth) exp.push_back((int32_t) c);
            for (int c = 0; c < k; ++c) mism += exp[c] != idx[r*k + c];
        }
        contract = mism ? " ORDER-MISMATCH" : " order-ok";
        if (mism) { fprintf(stderr, "order mismatches: %lld\n", (long long) mism); }
    }
    printf("ncols %lld nrows %lld k %d pat %d hash %016llx%s%s", (long long) nc, (long long) nr, k, pat,
           (unsigned long long) f, bad ? " OUT-OF-RANGE" : "", contract);
    if (contract[1] == 'O') bad = 1;
    if (strcmp(out, "-") != 0) { FILE * fo = fopen(out, "wb"); fwrite(idx.data(), 4, idx.size(), fo); fclose(fo); }

    const char * be = getenv("TOPK_AB_BENCH");
    if (be && atoi(be) > 0) {
        const int reps = atoi(be);
        std::vector<double> t;
        for (int i = 0; i < reps; ++i) {
            ggml_backend_synchronize(backend);
            const auto t0 = std::chrono::steady_clock::now();
            ggml_backend_graph_compute(backend, gf);
            ggml_backend_synchronize(backend);
            t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
        }
        std::sort(t.begin(), t.end());
        printf(" us %.1f", t[t.size()/2]);
    }
    printf("\n");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(backend);
    return bad;
}
