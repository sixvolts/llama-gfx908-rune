// li_ab: byte A/B + timing for GGML_OP_LIGHTNING_INDEXER (f32 K) on one GPU, GLM-5.3-Flash shapes.
//   li_ab <out.bin|-> <n_kv> <n_batch> <n_head> [mask_pattern] [n_stream] [seed]
//   mask patterns: 0 all visible | 1 trailing 2 % -inf (unusable pools) | 2 -inf runs incl. whole 64-key blocks
//                  3 everything -inf (all-hidden skip) | 4 per-query different hidden tails
//   LI_AB_BENCH=<reps> prints the median us. Prints the fnv1a hash of the f32 output (byte comparison between builds /
//   switches: GGML_CUDA_LI_DECODE=0 = vec kernel, GGML_CUDA_LI_TILED=2 = tiled kernel).
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
#include <vector>

static uint64_t S = 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { S ^= S << 13; S ^= S >> 7; S ^= S << 17; return (uint32_t) (S >> 16); }
static float urand() { return (rnd() & 0xffffff) / 16777216.0f; }

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: li_ab <out|-> <n_kv> <n_batch> <n_head> [mask] [n_stream] [seed]\n"); return 2; }
    const char * out = argv[1];
    const int64_t n_kv = atoll(argv[2]), nb = atoll(argv[3]), nh = atoll(argv[4]);
    const int mp = argc > 5 ? atoi(argv[5]) : 1;
    const int64_t ns = argc > 6 ? atoll(argv[6]) : 1;
    if (argc > 7) { S ^= (uint64_t) atoll(argv[7]) * 0x2545F4914F6CDD1Dull; for (int i = 0; i < 4; ++i) rnd(); }
    const int64_t d = 128;

    ggml_backend_t backend = ggml_backend_cuda_init(0);
    ggml_init_params ip = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, nh, nb, ns);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, d, 1, n_kv, ns);
    ggml_tensor * w = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, nh, nb, 1, ns);
    ggml_tensor * m = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, n_kv, nb, 1, ns);
    ggml_tensor * y = ggml_lightning_indexer(ctx, q, k, w, m);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    std::vector<float> hq(ggml_nelements(q)), hk(ggml_nelements(k)), hw(ggml_nelements(w));
    for (auto & v : hq) v = urand()*2.0f - 1.0f;
    for (auto & v : hk) v = urand()*2.0f - 1.0f;
    for (auto & v : hw) v = urand() * 0.1f;
    std::vector<ggml_fp16_t> hm(ggml_nelements(m));
    const ggml_fp16_t z = ggml_fp32_to_fp16(0.0f), ni = ggml_fp32_to_fp16(-INFINITY);
    for (int64_t s = 0; s < ns; ++s) for (int64_t i = 0; i < nb; ++i) for (int64_t j = 0; j < n_kv; ++j) {
        bool hid = false;
        switch (mp) {
            case 0: hid = false; break;
            case 1: hid = j >= n_kv - std::max<int64_t>(2, n_kv/50); break;
            case 2: hid = (j / 64) % 5 == 3 || ((j / 7) % 11 == 0); break;
            case 3: hid = true; break;
            case 4: hid = j >= n_kv - (i + 1)*std::max<int64_t>(1, n_kv/20); break;
        }
        hm[(s*nb + i)*n_kv + j] = hid ? ni : z;
    }
    ggml_backend_tensor_set(q, hq.data(), 0, hq.size()*4);
    ggml_backend_tensor_set(k, hk.data(), 0, hk.size()*4);
    ggml_backend_tensor_set(w, hw.data(), 0, hw.size()*4);
    ggml_backend_tensor_set(m, hm.data(), 0, hm.size()*2);

    ggml_backend_graph_compute(backend, gf);
    std::vector<float> o(ggml_nelements(y));
    ggml_backend_tensor_get(y, o.data(), 0, o.size()*4);
    uint64_t f = 1469598103934665603ull;
    for (float v : o) { const uint8_t * p = (const uint8_t *) &v; for (int b = 0; b < 4; ++b) { f ^= p[b]; f *= 1099511628211ull; } }
    printf("n_kv %lld n_batch %lld n_head %lld mask %d n_stream %lld hash %016llx", (long long) n_kv, (long long) nb,
           (long long) nh, mp, (long long) ns, (unsigned long long) f);
    if (strcmp(out, "-") != 0) { FILE * fo = fopen(out, "wb"); fwrite(o.data(), 4, o.size(), fo); fclose(fo); }
    const char * be = getenv("LI_AB_BENCH");
    if (be && atoi(be) > 0) {
        std::vector<double> t;
        for (int i = 0; i < atoi(be); ++i) {
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
    return 0;
}
