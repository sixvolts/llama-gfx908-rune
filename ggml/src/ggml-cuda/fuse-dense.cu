#include "fuse-dense.cuh"
#include "fuse-dense-ops.cuh"

#include "ggml-impl.h"
#include "mmf.cuh"
#include "mmvq.cuh"
#include "mmvq-q8.cuh"
#include "reduce-dpp.cuh"
#include "vecdotq.cuh"

// ---- fire counters (GGML_CUDA_FD_STATS=1 prints them at exit) ---------------------------------------

static const char * fd_names[FD_N] = { "kpool", "topk_mask", "conv_ms", "kda_gate", "kda_beta", "kda_tail" };
static int64_t fd_fired[2*FD_N];   // fired, then declined at run time (layout-dependent checks)

static void fd_print_stats() {
    fprintf(stderr, "fuse-dense fired:");
    for (int k = 0; k < FD_N; ++k) {
        fprintf(stderr, " %s %lld", fd_names[k], (long long) fd_fired[k]);
    }
    fprintf(stderr, " | declined:");
    for (int k = 0; k < FD_N; ++k) {
        fprintf(stderr, " %s %lld", fd_names[k], (long long) fd_fired[FD_N + k]);
    }
    fprintf(stderr, "\n");
}

void ggml_cuda_fd_count(int which) {   // which >= FD_N: a run-time decline of fusion which - FD_N
    static const bool stats = getenv("GGML_CUDA_FD_STATS") && atoi(getenv("GGML_CUDA_FD_STATS")) != 0;
    if (!stats) {
        return;
    }
    static bool registered = false;
    if (!registered) {
        registered = true;
        atexit(fd_print_stats);
    }
    fd_fired[which]++;
}

// ---- shared matcher helpers -------------------------------------------------------------------

static bool fd_is_view_op(const ggml_tensor * t) {
    return t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE || t->op == GGML_OP_NONE;
}
static int fd_next(const ggml_cgraph * g, int j) {
    while (j < g->n_nodes && fd_is_view_op(g->nodes[j])) { j++; }
    return j;
}
static const ggml_tensor * fd_root(const ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}
// t is `of` or a chain of RESHAPE/VIEW nodes over it at offset 0
static bool fd_is_reshape_of(const ggml_tensor * t, const ggml_tensor * of) {
    while (t != of) {
        if ((t->op != GGML_OP_RESHAPE && t->op != GGML_OP_VIEW) || t->view_offs != 0 || t->src[0] == nullptr) {
            return false;
        }
        t = t->src[0];
    }
    return true;
}
static bool fd_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
    const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}

// Nodes i..last are replaced by one launch that writes only `outs`. Every compute node in the range must be one of
// `chain`, and no tensor of the chain (or a view of one) other than the outputs may be read outside the range.
static bool fd_chain_closed(const ggml_cgraph * g, int i, int last, const ggml_tensor * const * chain, int n_chain,
        const ggml_tensor * const * outs, int n_outs) {
    auto in = [](const ggml_tensor * t, const ggml_tensor * const * set, int n) {
        for (int k = 0; k < n; ++k) { if (set[k] == t) { return true; } }
        return false;
    };
    for (int q = i; q <= last; ++q) {
        const ggml_tensor * t = g->nodes[q];
        if (!fd_is_view_op(t) && !in(t, chain, n_chain)) {
            return false;
        }
        const ggml_tensor * rt = fd_root(t);
        if (!in(rt, chain, n_chain) || in(rt, outs, n_outs)) {
            continue;
        }
        if (t->flags & GGML_TENSOR_FLAG_OUTPUT) {
            return false;
        }
        int uses = 0;
        for (int u = q + 1; u <= last; ++u) {
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                if (g->nodes[u]->src[s] == t) { uses++; }
            }
        }
        if (uses != ggml_node_get_use_count(g, q)) {
            return false;
        }
    }
    return true;
}

// ---- DSA sparse-gather cell list + mask (GGML_CUDA_FD_TOPK_MASK) ----------------------------------
//
// build_indexer + build_attn_sparse_gather (position mode) after the pool top-k:
//   sel  = CONT(TOP_K)                                     I32 [sk, n_tps, ns]
//   G1   = GET_ROWS(pool_cells [r, n_pools, ns], sel)      I32 -> top_k [r*sk, n_tps, ns]
//   idx  = CONCAT(top_k, tail_cells, 0)                    I32 [n_sel, n_tps, ns]      (output)
//   G2   = GET_ROWS(pool_bias [1, n_pools, n_tps, ns], sel), REPEAT to r -> top_k_mask
//   mpre = CONCAT(top_k_mask, tail_mask, 0)
//   G3   = GET_ROWS(kv_pos [1, n_kv, 1, ns], idx), SUB(G3, q_pos), STEP, SCALE(s, b)
//   mask = ADD(mpre, scale)                                F32 [n_sel, n_tps, ns]      (output)
// 11 launches -> 1. Copies and selects only; the SCALE is fmaf(s, x, b) like scale_f32, the ADD/SUB single ops.

struct fd_topk_mask_match {
    const ggml_tensor * topk;    // TOP_K dst (the CONT's source)
    const ggml_tensor * pc;      // pool cells
    const ggml_tensor * pb;      // pool bias view
    const ggml_tensor * tc;      // tail cells
    const ggml_tensor * tm;      // tail mask
    const ggml_tensor * kvp;     // kv positions
    const ggml_tensor * qp;      // query positions
    ggml_tensor       * idx;
    ggml_tensor       * mask;
    const ggml_tensor * mpre;
    float scale, bias;
    int r, sk, n_sel;
    int last;
};

static bool fd_topk_mask_find(const ggml_cgraph * g, int i, fd_topk_mask_match & m) {
    const int n = g->n_nodes;
    const ggml_tensor * c = g->nodes[i];
    if (c->op != GGML_OP_CONT || c->type != GGML_TYPE_I32 || !ggml_is_contiguous(c) || c->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * topk = c->src[0];
    if (topk->type != GGML_TYPE_I32 || topk->op != GGML_OP_TOP_K || !ggml_are_same_shape(topk, c) || topk->nb[0] != sizeof(int32_t)) {
        return false;
    }
    const int64_t sk = c->ne[0], n_tps = c->ne[1], ns = c->ne[2];
    if (n_tps < 1 || n_tps > 16) {   // decode-sized ubatches only
        return false;
    }
    auto at = [&](int j, ggml_op op) -> ggml_tensor * {
        return j < n && g->nodes[j]->op == op ? g->nodes[j] : nullptr;
    };
    int j = fd_next(g, i + 1);
    ggml_tensor * g1 = at(j, GGML_OP_GET_ROWS);
    if (!g1 || g1->type != GGML_TYPE_I32 || !fd_is_reshape_of(g1->src[1], c)) { return false; }
    const ggml_tensor * pc = g1->src[0];
    const int64_t r = pc->ne[0];
    if (pc->type != GGML_TYPE_I32 || pc->nb[0] != sizeof(int32_t) || pc->ne[2] != ns || pc->ne[3] != 1 || r < 1 ||
            g1->ne[0] != r || g1->ne[1] != sk*n_tps || g1->ne[2] != ns || !ggml_is_contiguous(g1)) { return false; }

    j = fd_next(g, j + 1);
    ggml_tensor * idx = at(j, GGML_OP_CONCAT);
    if (!idx || idx->type != GGML_TYPE_I32 || ggml_get_op_params_i32(idx, 0) != 0 || !fd_is_reshape_of(idx->src[0], g1) ||
            !ggml_is_contiguous(idx) || idx->ne[1] != n_tps || idx->ne[2] != ns || idx->ne[3] != 1) { return false; }
    const ggml_tensor * tc = idx->src[1];
    const int64_t n_sel = idx->ne[0];
    if (tc->type != GGML_TYPE_I32 || tc->nb[0] != sizeof(int32_t) || tc->ne[0] != n_sel - r*sk || tc->ne[1] != n_tps || tc->ne[2] != ns) { return false; }

    j = fd_next(g, j + 1);
    ggml_tensor * g2 = at(j, GGML_OP_GET_ROWS);
    if (!g2 || g2->type != GGML_TYPE_F32 || g2->src[1] != c) { return false; }
    const ggml_tensor * pb = g2->src[0];
    if (pb->type != GGML_TYPE_F32 || pb->ne[0] != 1 || pb->ne[1] != pc->ne[1] || pb->ne[2] != n_tps || pb->ne[3] != ns) { return false; }

    j = fd_next(g, j + 1);
    ggml_tensor * rep = at(j, GGML_OP_REPEAT);
    if (!rep || rep->src[0] != g2 || rep->ne[0] != r || rep->ne[1] != sk || rep->ne[2] != n_tps || rep->ne[3] != ns) { return false; }

    j = fd_next(g, j + 1);
    ggml_tensor * mpre = at(j, GGML_OP_CONCAT);
    if (!mpre || mpre->type != GGML_TYPE_F32 || ggml_get_op_params_i32(mpre, 0) != 0 || !fd_is_reshape_of(mpre->src[0], rep) ||
            !ggml_are_same_shape(mpre, idx)) { return false; }
    const ggml_tensor * tm = mpre->src[1];
    if (tm->type != GGML_TYPE_F32 || tm->nb[0] != sizeof(float) || !ggml_are_same_shape(tm, tc)) { return false; }

    j = fd_next(g, j + 1);
    ggml_tensor * g3 = at(j, GGML_OP_GET_ROWS);
    if (!g3 || g3->type != GGML_TYPE_F32 || !fd_is_reshape_of(g3->src[1], idx)) { return false; }
    const ggml_tensor * kvp = g3->src[0];
    if (kvp->type != GGML_TYPE_F32 || kvp->ne[0] != 1 || kvp->ne[2] != 1 || kvp->ne[3] != ns ||
            g3->src[1]->ne[0] != n_sel*n_tps || g3->src[1]->ne[1] != 1 || g3->src[1]->ne[2] != ns) { return false; }

    j = fd_next(g, j + 1);
    ggml_tensor * sub = at(j, GGML_OP_SUB);
    if (!sub || sub->type != GGML_TYPE_F32 || !fd_is_reshape_of(sub->src[0], g3) || !ggml_are_same_shape(sub, idx)) { return false; }
    const ggml_tensor * qp = sub->src[1];
    if (qp->type != GGML_TYPE_F32 || qp->ne[3] != 1 || n_sel % qp->ne[0] != 0 || n_tps % qp->ne[1] != 0 || ns % qp->ne[2] != 0) { return false; }

    j = fd_next(g, j + 1);
    ggml_tensor * stp = at(j, GGML_OP_UNARY);
    if (!stp || ggml_get_unary_op(stp) != GGML_UNARY_OP_STEP || stp->src[0] != sub || stp->type != GGML_TYPE_F32) { return false; }

    j = fd_next(g, j + 1);
    ggml_tensor * sc = at(j, GGML_OP_SCALE);
    if (!sc || sc->src[0] != stp || sc->type != GGML_TYPE_F32) { return false; }

    j = fd_next(g, j + 1);
    ggml_tensor * add = at(j, GGML_OP_ADD);
    if (!add || add->src[0] != mpre || add->src[1] != sc || !ggml_is_contiguous(add) || !ggml_are_same_shape(add, idx)) { return false; }

    const ggml_tensor * chain[] = { c, g1, idx, g2, rep, mpre, g3, sub, stp, sc, add };
    const ggml_tensor * outs[]  = { idx, add };
    if (!fd_chain_closed(g, i, j, chain, 11, outs, 2)) { return false; }

    m.topk = topk; m.pc = pc; m.pb = pb; m.tc = tc; m.tm = tm; m.kvp = kvp; m.qp = qp;
    m.idx = idx; m.mask = add; m.mpre = mpre;
    memcpy(&m.scale, (const float *) sc->op_params + 0, sizeof(float));
    memcpy(&m.bias,  (const float *) sc->op_params + 1, sizeof(float));
    m.r = (int) r; m.sk = (int) sk; m.n_sel = (int) n_sel;
    m.last = j;
    return true;
}

static __global__ void fd_topk_mask_kernel(
        const int32_t * __restrict__ topk, const int64_t tk_s1, const int64_t tk_s2,
        const char * __restrict__ pc, const int64_t pc_nb1, const int64_t pc_nb2,
        const char * __restrict__ pb, const int64_t pb_nb1, const int64_t pb_nb2, const int64_t pb_nb3,
        const char * __restrict__ tc, const int64_t tc_nb1, const int64_t tc_nb2,
        const char * __restrict__ tm, const int64_t tm_nb1, const int64_t tm_nb2,
        const char * __restrict__ kvp, const int64_t kvp_nb1, const int64_t kvp_nb3,
        const char * __restrict__ qp, const int qp_ne0, const int qp_ne1, const int qp_ne2,
        const int64_t qp_nb0, const int64_t qp_nb1, const int64_t qp_nb2,
        int32_t * __restrict__ idx, float * __restrict__ mask,
        const int r, const int sk, const int n_sel, const int n_tps, const int64_t n_all,
        const float scale, const float bias) {
#pragma clang fp contract(off)
    for (int64_t e = (int64_t) blockIdx.x*blockDim.x + threadIdx.x; e < n_all; e += (int64_t) gridDim.x*blockDim.x) {
        const int i = (int) (e % n_sel);
        const int t = (int) ((e / n_sel) % n_tps);
        const int s = (int) (e / ((int64_t) n_sel*n_tps));
        int32_t cell;
        float   m;
        if (i < r*sk) {
            const int32_t p = topk[i/r + t*tk_s1 + s*tk_s2];
            cell = *(const int32_t *) (pc + (i % r)*sizeof(int32_t) + p*pc_nb1 + s*pc_nb2);
            m    = *(const float *) (pb + p*pb_nb1 + t*pb_nb2 + s*pb_nb3);
        } else {
            const int it = i - r*sk;
            cell = *(const int32_t *) (tc + it*sizeof(int32_t) + t*tc_nb1 + s*tc_nb2);
            m    = *(const float *) (tm + it*sizeof(float) + t*tm_nb1 + s*tm_nb2);
        }
        idx[e] = cell;
        const float cp = *(const float *) (kvp + (int64_t) cell*kvp_nb1 + s*kvp_nb3);
        const float q  = *(const float *) (qp + (i % qp_ne0)*qp_nb0 + (t % qp_ne1)*qp_nb1 + (s % qp_ne2)*qp_nb2);
        const float d  = cp - q;
        const float st = d > 0.0f;
        mask[e] = m + fmaf(scale, st, bias);
    }
}

int ggml_cuda_fd_try_topk_mask(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = ggml_cuda_fd_on("GGML_CUDA_FD_TOPK_MASK");
    if (!on) {
        return 0;
    }
    fd_topk_mask_match m;
    if (!fd_topk_mask_find(cgraph, i, m)) {
        return 0;
    }
    // the kernel reads every input while it writes idx and mask
    const ggml_tensor * ins[] = { m.topk, m.pc, m.pb, m.tc, m.tm, m.kvp, m.qp };
    for (const ggml_tensor * t : ins) {
        if (fd_overlap(t, m.idx) || fd_overlap(t, m.mask)) {
            static bool logged = false;
            if (!logged) {
                logged = true;
                GGML_LOG_INFO("%s: %s overlaps input %s (%s), unfused path (logged once)\n", __func__, m.mask->name, t->name,
                        fd_overlap(t, m.idx) ? "idx" : "mask");
            }
            ggml_cuda_fd_count(FD_N + FD_TOPK_MASK);
            return 0;
        }
    }
    const int64_t n_tps = m.idx->ne[1], ns = m.idx->ne[2];
    const int64_t n_all = (int64_t) m.n_sel*n_tps*ns;
    const int nth = 256;
    const int nblk = (int) std::min<int64_t>((n_all + nth - 1)/nth, 960);   // grid-stride; stays off the 1024-block cliff
    fd_topk_mask_kernel<<<nblk, nth, 0, ctx.stream()>>>(
        (const int32_t *) m.topk->data, m.topk->nb[1]/sizeof(int32_t), m.topk->nb[2]/sizeof(int32_t),
        (const char *) m.pc->data, m.pc->nb[1], m.pc->nb[2],
        (const char *) m.pb->data, m.pb->nb[1], m.pb->nb[2], m.pb->nb[3],
        (const char *) m.tc->data, m.tc->nb[1], m.tc->nb[2],
        (const char *) m.tm->data, m.tm->nb[1], m.tm->nb[2],
        (const char *) m.kvp->data, m.kvp->nb[1], m.kvp->nb[3],
        (const char *) m.qp->data, (int) m.qp->ne[0], (int) m.qp->ne[1], (int) m.qp->ne[2], m.qp->nb[0], m.qp->nb[1], m.qp->nb[2],
        (int32_t *) m.idx->data, (float *) m.mask->data,
        m.r, m.sk, m.n_sel, (int) n_tps, n_all, m.scale, m.bias);
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_fd_count(FD_TOPK_MASK);
    return m.last - i;
}

// ---- KDA conv-input assembly, several sequences (GGML_CUDA_FD_CONV_MS) ---------------------------------
//
// build_conv_state at n_seqs > 1 (the n_seqs == 1 case is ggml_cuda_try_fuse_kda_conv_rows):
//   c1 = CONCAT(q, k, 0), [the v MUL_MAT], c2 = CONCAT(c1, v, 0)          [C, nst*n_seqs]
//   c3 = CONCAT(states [ns, C, n_seqs], TRANSPOSE(RESHAPE(c2)), 0)        [ns + nst, C, n_seqs]   (output)
//   K x CPY(VIEW(c3, ns, C, n_seqs at column s_idx), slot rows [ns*C, n_seqs])
// 3 + K launches -> 1 (plus the v GEMV, which runs first as in the graph). Copies only.

#define FD_CONV_MS_MAX_DST 8

struct fd_conv_ms_match {
    const ggml_tensor * q, * k, * v, * st;
    ggml_tensor * c1, * c2, * c3, * mid;
    int n_cpy;
    ggml_tensor * cpy[FD_CONV_MS_MAX_DST];
    int s_idx[FD_CONV_MS_MAX_DST];
    int last;
};

static bool fd_conv_ms_find(const ggml_cgraph * g, int i, fd_conv_ms_match & m) {
    const int n = g->n_nodes;
    auto concat0 = [](const ggml_tensor * t) {
        return t->op == GGML_OP_CONCAT && ggml_get_op_params_i32(t, 0) == 0 && t->type == GGML_TYPE_F32 &&
               !(t->flags & GGML_TENSOR_FLAG_OUTPUT);
    };
    auto cols = [](const ggml_tensor * t, int64_t nT) {
        return t->type == GGML_TYPE_F32 && t->ne[1] == nT && t->ne[2] == 1 && t->ne[3] == 1 && t->nb[0] == sizeof(float);
    };
    ggml_tensor * c1 = g->nodes[i];
    if (!concat0(c1)) {
        return false;
    }
    const ggml_tensor * q = c1->src[0];
    const ggml_tensor * k = c1->src[1];
    const int64_t nT = q->ne[1];
    if (!cols(q, nT) || !cols(k, nT)) {
        return false;
    }
    int j = fd_next(g, i + 1);
    m.mid = nullptr;
    if (j < n && g->nodes[j]->op == GGML_OP_MUL_MAT) {
        m.mid = g->nodes[j];
        if (m.mid->src[0] == c1 || m.mid->src[1] == c1) {
            return false;
        }
        j = fd_next(g, j + 1);
    }
    if (j >= n || !concat0(g->nodes[j])) {
        return false;
    }
    ggml_tensor * c2 = g->nodes[j];
    const ggml_tensor * v = c2->src[1];
    if (c2->src[0] != c1 || !cols(v, nT) || (m.mid && v != m.mid)) {
        return false;
    }
    const int64_t C = c2->ne[0];
    const int j3 = fd_next(g, j + 1);
    if (j3 >= n || !concat0(g->nodes[j3])) {
        return false;
    }
    ggml_tensor * c3 = g->nodes[j3];
    const ggml_tensor * st = c3->src[0];
    const ggml_tensor * xt = c3->src[1];
    const int64_t ns = st->ne[0], n_seqs = st->ne[2];
    if (n_seqs < 2 || nT % n_seqs != 0 || C > INT_MAX/2) {
        return false;
    }
    const int64_t nst = nT / n_seqs;
    if (nst < 1 || nst > 8) {   // decode-sized
        return false;
    }
    // x^T: element (t, c, s) at c2 + ((s*nst + t)*C + c)*4, through view nodes only
    if (xt->view_src != c2 || xt->view_offs != 0 || xt->ne[0] != nst || xt->ne[1] != C || xt->ne[2] != n_seqs || xt->ne[3] != 1 ||
            xt->nb[0] != (size_t) C*sizeof(float) || xt->nb[1] != sizeof(float) || xt->nb[2] != (size_t) (nst*C)*sizeof(float)) {
        return false;
    }
    if (st->type != GGML_TYPE_F32 || ns < 1 || ns > 8 || st->ne[1] != C || st->ne[3] != 1 || st->nb[0] != sizeof(float) ||
            !ggml_is_contiguous(c3) || c3->ne[0] != ns + nst || c3->ne[1] != C || c3->ne[2] != n_seqs) {
        return false;
    }
    m.q = q; m.k = k; m.v = v; m.st = st; m.c1 = c1; m.c2 = c2; m.c3 = c3;
    m.n_cpy = 0;
    m.last  = j3;
    for (int p = fd_next(g, j3 + 1); p < n && m.n_cpy < FD_CONV_MS_MAX_DST; p = fd_next(g, p + 1)) {
        ggml_tensor * cp = g->nodes[p];
        if (cp->op != GGML_OP_CPY) {
            break;
        }
        const ggml_tensor * s = cp->src[0];
        const ggml_tensor * d = cp->src[1];
        if (s->view_src != c3 || s->type != GGML_TYPE_F32 || s->ne[0] != ns || s->ne[1] != C || s->ne[2] != n_seqs || s->ne[3] != 1 ||
                s->nb[0] != sizeof(float) || s->nb[1] != c3->nb[1] || s->nb[2] != c3->nb[2] || s->view_offs % sizeof(float) != 0 ||
                (int64_t) (s->view_offs / sizeof(float)) + ns > ns + nst) {
            break;
        }
        if (d->type != GGML_TYPE_F32 || d->ne[0] != ns*C || d->ne[1] != n_seqs || d->ne[2] != 1 || d->ne[3] != 1 ||
                d->nb[0] != sizeof(float) || d->nb[1] % sizeof(float) != 0) {
            break;
        }
        m.cpy[m.n_cpy]   = cp;
        m.s_idx[m.n_cpy] = (int) (s->view_offs / sizeof(float));
        m.n_cpy++;
        m.last = p;
    }
    const ggml_tensor * chain[3 + 1 + FD_CONV_MS_MAX_DST] = { c1, c2, c3 };
    int n_chain = 3;
    if (m.mid) { chain[n_chain++] = m.mid; }
    for (int a = 0; a < m.n_cpy; ++a) { chain[n_chain++] = m.cpy[a]; }
    const ggml_tensor * outs[1 + FD_CONV_MS_MAX_DST] = { c3 };
    int n_outs = 1;
    if (m.mid) { outs[n_outs++] = m.mid; }
    for (int a = 0; a < m.n_cpy; ++a) { outs[n_outs++] = m.cpy[a]; }
    return fd_chain_closed(g, i, m.last, chain, n_chain, outs, n_outs);
}

struct fd_conv_ms_dst {
    float * d[FD_CONV_MS_MAX_DST];
    int     s_idx[FD_CONV_MS_MAX_DST];
};

static __global__ void fd_conv_ms_kernel(
        const char * __restrict__ q, const char * __restrict__ k, const char * __restrict__ v,
        const int64_t nbq1, const int64_t nbk1, const int64_t nbv1,
        const int d0, const int d1, const int C, const int nst,
        const char * __restrict__ st, const int64_t nbs0, const int64_t nbs1, const int64_t nbs2, const int ns,
        float * __restrict__ ci, const int n_dst, const fd_conv_ms_dst out, const int64_t dst_s1) {
    const int c = blockIdx.x*blockDim.x + threadIdx.x;
    const int s = blockIdx.y;
    if (c >= C) {
        return;
    }
    const char * src;
    int64_t nb1;
    if (c < d0) {
        src = q + (int64_t) c*sizeof(float);        nb1 = nbq1;
    } else if (c < d1) {
        src = k + (int64_t) (c - d0)*sizeof(float); nb1 = nbk1;
    } else {
        src = v + (int64_t) (c - d1)*sizeof(float); nb1 = nbv1;
    }
    const int ne0 = ns + nst;
    float row[16];   // ns <= 8, nst <= 8 (matcher)
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        if (j < ns) {
            row[j] = *(const float *) (st + j*nbs0 + (int64_t) c*nbs1 + s*nbs2);
        }
    }
    for (int t = 0; t < nst; ++t) {
        row[ns + t] = *(const float *) (src + ((int64_t) s*nst + t)*nb1);
    }
    float * o = ci + ((int64_t) s*C + c)*ne0;
    for (int j = 0; j < ne0; ++j) {
        o[j] = row[j];
    }
    for (int kk = 0; kk < n_dst; ++kk) {
        float * d = out.d[kk] + s*dst_s1 + (int64_t) c*ns;
        for (int j = 0; j < ns; ++j) {
            d[j] = row[out.s_idx[kk] + j];
        }
    }
}

int ggml_cuda_fd_try_conv_ms(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, ggml_cuda_fd_forward_t fwd) {
    static const bool on = ggml_cuda_fd_on("GGML_CUDA_FD_CONV_MS");
    if (!on) {
        return 0;
    }
    fd_conv_ms_match m;
    if (!fd_conv_ms_find(cgraph, i, m)) {
        return 0;
    }
    // the kernel reads q/k/v/states while it writes conv_input and the slots: no overlap at all (states come from the
    // build_rs gather at n_seqs > 1); the slots' row stride must be the same for all of them
    bool ok = !fd_overlap(m.c3, m.q) && !fd_overlap(m.c3, m.k) && !fd_overlap(m.c3, m.v) && !fd_overlap(m.c3, m.st) &&
              !(m.mid && (fd_overlap(m.v, m.q) || fd_overlap(m.v, m.k)));
    for (int a = 0; ok && a < m.n_cpy; ++a) {
        const ggml_tensor * d = m.cpy[a]->src[1];
        ok = !fd_overlap(d, m.q) && !fd_overlap(d, m.k) && !fd_overlap(d, m.v) && !fd_overlap(d, m.c3) && !fd_overlap(d, m.st) &&
             d->nb[1] == m.cpy[0]->src[1]->nb[1];
        for (int b = 0; ok && b < a; ++b) {
            ok = !fd_overlap(d, m.cpy[b]->src[1]);
        }
    }
    if (!ok) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: %s overlaps q/k/v/states or a slot, unfused path (logged once)\n", __func__, m.c3->name);
        }
        ggml_cuda_fd_count(FD_N + FD_CONV_MS);
        return 0;
    }
    if (m.mid && !fwd(ctx, m.mid)) {
        GGML_ABORT("%s: op not supported %s (%s)", __func__, m.mid->name, ggml_op_name(m.mid->op));
    }
    fd_conv_ms_dst out = {};
    for (int a = 0; a < m.n_cpy; ++a) {
        out.d[a]     = (float *) m.cpy[a]->src[1]->data;
        out.s_idx[a] = m.s_idx[a];
    }
    const int d0 = (int) m.q->ne[0];
    const int d1 = d0 + (int) m.k->ne[0];
    const int C  = (int) m.c3->ne[1];
    const int ns = (int) m.st->ne[0];
    const int n_seqs = (int) m.st->ne[2];
    const int nst = (int) (m.q->ne[1] / n_seqs);
    const int64_t dst_s1 = m.n_cpy > 0 ? (int64_t) (m.cpy[0]->src[1]->nb[1]/sizeof(float)) : 0;
    const dim3 grid((C + 255)/256, n_seqs, 1);
    fd_conv_ms_kernel<<<grid, 256, 0, ctx.stream()>>>(
        (const char *) m.q->data, (const char *) m.k->data, (const char *) m.v->data, m.q->nb[1], m.k->nb[1], m.v->nb[1],
        d0, d1, C, nst, (const char *) m.st->data, m.st->nb[0], m.st->nb[1], m.st->nb[2], ns,
        (float *) m.c3->data, m.n_cpy, out, dst_s1);
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_fd_count(FD_CONV_MS);
    return m.last - i;
}

// ---- KDA gate / output tail (GGML_CUDA_FD_KDA_GATE, GGML_CUDA_FD_KDA_TAIL) ----------------------------
//
// Both replay a K = 128 Q8_0 GEMV exactly as mmvq_q8_0_smallk<4, ncols, R> computes it (16 lanes per row: lane l
// owns quant block l/4 at int offset 2*(l%4), one vec_dot_q8_0_q8_1, xor butterfly 8/4/2/1), on a q8_1 copy of the
// activation built in LDS exactly as quantize_q8_1 builds it (32-lane groups, warp_reduce_max/sum<32>, d = amax/127,
// q = roundf(x/d)). The elementwise tails are written as the original kernels compute them, with contraction off and
// fmaf where scale_f32 (scale*x + bias) is contracted.

static bool fd_mmvq_q8_k128(const ggml_backend_cuda_context & ctx, const ggml_tensor * mm, int64_t nc) {
    static const bool smallk = !getenv("GGML_MMVQ_Q8_SMALLK") || atoi(getenv("GGML_MMVQ_Q8_SMALLK")) != 0;
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    return smallk && GGML_CUDA_CC_IS_CDNA(cc) && mm->op == GGML_OP_MUL_MAT && w->type == GGML_TYPE_Q8_0 && w->ne[0] == 128 &&
        w->ne[2] == 1 && w->ne[3] == 1 && ggml_is_contiguous(w) && w->buffer && ggml_cuda_fd_buffer_is_cuda(w->buffer) &&
        ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        x->type == GGML_TYPE_F32 && x->ne[0] == 128 && x->ne[1] == nc && x->ne[2] == 1 && x->ne[3] == 1 && x->nb[0] == sizeof(float) &&
        mm->type == GGML_TYPE_F32 && ggml_is_contiguous(mm) && mm->ne[1] == nc &&
        nc >= 1 && nc <= 8 && !ggml_cuda_should_use_mmf(w->type, cc, warp_size, w->ne, w->nb, (int) nc, false) &&
        ggml_cuda_should_use_mmvq(w->type, cc, nc);
}

// quantize_q8_1 of n = 128 floats per column (cols x[j*s1 + i]) into yq[j*4 + i/32]; nthreads a multiple of 32
template <int ncols>
static __device__ __forceinline__ void fd_q8_1_128(const float * x, const int64_t s1, block_q8_1 * yq, const int tid, const int nthreads) {
    for (int e = tid; e < ncols*128; e += nthreads) {
        const int j  = e / 128;
        const int i0 = e % 128;
        const float xi = x[j*s1 + i0];
        float amax = fabsf(xi);
        float sum  = xi;
        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
        block_q8_1 & b = yq[j*4 + i0/QK8_1];
        b.qs[i0 % QK8_1] = q;
        if (i0 % QK8_1 == 0) {
            b.ds = make_half2(d, sum);
        }
    }
}

// one K = 128 row for all columns: the smallk lane terms and butterfly; valid at lane l == 0 of the 16-lane group.
// (issuing the weight loads before the activation quantize measured slower: 19 -> 24 us tail, 11 -> 12.7 us gate)
struct fd_w_k128 {
    int  v[VDR_Q8_0_Q8_1_MMVQ];
    half dw;
};
static __device__ __forceinline__ fd_w_k128 fd_load_k128(const block_q8_0 * wrow, const int l) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;
    const int kbx = l/(QI8_0/vdr);
    const int kqs = vdr*(l % (QI8_0/vdr));
    const block_q8_0 * bq = wrow + kbx;
    fd_w_k128 w;
    int2 q;
    __builtin_memcpy(&q, bq->qs + 4*kqs, sizeof(q));
    w.v[0] = q.x;
    w.v[1] = q.y;
    w.dw   = bq->d;
    return w;
}
template <int ncols>
static __device__ __forceinline__ void fd_row_k128(const fd_w_k128 & w, const block_q8_1 * yq, const int l, float * tmp) {
    constexpr int vdr = VDR_Q8_0_Q8_1_MMVQ;   // 2
    const int kbx = l/(QI8_0/vdr);
    const int kqs = vdr*(l % (QI8_0/vdr));
#pragma unroll
    for (int j = 0; j < ncols; ++j) {
        const block_q8_1 * by = yq + j*4 + kbx;
        int u[vdr];
        u[0] = get_int_b4(by->qs, kqs);
        u[1] = get_int_b4(by->qs, kqs + 1);
        tmp[j] = 0.0f;
        tmp[j] += vec_dot_q8_0_q8_1_impl<float, vdr>(w.v, u, w.dw, __low2half(by->ds));
    }
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
#pragma unroll
        for (int j = 0; j < ncols; ++j) {
            tmp[j] += ggml_cuda_shfl_xor_td<64>(tmp[j], offset);
        }
    }
}

// gate: dst = s2*sigmoid(s1*((f_b(q8(fa)) + dt) * a[row/hd]) + b1) + b2; NWV waves per block, 8 rows per wave (smallk
// R = 2 passes of 4 rows), the q8_1 of fa built once per block
template <int ncols, int NWV>
static __global__ void __launch_bounds__(64*NWV, 1) fd_kda_gate_kernel(
        const float * __restrict__ fa, const int64_t fa_s1, const void * __restrict__ vx, const int stride_row_x,
        const float * __restrict__ dt, const float * __restrict__ a, const int hd,
        float * __restrict__ dst, const int64_t dst_s1, const int nrows,
        const float s1, const float b1, const float s2, const float b2) {
    __shared__ block_q8_1 yq[ncols*4];
    const int tid = threadIdx.x;
    const int wv  = tid/64;
    const int sub = (tid % 64)/16;
    const int l   = tid % 16;
    fd_q8_1_128<ncols>(fa, fa_s1, yq, tid, 64*NWV);
    __syncthreads();
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = (blockIdx.x*NWV + wv)*8 + r*4 + sub;
        if (row >= nrows) {
            continue;
        }
        float tmp[ncols];
        fd_row_k128<ncols>(fd_load_k128((const block_q8_0 *) vx + (int64_t) row*stride_row_x, l), yq, l, tmp);
        if (l == 0) {
#pragma clang fp contract(off)
            const float dtr = dt[row];
            const float ah  = a[row / hd];
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                float x = tmp[j] + dtr;
                x = x * ah;
                x = fmaf(s1, x, b1);
                x = 1.0f / (1.0f + expf(-x));
                x = fmaf(s2, x, b2);
                dst[j*dst_s1 + row] = x;
            }
        }
    }
}

// tail, BPH blocks of 128 threads per head h (hd = 128 channels); block b of the head owns rows e0 .. e0 + 128/BPH:
//   nrm  = rms_norm_f32(o[h, t]) * w   (128 squares at lanes 0..127; block_reduce<SUM, 128> sums the same nonzero
//                                       warp partials in the same tree as the graph's 256-thread rms_norm_f32)
//   gate = g_b(q8(ga)) rows h*128 + e
//   out  = sigmoid(gate) * nrm  -> dst (the cont the wo GEMV reads) and, if q8 != nullptr, its q8_1 blocks
template <int ncols, int BPH>
static __global__ void __launch_bounds__(128, 1) fd_kda_tail_kernel(
        const char * __restrict__ o, const int64_t o_nb1, const int64_t o_nb2, const int64_t o_nb3, const int o_ne2,
        const float * __restrict__ w, const float eps,
        const float * __restrict__ ga, const int64_t ga_s1, const void * __restrict__ vx, const int stride_row_x,
        float * __restrict__ dst, const int64_t dst_s1, block_q8_1 * __restrict__ q8, const int64_t q8_s1) {
    constexpr int bph = BPH;
    constexpr int nr  = 128 / BPH;
    __shared__ float nrm[ncols][128];
    __shared__ float gt[ncols][128];
    __shared__ float s_sum[32];
    __shared__ block_q8_1 yq[ncols*4];
    const int h   = blockIdx.x / bph;
    const int e0  = (blockIdx.x % bph)*nr;
    const int tid = threadIdx.x;
    const int wv  = tid/64;
    const int sub = (tid % 64)/16;
    const int l   = tid % 16;
    // rms_norm of the head per column (one barrier per column measured faster than batching the reductions)
    for (int t = 0; t < ncols; ++t) {
#pragma clang fp contract(off)
        float tmp = 0.0f;
        const float xi = *(const float *) (o + h*o_nb1 + (t % o_ne2)*o_nb2 + (t / o_ne2)*o_nb3 + tid*sizeof(float));
        tmp += xi * xi;
        tmp = block_reduce<block_reduce_method::SUM, 128>(tmp, s_sum);
        const float mean  = tmp / 128;
        const float scale = rsqrtf(mean + eps);
        nrm[t][tid] = scale * xi * w[tid];
        __syncthreads();   // s_sum is reused by the next row
    }
    fd_q8_1_128<ncols>(ga, ga_s1, yq, tid, 128);
    __syncthreads();
#pragma unroll
    for (int p = 0; p < nr/8; ++p) {
        const int e   = e0 + wv*(nr/2) + p*4 + sub;
        const int row = h*128 + e;
        float tmp[ncols];
        fd_row_k128<ncols>(fd_load_k128((const block_q8_0 *) vx + (int64_t) row*stride_row_x, l), yq, l, tmp);
        if (l == 0) {
#pragma clang fp contract(off)
#pragma unroll
            for (int j = 0; j < ncols; ++j) {
                const float x = (1.0f / (1.0f + expf(-tmp[j]))) * nrm[j][e];
                gt[j][e] = x;
                dst[j*dst_s1 + row] = x;
            }
        }
    }
    if (q8 == nullptr) {
        return;
    }
    __syncthreads();
    // quantize_q8_1 of the block's outputs per column (nr >= 32: whole q8_1 blocks)
    for (int k = tid; k < ncols*nr; k += 128) {
        const int j  = k / nr;
        const int i0 = e0 + k % nr;
        const float xi = gt[j][i0];
        float amax = fabsf(xi);
        float sum  = xi;
        amax = warp_reduce_max<QK8_1>(amax);
        sum  = warp_reduce_sum<QK8_1>(sum);
        const float  d = amax / 127.0f;
        const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);
        block_q8_1 & b = q8[j*q8_s1 + h*4 + i0/QK8_1];
        b.qs[i0 % QK8_1] = q;
        if (i0 % QK8_1 == 0) {
            b.ds = make_half2(d, sum);
        }
    }
}

struct fd_kda_gate_match {
    const ggml_tensor * fa, * wfb, * dt, * a;
    ggml_tensor * out;
    float s1, b1, s2, b2;
    int last;
};

// [i] MUL_MAT(f_b, fa) [ADD(., dt)] RESHAPE [RESHAPE(a)] MUL(., a) SCALE SIGMOID SCALE
static bool fd_kda_gate_find(const ggml_backend_cuda_context & ctx, const ggml_cgraph * g, int i, fd_kda_gate_match & m) {
    const int n = g->n_nodes;
    const ggml_tensor * mm = g->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const int64_t nc = mm->ne[1];
    // 1 column: f_b + add runs as the fused mul_mat_vec_q with bias (different kernel), left alone
    if (nc < 2 || !fd_mmvq_q8_k128(ctx, mm, nc) || mm->ne[0] % 8 != 0) {
        return false;
    }
    auto at = [&](int j, ggml_op op) -> ggml_tensor * {
        return j < n && g->nodes[j]->op == op ? g->nodes[j] : nullptr;
    };
    int j = fd_next(g, i + 1);
    ggml_tensor * add = at(j, GGML_OP_ADD);
    if (!add || add->src[0] != mm || add->type != GGML_TYPE_F32 || !ggml_are_same_shape(add, mm)) { return false; }
    const ggml_tensor * dt = add->src[1];
    if (dt->type != GGML_TYPE_F32 || !ggml_is_contiguous(dt) || ggml_nelements(dt) != mm->ne[0] || dt->ne[0] != mm->ne[0]) { return false; }
    j = fd_next(g, j + 1);
    ggml_tensor * mul = at(j, GGML_OP_MUL);
    if (!mul || !fd_is_reshape_of(mul->src[0], add) || mul->type != GGML_TYPE_F32 || ggml_nelements(mul) != ggml_nelements(add) ||
            !ggml_is_contiguous(mul)) { return false; }
    const ggml_tensor * a = mul->src[1];
    const int64_t hd = mul->ne[0];
    if (a->type != GGML_TYPE_F32 || !ggml_is_contiguous(a) || a->ne[0] != 1 || a->ne[1] != mul->ne[1] || ggml_nelements(a) != a->ne[1] ||
            hd*mul->ne[1] != mm->ne[0] || mul->ne[2]*mul->ne[3] != nc) { return false; }
    j = fd_next(g, j + 1);
    ggml_tensor * sc1 = at(j, GGML_OP_SCALE);
    if (!sc1 || sc1->src[0] != mul) { return false; }
    j = fd_next(g, j + 1);
    ggml_tensor * sig = at(j, GGML_OP_UNARY);
    if (!sig || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || sig->src[0] != sc1) { return false; }
    j = fd_next(g, j + 1);
    ggml_tensor * sc2 = at(j, GGML_OP_SCALE);
    if (!sc2 || sc2->src[0] != sig || !ggml_is_contiguous(sc2) || sc2->type != GGML_TYPE_F32) { return false; }
    const ggml_tensor * chain[] = { mm, add, mul, sc1, sig, sc2 };
    const ggml_tensor * outs[]  = { sc2 };
    if (!fd_chain_closed(g, i, j, chain, 6, outs, 1)) { return false; }
    m.fa = mm->src[1]; m.wfb = mm->src[0]; m.dt = dt; m.a = a; m.out = sc2;
    memcpy(&m.s1, (const float *) sc1->op_params + 0, sizeof(float));
    memcpy(&m.b1, (const float *) sc1->op_params + 1, sizeof(float));
    memcpy(&m.s2, (const float *) sc2->op_params + 0, sizeof(float));
    memcpy(&m.b2, (const float *) sc2->op_params + 1, sizeof(float));
    m.last = j;
    return true;
}

int ggml_cuda_fd_try_kda_gate(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = ggml_cuda_fd_on("GGML_CUDA_FD_KDA_GATE");
    if (!on) {
        return 0;
    }
    fd_kda_gate_match m;
    if (!fd_kda_gate_find(ctx, cgraph, i, m)) {
        return 0;
    }
    if (fd_overlap(m.out, m.fa)) {   // each block reads all of fa
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: %s overlaps its input, unfused path (logged once)\n", __func__, m.out->name);
        }
        ggml_cuda_fd_count(FD_N + FD_KDA_GATE);
        return 0;
    }
    const int nrows = (int) m.wfb->ne[1];
    const int nc    = (int) m.fa->ne[1];
    const int hd    = nrows / (int) m.a->ne[1];
    const int64_t fa_s1  = m.fa->nb[1]/sizeof(float);
    const int64_t dst_s1 = nrows;
    const int stride_row_x = (int) (m.wfb->nb[1]/sizeof(block_q8_0));
    // waves per block: tuning only (a row's order does not depend on it); GGML_CUDA_FD_GATE_WAVES=<1|2|4|8>
    static const int nwv = getenv("GGML_CUDA_FD_GATE_WAVES") ? atoi(getenv("GGML_CUDA_FD_GATE_WAVES")) : 8;
#define FD_GATE_W(N, W) fd_kda_gate_kernel<N, W><<<(nrows + 8*W - 1)/(8*W), 64*W, 0, ctx.stream()>>>((const float *) m.fa->data, \
        fa_s1, m.wfb->data, stride_row_x, (const float *) m.dt->data, (const float *) m.a->data, hd, (float *) m.out->data, \
        dst_s1, nrows, m.s1, m.b1, m.s2, m.b2)
#define FD_GATE(N) do { if (nwv == 1) { FD_GATE_W(N, 1); } else if (nwv == 2) { FD_GATE_W(N, 2); } \
        else if (nwv == 8) { FD_GATE_W(N, 8); } else { FD_GATE_W(N, 4); } } while (0)
    switch (nc) {
        case 2: FD_GATE(2); break;
        case 3: FD_GATE(3); break;
        case 4: FD_GATE(4); break;
        case 5: FD_GATE(5); break;
        case 6: FD_GATE(6); break;
        case 7: FD_GATE(7); break;
        default: FD_GATE(8); break;
    }
#undef FD_GATE_W
#undef FD_GATE
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_fd_count(FD_KDA_GATE);
    return m.last - i;
}

struct fd_kda_tail_match {
    const ggml_tensor * o, * w, * ga, * wgb;
    ggml_tensor * ga_mm;   // the g_a GEMV (run first)
    ggml_tensor * out;     // the CONT the wo GEMV reads
    float eps;
    int last;
};

// [i] CONT(o) RMS_NORM MUL(., w) MUL_MAT(g_a, x) MUL_MAT(g_b, g_a) RESHAPE SIGMOID MUL(nrm, sig) CONT
static bool fd_kda_tail_find(const ggml_backend_cuda_context & ctx, const ggml_cgraph * g, int i, fd_kda_tail_match & m) {
    const int n = g->n_nodes;
    const ggml_tensor * c0 = g->nodes[i];
    if (c0->op != GGML_OP_CONT || c0->type != GGML_TYPE_F32 || c0->ne[0] != 128 || c0->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * o = c0->src[0];
    const int64_t nh = c0->ne[1], nc = c0->ne[2];
    // o: [128, nh, n_seq_tokens, n_seqs] (the cont flattens the last two)
    if (o->type != GGML_TYPE_F32 || o->nb[0] != sizeof(float) || o->ne[0] != 128 || o->ne[1] != nh || o->ne[2]*o->ne[3] != nc ||
            nc < 1 || nc > 8) {
        return false;
    }
    auto at = [&](int j, ggml_op op) -> ggml_tensor * {
        return j < n && g->nodes[j]->op == op ? g->nodes[j] : nullptr;
    };
    int j = fd_next(g, i + 1);
    ggml_tensor * rn = at(j, GGML_OP_RMS_NORM);
    if (!rn || rn->src[0] != c0) { return false; }
    j = fd_next(g, j + 1);
    ggml_tensor * mw = at(j, GGML_OP_MUL);
    if (!mw || mw->src[0] != rn) { return false; }
    const ggml_tensor * w = mw->src[1];
    if (w->type != GGML_TYPE_F32 || !ggml_is_contiguous(w) || ggml_nelements(w) != 128 || w->ne[0] != 128) { return false; }
    j = fd_next(g, j + 1);
    ggml_tensor * ga_mm = at(j, GGML_OP_MUL_MAT);
    if (!ga_mm || ga_mm->type != GGML_TYPE_F32 || ga_mm->ne[0] != 128 || ga_mm->ne[1] != nc || !ggml_is_contiguous(ga_mm)) { return false; }
    j = fd_next(g, j + 1);
    ggml_tensor * gb = at(j, GGML_OP_MUL_MAT);
    if (!gb || gb->src[1] != ga_mm || !fd_mmvq_q8_k128(ctx, gb, nc) || gb->ne[0] != 128*nh) { return false; }
    j = fd_next(g, j + 1);
    ggml_tensor * sig = at(j, GGML_OP_UNARY);
    if (!sig || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || !fd_is_reshape_of(sig->src[0], gb) || !ggml_are_same_shape(sig, c0)) { return false; }
    j = fd_next(g, j + 1);
    ggml_tensor * mg = at(j, GGML_OP_MUL);
    if (!mg || !((mg->src[0] == mw && mg->src[1] == sig) || (mg->src[0] == sig && mg->src[1] == mw)) || !ggml_are_same_shape(mg, c0)) { return false; }
    j = fd_next(g, j + 1);
    ggml_tensor * c1 = at(j, GGML_OP_CONT);
    if (!c1 || c1->src[0] != mg || c1->type != GGML_TYPE_F32 || !ggml_is_contiguous(c1) || c1->ne[0] != 128*nh || c1->ne[1] != nc) { return false; }
    const ggml_tensor * chain[] = { c0, rn, mw, ga_mm, gb, sig, mg, c1 };
    const ggml_tensor * outs[]  = { ga_mm, c1 };
    if (!fd_chain_closed(g, i, j, chain, 8, outs, 2)) { return false; }
    m.o = o; m.w = w; m.ga = ga_mm; m.wgb = gb->src[0]; m.ga_mm = ga_mm; m.out = c1;
    memcpy(&m.eps, (const float *) rn->op_params + 0, sizeof(float));
    m.last = j;
    return true;
}

int ggml_cuda_fd_try_kda_tail(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, ggml_cuda_fd_forward_t fwd) {
    static const bool on = ggml_cuda_fd_on("GGML_CUDA_FD_KDA_TAIL");
    if (!on) {
        return 0;
    }
    fd_kda_tail_match m;
    if (!fd_kda_tail_find(ctx, cgraph, i, m)) {
        return 0;
    }
    // the g_a GEMV runs before the fused launch; it must not land on o (the unfused graph read o first)
    if (fd_overlap(m.out, m.o) || fd_overlap(m.out, m.ga) || fd_overlap(m.ga, m.o)) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            GGML_LOG_INFO("%s: %s overlaps its inputs, unfused path (logged once)\n", __func__, m.out->name);
        }
        ggml_cuda_fd_count(FD_N + FD_KDA_TAIL);
        return 0;
    }
    if (!fwd(ctx, m.ga_mm)) {
        GGML_ABORT("%s: op not supported %s (%s)", __func__, m.ga_mm->name, ggml_op_name(m.ga_mm->op));
    }
    const int nh = (int) m.o->ne[1];
    const int nc = (int) (m.o->ne[2]*m.o->ne[3]);
    // hand the wo GEMV its q8_1 input through the q8 cache (the bytes quantize_q8_1 would write for m.out)
    block_q8_1 * q8 = nullptr;
    int64_t q8_s1 = 0;
    static const bool hand_q8 = !(getenv("GGML_CUDA_FD_KDA_TAIL_Q8") && atoi(getenv("GGML_CUDA_FD_KDA_TAIL_Q8")) == 0);
    ggml_cuda_q8_cache & cache = ggml_cuda_q8_cache_get(ctx.device);
    const int64_t ne10 = m.out->ne[0];
    if (hand_q8 && cache.active && ne10 % MATRIX_ROW_PADDING == 0) {
        bool have = false;
        for (auto & e : cache.entries) {
            have = have || e.t == m.out;
        }
        if (!have) {
            ggml_cuda_q8_cache_entry e;
            e.t = m.out; e.data = m.out->data;
            e.bytes = (size_t) nc*ne10*sizeof(block_q8_1)/QK8_1;
            memcpy(e.ne, m.out->ne, sizeof(e.ne)); memcpy(e.nb, m.out->nb, sizeof(e.nb));
            e.buf = std::make_unique<ggml_cuda_pool_alloc<char>>(ctx.pool(), e.bytes);
            q8    = (block_q8_1 *) e.buf->get();
            q8_s1 = ne10/QK8_1;
            cache.entries.push_back(std::move(e));
        }
    }
    const int64_t ga_s1  = m.ga->nb[1]/sizeof(float);
    const int64_t dst_s1 = m.out->nb[1]/sizeof(float);
    const int stride_row_x = (int) (m.wgb->nb[1]/sizeof(block_q8_0));
    // blocks per head: tuning only; GGML_CUDA_FD_TAIL_BPH=<1|2|4>
    static const int bph_env = getenv("GGML_CUDA_FD_TAIL_BPH") ? atoi(getenv("GGML_CUDA_FD_TAIL_BPH")) : 4;
    const int bph = bph_env == 1 || bph_env == 2 ? bph_env : 4;
#define FD_TAIL_B(N, B) fd_kda_tail_kernel<N, B><<<nh*B, 128, 0, ctx.stream()>>>((const char *) m.o->data, m.o->nb[1], \
        m.o->nb[2], m.o->nb[3], (int) m.o->ne[2], (const float *) m.w->data, m.eps, (const float *) m.ga->data, ga_s1, \
        m.wgb->data, stride_row_x, (float *) m.out->data, dst_s1, q8, q8_s1)
#define FD_TAIL(N) do { if (bph == 1) { FD_TAIL_B(N, 1); } else if (bph == 2) { FD_TAIL_B(N, 2); } else { FD_TAIL_B(N, 4); } } while (0)
    switch (nc) {
        case 1: FD_TAIL(1); break;
        case 2: FD_TAIL(2); break;
        case 3: FD_TAIL(3); break;
        case 4: FD_TAIL(4); break;
        case 5: FD_TAIL(5); break;
        case 6: FD_TAIL(6); break;
        case 7: FD_TAIL(7); break;
        default: FD_TAIL(8); break;
    }
#undef FD_TAIL
#undef FD_TAIL_B
    CUDA_CHECK(cudaGetLastError());
    ggml_cuda_fd_count(FD_KDA_TAIL);
    return m.last - i;
}

// ---- KDA beta: GEMV + sigmoid (GGML_CUDA_FD_KDA_BETA) ---------------------------------------------------
//
// [i] MUL_MAT(beta, x) RESHAPE SIGMOID: the GEMV writes straight into the sigmoid's dst through mmvq_q8_0_v2's
// SIG epilogue (same kernel, op_sigmoid at the store). 2 launches -> 1.

static __global__ void fd_sigmoid_inplace(float * x, const int64_t n) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = 1.0f / (1.0f + expf(-x[i]));
    }
}

int ggml_cuda_fd_try_kda_beta(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    static const bool on = ggml_cuda_fd_on("GGML_CUDA_FD_KDA_BETA");
    if (!on) {
        return 0;
    }
    // columns up to GGML_CUDA_FD_KDA_BETA_MAXC (default 1): at 2..8 columns the SIG instance measured ~12 us against
    // 5 + 4.5 us unfused (MI100, 64 rows; cause not found), at 1 column 4.8 against 4.3 + 4.8
    static const int max_cols = getenv("GGML_CUDA_FD_KDA_BETA_MAXC") ? atoi(getenv("GGML_CUDA_FD_KDA_BETA_MAXC")) : 1;
    ggml_tensor * mm = cgraph->nodes[i];
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * x = mm->src[1];
    const int64_t nc = mm->ne[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    if (mm->op != GGML_OP_MUL_MAT || w->type != GGML_TYPE_Q8_0 || w->ne[0] != 4096 || w->ne[2] != 1 || w->ne[3] != 1 ||
            !ggml_is_contiguous(w) || !w->buffer || !ggml_cuda_fd_buffer_is_cuda(w->buffer) ||
            ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            x->type != GGML_TYPE_F32 || x->ne[2] != 1 || x->ne[3] != 1 || x->nb[0] != sizeof(float) ||
            mm->type != GGML_TYPE_F32 || !ggml_is_contiguous(mm) || nc < 1 || nc > max_cols || !GGML_CUDA_CC_IS_CDNA(cc) ||
            ggml_cuda_should_use_mmf(w->type, cc, warp_size, w->ne, w->nb, (int) nc, false) || !ggml_cuda_should_use_mmvq(w->type, cc, nc)) {
        return 0;
    }
    const int j = fd_next(cgraph, i + 1);
    if (j >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * sig = cgraph->nodes[j];
    if (sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID || !fd_is_reshape_of(sig->src[0], mm) ||
            sig->type != GGML_TYPE_F32 || !ggml_is_contiguous(sig) || ggml_nelements(sig) != ggml_nelements(mm)) {
        return 0;
    }
    const ggml_tensor * chain[] = { mm, sig };
    const ggml_tensor * outs[]  = { sig };
    if (!fd_chain_closed(cgraph, i, j, chain, 2, outs, 1)) {
        return 0;
    }
    ggml_tensor tmp = *mm;   // the GEMV as in the graph, stored into the sigmoid's dst
    tmp.data = sig->data;
    ggml_cuda_mmvq_q8_set_sigmoid(true);
    ggml_cuda_mul_mat_vec_q(ctx, w, x, nullptr, &tmp, nullptr);
    if (ggml_cuda_mmvq_q8_sigmoid_pending()) {   // another kernel took the GEMV: the sigmoid as its own launch
        ggml_cuda_mmvq_q8_set_sigmoid(false);
        const int64_t n = ggml_nelements(sig);
        fd_sigmoid_inplace<<<(n + 255)/256, 256, 0, ctx.stream()>>>((float *) sig->data, n);
        CUDA_CHECK(cudaGetLastError());
    } else {
        ggml_cuda_fd_count(FD_KDA_BETA);
    }
    return j - i;
}

void ggml_cuda_fd_alloc_deps(const ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    static const bool topk_mask = ggml_cuda_fd_on("GGML_CUDA_FD_TOPK_MASK");
    static int dump = getenv("GGML_CUDA_FD_DUMP") ? atoi(getenv("GGML_CUDA_FD_DUMP")) : 0;   // debug: print graphs
    if (dump > 0 && cgraph->n_nodes > 100) {
        dump--;
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            const ggml_tensor * t = cgraph->nodes[i];
            fprintf(stderr, "FD %4d %-12s %-28s [%lld,%lld,%lld,%lld]", i, ggml_op_desc(t), t->name,
                    (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
            for (int k = 0; k < GGML_MAX_SRC && t->src[k]; ++k) {
                fprintf(stderr, " | %s", t->src[k]->name);
            }
            fprintf(stderr, "\n");
        }
    }
    static const bool conv_ms  = ggml_cuda_fd_on("GGML_CUDA_FD_CONV_MS");
    static const bool kda_gate = ggml_cuda_fd_on("GGML_CUDA_FD_KDA_GATE");
    static const bool kda_tail = ggml_cuda_fd_on("GGML_CUDA_FD_KDA_TAIL");
    auto dep = [&](const ggml_tensor * keep, ggml_tensor * until) {
        params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(fd_root(keep)), until);
    };
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * t = cgraph->nodes[i];
        if (kda_gate && t->op == GGML_OP_MUL_MAT) {
            fd_kda_gate_match m;
            if (fd_kda_gate_find(ctx, cgraph, i, m)) {
                dep(m.fa, m.out);   // every block reads all of fa
                i = m.last;
                continue;
            }
        }
        if (kda_tail && t->op == GGML_OP_CONT) {
            fd_kda_tail_match m;
            if (fd_kda_tail_find(ctx, cgraph, i, m)) {
                dep(m.o, m.ga_mm);  // g_a runs before the fused launch reads o
                dep(m.o, m.out);
                dep(m.ga, m.out);
                i = m.last;
                continue;
            }
        }
        if (conv_ms && t->op == GGML_OP_CONCAT) {
            fd_conv_ms_match m;
            if (fd_conv_ms_find(cgraph, i, m)) {
                // q/k/v and the gathered states stay alive until conv_input is allocated
                const ggml_tensor * ins[] = { m.q, m.k, m.v, m.st };
                for (const ggml_tensor * in : ins) {
                    params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(fd_root(in)), m.c3);
                }
                i = m.last;
                continue;
            }
        }
        if (topk_mask && t->op == GGML_OP_CONT) {
            fd_topk_mask_match m;
            if (fd_topk_mask_find(cgraph, i, m)) {
                // the TOP_K result is read by the fused kernel until both outputs are written
                // (and the split-input copies of the pool table / bias, tails and positions)
                const ggml_tensor * ins[] = { m.topk, m.pc, m.pb, m.tc, m.tm, m.kvp, m.qp };
                for (const ggml_tensor * in : ins) {
                    ggml_tensor * rt = const_cast<ggml_tensor *>(fd_root(in));
                    params->add_alloc_dep(params->user_data, rt, m.idx);
                    params->add_alloc_dep(params->user_data, rt, m.mask);
                }
                i = m.last;
            }
        }
    }
}
