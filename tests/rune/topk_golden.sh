#!/usr/bin/env bash
# rune (gfx908) top-k regression check, no model needed. Runs tests/rune/topk_ab over 320 GLM-5.3-Flash-like shapes
# (1025..98306 columns, 1..512 rows, k = 512, 8 key patterns incl. ties, NaN, -inf k-th key) and compares the fnv1a hash
# of every index list with tests/rune/topk-golden.txt (made from the 10th build, 2026-10-11; identical for the 9th build
# and with GGML_CUDA_TOPK_SCAN / FUSED / TIEPAR off). topk_ab also checks every row against a CPU model of the output
# order contract (greater keys in the radix gather's block-strided order, then the first ties in column order), which
# the sparse attention consumes in list order; test-backend-ops only compares sets.
#   build: hipcc -O2 -std=c++17 -Iggml/include tests/rune/topk_ab.cpp -o topk_ab -L<lib dir> -lggml -lggml-base -lggml-hip
#   run:   tests/rune/topk_golden.sh <lib dir> <topk_ab binary> [gpu]    (exit 1 on any difference; ~20 min on one GPU)
set -u
LIB=$1; BINARY=$2; GPU=${3:-0}
HERE=$(cd "$(dirname "$0")" && pwd)
out=$(mktemp)
for spec in "1025 1 3 33" "1100 1 3" "2051 1 3 18 33" "3072 1 3 32 33" "3073 1 3 33" "4100 3" "8258 1 3 18 33 64 65 512" \
            "16390 3 18" "32770 1 3 18 33 64 65 512" "32780 3" "65536 3" "65547 3 33 512" "98306 3 33"; do
  set -- $spec; nc=$1; shift
  for nr in "$@"; do for pat in 0 1 2 3 4 5 6 7; do
    HIP_VISIBLE_DEVICES=$GPU LD_LIBRARY_PATH=$LIB "$BINARY" - $nc $nr 512 $pat 13 2>/dev/null | grep ncols
  done; done
done | awk '{print $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11}' > "$out"
if diff -u "$HERE/topk-golden.txt" "$out"; then echo "top-k golden: $(wc -l < "$out") shapes identical"; rm -f "$out"; exit 0; fi
echo "top-k golden: DIFFERENCES (output kept in $out)"; exit 1
