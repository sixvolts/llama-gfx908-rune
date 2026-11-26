# llama-gfx908-rune

llama.cpp tuned for **AMD MI100 (gfx908)** to serve **GLM-5.3-Flash** (`glm5next` architecture) with an MTP
(NextN) draft head, pipeline-parallel across 9 GPUs plus 1 drafter GPU. `main` is the current production build;
everything here is built and validated on one 10x MI100 box ("rune").

**Current build: 20th (`glm53-build20`, 2026-11-15).** Single-stream decode ~49-56 tok/s (2k-8k context),
~44 tok/s at 98k; 6 streams ~100-110 tok/s aggregate; prefill ~2.9k tok/s at 8k, ~4.1k tok/s at 32k-98k.
Every optimization is gated by an env flag and checked bit-exact (greedy server oracles, checkpoint hashes,
KL vs the previous build) before it goes into production.

## What to grab

| You want | Use |
|---|---|
| The production build | `main` (= tag `glm53-build20` + this README) |
| An earlier production build | tags `glm53-build02` ... `glm53-build20` (one per promotion) |
| Retired / unmerged experiments | tags `archive/*` (not maintained, not validated as a whole) |
| The old Qwen3.8-Flash-Next line | tag `archive/flash-next-main` |

## Build

ROCm 6.4, gfx908:

```sh
cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx908 -DGGML_HIP_GRAPHS=ON -DGGML_HIP_NO_VMM=ON \
      -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DLLAMA_CURL=OFF
cmake --build build -j --target llama-server llama-perplexity
```

## Run (production configuration)

Most of the gfx908 work is off by default and enabled by environment variables. Production sets:

```sh
# pipeline / prefill
LLAMA_PIPELINE_PARALLEL=1 GGML_SCHED_COPIES=8 GGML_CUDA_STAGE_RING=16 LLAMA_KV_PAD_PREFILL=32768
LLAMA_PP_RESERVE_FULL=1 GGML_SCHED_STAGE_VIEWS=1 GGML_CUDA_H2D_KERNEL=1 LLAMA_PROMPT_SJF=1000
GGML_CUDA_STAGE_RING_MIN_MB=33 GGML_CUDA_STAGE_RING_GROW=2 ROC_P2P_SDMA_SIZE=1048576
# decode
LLAMA_DSA_SPARSE=1 LLAMA_DECODE_GROUPS=6 LLAMA_DECODE_GROUPS_BY_SEQ=1 LLAMA_DECODE_GROUPS_MIN_SEQS=2
LLAMA_KPOOL_INC=1 GGML_CUDA_LI_MFMA=1 GGML_CUDA_FUSE_DENSE=1 GGML_CUDA_FUSE_DEPS=1 GGML_CUDA_GRAPH_FULL_BATCH=512
# MTP drafter
LLAMA_SPEC_MAX_GEN=2 LLAMA_SPEC_NMAX_BY_GEN=2 LLAMA_SPEC_DEFER=3 LLAMA_NEXTN_RING=4
# checkpoints / prompt cache
LLAMA_CKPT_RECYCLE=1 LLAMA_CKPT_SPEC_MARKER=1 LLAMA_CKPT_ASYNC=1 LLAMA_CKPT_TAIL_ALIGN=1
```

Keep `GGML_CUDA_FUSE_DEPS=1` whenever `GGML_CUDA_FUSE_DENSE` or `LLAMA_PP_RESERVE_FULL` is on: the stock
topk-moe fusion is not bit-exact and otherwise fires or declines depending on compute-buffer layout.

```sh
llama-server -m GLM-5.3-Flash-Q4_K_XL.gguf -md GLM-5.3-Flash-MTP.gguf --spec-type draft-mtp \
  --spec-draft-n-max 2 --spec-draft-sampling probabilistic -ngld 99 -devd ROCm9 \
  -ngl 99 -dev ROCm0,ROCm1,ROCm2,ROCm3,ROCm4,ROCm5,ROCm6,ROCm7,ROCm8 -ts 6,4,5,5,5,5,5,5,7 \
  -fa off --load-mode none -lzm off -t 16 -c 786432 -np 6 -ub 512 -b 8192 \
  --no-cache-idle-slots --cache-ram 65536
```

Design notes and measurements: [docs/rune-glm53-flash-plan.md](docs/rune-glm53-flash-plan.md),
[prefill](docs/rune-glm53-flash-prefill.md), [prefill round 2](docs/rune-glm53-flash-prefill-r2.md),
[decode round 3](docs/rune-glm53-flash-decode-r3.md), [gfx908 notes](docs/rune-gfx908.md).

---

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Ajhen0409%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3Aravi9%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Awine99%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
