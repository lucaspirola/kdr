# kdr — Deployment Reality

**Status:** 2026-05-15. This document is the ground truth for where a
kdr-recovered **ZAYA1-8B** can actually be deployed. It exists because the
README's Phase 7.3/7.4 plan assumed a GGUF→llama.cpp→GTX-1050 path that does
not work today. Read this before planning deployment work.

## TL;DR

| Target | Works today? | Why |
| ------ | ------------ | --- |
| **GGUF → llama.cpp** (Phases 7.3/7.4 as written) | ❌ **No** | llama.cpp has no ZAYA1 architecture support — [ggml-org/llama.cpp#22776](https://github.com/ggml-org/llama.cpp/issues/22776), open since 2026-05-06, no owner, no PR. `kdr_to_gguf` can *write* the file; llama.cpp cannot *load* it. |
| **compressed-tensors → vLLM** | ✅ **Yes** — on a CC≥7.0, ≥8 GB GPU | vLLM runs ZAYA1 via Zyphra's fork (`git+https://github.com/Zyphra/vllm.git@zaya1-pr`); `compressed-tensors` is vLLM's native quant format. |
| **Either path on a GTX 1050 Mobile (4 GB)** | ❌ **No** | See VRAM analysis below — the model does not fit, and vLLM cannot run on the GTX 1050 at all. |

## The two deployment universes

kdr's `da_qad` mode can emit two different artifacts, and they go to two
mutually exclusive runtimes:

1. **compressed-tensors** — HF safetensors + a `quantization_config` block.
   This is vLLM's *native* quant format (built/maintained by the vLLM team).
   Schemes: INT4/INT8 (W4A16/W8A16), FP8, NVFP4. **This is the working path.**
2. **GGUF** (the Profile-J recipe: IQ2_XS/Q3_K/IQ4_XS/Q5_K) — for llama.cpp.
   GGUF codebook/K-quant types are *not* loadable by vLLM, and llama.cpp
   cannot run the ZAYA1 architecture (see below).

## Blocker 1 — llama.cpp has no ZAYA1 support

ZAYA1-8B is `ZayaForCausalLM`: sparse MoE + Compressed Convolutional Attention
(CCA) + an MLP router. llama.cpp implements architectures natively in C++/ggml;
it has **zero** ZAYA1 / Zyphra support — confirmed in the source tree (nothing
in `src/`, `convert_hf_to_gguf.py`, or `gguf-py`) and tracked upstream as
[#22776](https://github.com/ggml-org/llama.cpp/issues/22776) (open, unassigned,
no linked PR).

Consequence: **Phases 7.3 and 7.4 are blocked on an upstream gap with no ETA
and no owner.** Unblocking GGUF/llama.cpp deployment requires *someone* to port
the ZAYA1 architecture into llama.cpp — the GGUF arch enum, the
`convert_hf_to_gguf.py` converter, and the C++ build-graph for CCA + MLP router
+ MoE, adding new ggml ops where CCA needs them. That is a multi-week
architecture port, not a kdr-side fix.

## Blocker 2 — vLLM hardware + quant floors

vLLM runs ZAYA1 (Zyphra fork `zaya1-pr`), but:

- **CUDA compute capability ≥ 7.0 required.** A GTX 1050 Mobile is Pascal,
  **CC 6.1** — below vLLM's floor. vLLM cannot run on it at all.
- **compressed-tensors integer weights: 4-bit or 8-bit only**
  (`WNA16_SUPPORTED_BITS = [4, 8]`). Sub-4-bit (INT2) experts are *not*
  loadable in vLLM.

## VRAM analysis — GTX 1050 Mobile, 4 GB

Computed from the real `Zyphra/ZAYA1-8B` config + safetensors headers
(8,840,488,784 params; 80 layers = 40 attention + 40 MoE; **experts are 91 %**
of the model — `linear_fc1` 60.7 % + `linear_fc2` 30.4 %; embed/lm_head tied).

**CCA KV cache is negligible.** CCA compresses K/V to a 256-dim latent; only
the 40 attention layers carry a cache. At 4-bit: **<80 MB even at 8k context**.
KV is never the constraint — the 91 %-expert weights are the whole fight.

**Weights + KV@8k + ~1.5 GB vLLM overhead:**

| Mapping | Weights | Total | Fits 4 GB? |
| ------- | ------- | ----- | ---------- |
| All-INT4 W4A16 (vLLM legal floor) | 4.69 GB | ~6.26 GB | ❌ |
| 2/4-bit experts (JANG-style) — *not vLLM-loadable* | 3.78 GB | ~5.36 GB | ❌ |
| INT2 experts (hypothetical, *not loadable*) | 2.77 GB | ~4.35 GB | ❌ |

**The 8.4 B model does not fit 4 GB on vLLM** — the INT4 weights alone (4.69 GB)
exceed the budget. vLLM needs an **≥ 8 GB, CC ≥ 7.0** GPU (e.g. a T4 — though a
T4 is 16 GB, where all-INT4 fits with huge headroom). The only artifact that
could approach ~4 GB is a 2-bit-expert GGUF (~3.8 GB weights) on **llama.cpp** —
which is Blocker 1.

## Training-cost note — IQ2_XS STE snap

Separate from deployment, the Profile-J **GGUF** recipe is also expensive to
*train*: the IQ2_XS STE simulator does a 65 536-way codebook argmin, measured at
a flat ~233 ms / Melem (RTX 5080). One per-step snap-all of the IQ2_XS expert
weights costs ~2–6 min on a B200, so the full 381-step Profile-J run is
~20–38 h ≈ $75–145 on a rented B200. The compressed-tensors path (NVFP4 — a
cheap scale-based STE) does not have this cost.

## Recommendations

- **If GPU/server deployment is acceptable:** target **compressed-tensors
  (INT4 / FP8 / NVFP4) → vLLM (Zyphra `zaya1-pr` fork)** on a CC≥7.0, ≥8 GB
  GPU. Works today, no architecture work, and NVFP4's cheap STE sidesteps the
  IQ2_XS training cost.
- **If edge / CPU / GTX-1050 deployment is a hard requirement:** it requires
  porting ZAYA1 into llama.cpp first (multi-week). Until then, Phases 7.3/7.4
  cannot be executed.
- **The GTX 1050 / 4 GB target is not reachable** by either path for an 8.4 B
  model. Drop one of {4 GB, vLLM, GTX 1050}, or accept a smaller model.

vLLM serving of ZAYA1 also needs `--reasoning-parser qwen3
--tool-call-parser zaya_xml`; CCA does not support tensor parallelism (use
DP+EP for multi-GPU).
