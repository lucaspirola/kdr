# kdr — Knowledge Distillation Recovery

Unified BF16 KD + Deployment-Aware QAD trainer for MoE-LLM recovery after
aggressive weight quantization. One mode flag, one FKLD loss, asymmetric K/V
quant recipes. See [`requirements/`](requirements/) for the HLR/LLR set,
[`docs/PROFILEJ_STRATEGY.md`](docs/PROFILEJ_STRATEGY.md) for the recipe
rationale, and **[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for where a
recovered ZAYA1-8B can actually be deployed** (the GGUF/llama.cpp path is
currently blocked upstream — read it before planning Phase 7.3/7.4).

## Quick start

```bash
# Local dev (CPU/single-GPU smoke):
uv sync
uv run pytest tests/

# vast.ai / RunPod training: see docker/README.md.
# The image at ghcr.io/lucaspirola/kdr:latest runs docker/bootstrap.sh as
# ENTRYPOINT; pass env vars HF_TOKEN, STUDENT_REPO, CACHE_MOUNT, KDR_CONFIG,
# KDR_MODE per docker/README.md.
```

## Status

| Phase                                         | State                                       |
|-----------------------------------------------|---------------------------------------------|
| 0 — `req` Python `#` scanner patch            | ✅ Upstreamed                               |
| 1 — Requirements (12 HLRs, 49 LLRs)           | ✅ Authored                                 |
| 2 — Skeleton (Pydantic, mypy strict, ruff)    | ✅ Landed                                   |
| 3a — FKLD numerical parity                    | ✅ Bit-equal vs structural_recovery         |
| 3b — BF16 loop on stand-ins                   | ✅ Code-port complete                       |
| 4 — QuantBackend layer                        | ✅ ModelOpt + Native + factory + save_kdr_artifact |
| 5 — Mode integration + ZAYA1 adapter          | ✅ All adapter methods + router replay      |
| 6 — Vast.ai docker bootstrap                  | ✅ bootstrap.sh + run_id + HF Hub upload    |
| 7.1 — ZAYA1-8B `bf16` parity smoke            | 🟡 A100 train/save/upload green; load-back verified by hand |
| 7.2 — ZAYA1-8B `da_qad` Profile-J GGUF smoke (IQ2_XS/Q3_K/IQ4_XS/Q5_K + INT4 KV) | ⛔ Blocked — IQ2_XS STE-snap cost (~20–38 h/run); see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| 7.3 — Profile-J `.gguf` round-trip in llama.cpp | ⛔ Blocked upstream — llama.cpp has no ZAYA1 arch ([#22776](https://github.com/ggml-org/llama.cpp/issues/22776)) |
| 7.4 — GGUF deployment validation                | ⛔ Blocked upstream — see [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) |
| 8 — Cutover (SUPERSEDED-BY tagging)           | ⏸️ Gated on 7.1 sign-off                   |

`mypy --strict` clean (31 source files); `ruff` clean; `pytest` 176/176;
`req coverage` 37/49 LLRs implemented + tested. Remaining unimplemented LLRs
(LLR-0008/0034/0044/0046/0047) are CLI polish + Phase 7 hardware gates.

## Modes

| Mode      | What it does                                                                |
|-----------|-----------------------------------------------------------------------------|
| `bf16`    | Plain forward-KLD logit distillation (Chapter 1). No `mtq.quantize` call.   |
| `da_qad`  | Chapter 3 DA-QAD: `mtq.quantize` installs fake-quant per the YAML's `quant` block; same FKLD loop, plus router replay (LLR-0025) for MoE QAD stability. |

## Output

`bf16` mode writes a plain HF safetensors checkpoint, loadable directly via
`AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`.

`da_qad` mode writes a kdr artifact directory — `*.safetensors` weights plus a
`config.json` carrying a `quantization_config` block (LLR-0056) that records
the per-group Profile-J recipe. `src/kdr/tools/kdr_to_gguf.py` (LLR-0059)
converts that directory into a single llama.cpp `.gguf`:

```bash
python -m kdr.tools.kdr_to_gguf --kdr-dir path/to/kdr_output --output model.gguf
```

Per-tensor GGUF encoding is driven by the artifact's
`quantization_config.config_groups` (first-match-wins substring match); FP32
carve-outs are emitted as F16. The input student's `compressed_metadata.json`
(MoE-factored topology) is preserved verbatim if present; omitted otherwise.

## Validation runs (Phase 7)

Phase 7 is real-hardware validation. The kdr code is feature-complete (Phases
0–6 closed), but the static-analysis surface (mypy/ruff/pytest) cannot detect
GPU-side regressions. Phase 7 runs are required before SUPERSEDED-BY tagging
of `structural_recovery` (Phase 8 cutover).

| Run    | Hardware              | What it proves                                                       |
|--------|-----------------------|----------------------------------------------------------------------|
| 7.1    | 1× H200 OR A100-80GB  | `bf16` 200-step parity smoke; PPL within 0.5% of `structural_recovery` |
| 7.2    | 1× B200 (≥180 GB)     | `da_qad` Profile-J 50M-token smoke; no NaN; FKLD monotone-decreasing  |
| 7.3    | RTX 5080 (this host)  | Profile-J `.gguf` loads in llama.cpp + PPL matches training-time sim   |
| 7.4    | RTX 5080 (this host)  | `.gguf` runs under llama.cpp at usable context; deployed PPL vs sim   |

Launch via `docker/bootstrap.sh` on a vast.ai instance — see
[`docker/README.md`](docker/README.md) for the operator runbook.

7.1 trains on a rented A100/H200. The Profile-J recipe (7.2) is near-OOM and
runs **only on a B200 (≥180 GB)** — an H200's 141 GB does not fit.

> **⚠️ 7.2/7.3/7.4 are currently blocked — read [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).**
> Two issues surfaced during the 7.2 B200 run: (1) the Profile-J IQ2_XS STE
> simulator makes a full run ~20–38 h / ~$100+ on a B200; (2) **llama.cpp has
> no ZAYA1 architecture support** ([#22776](https://github.com/ggml-org/llama.cpp/issues/22776)),
> so a Profile-J `.gguf` cannot be loaded or run there at all — 7.3/7.4 as
> written are not executable. The viable deployment path is
> compressed-tensors → vLLM (Zyphra `zaya1-pr` fork) on a CC≥7.0 ≥8 GB GPU;
> `docs/DEPLOYMENT.md` has the full analysis.

## Cutover (Phase 8)

After Phase 7.1 closes, `structural_recovery` is marked
`SUPERSEDED-BY: kdr` (project-level note in `../requirements/` since
`structural_recovery` is outside the `.req` graph). The `structural_recovery`
files are RETAINED as historical reference for future audit; they are NOT
deleted.

## Out of scope

MLX (jangq) deployment is out of scope for v0; users targeting it re-quantize
from the BF16 master in the kdr output and accept the recovery loss — the
`JangqBackend` plugin is v1+ if ever.

`kdr_to_gguf` emits a Profile-J `.gguf`, but **llama.cpp cannot currently run
it** — llama.cpp has no ZAYA1 architecture support
([#22776](https://github.com/ggml-org/llama.cpp/issues/22776)). The deployment
path that works today is `da_qad` → compressed-tensors → vLLM (Zyphra
`zaya1-pr` fork). See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for the full
analysis, the VRAM math, and why the GTX 1050 / 4 GB target is not reachable.

Multi-GPU ZeRO-3 is not validated in Phase 7 (single-GPU is sufficient for
ZAYA1's 8.4 B-total / 760 M-active footprint). The loop dispatches correctly
under DS3 in principle (LLR-0048 call order is preserved), but no run has
been executed against multi-GPU.
