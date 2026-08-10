# Intel GPU/NPU Acceleration Roadmap

audio2report's transcription step (WhisperX/faster-whisper/CTranslate2) only
ever supported NVIDIA CUDA or CPU. Modern Intel laptop silicon — especially
Core Ultra "Series 3" (Panther Lake) chips with a capable Arc integrated GPU
and a dedicated NPU — sits idle unless a second, Intel-native transcription
path exists. This document tracks that work across the phases it's actually
going to take, since parts of it depend on operating systems and driver
stacks that aren't all available in any one place.

**Scope**: this covers the **transcription step only**. The LLM
report-generation step (`audio2report/llm/`) is already a pure HTTP client to
Ollama/OpenAI-compatible servers — it needs no code change to benefit from
Intel acceleration; see the [Appendix](#appendix--intel-optimized-local-llm-servers)
for pointers.

## Phase 1 — Linux-native Intel acceleration (this repo, implemented)

**Status: done.** Adds a second transcription backend, `openvino`, built on
Intel's [OpenVINO](https://github.com/openvinotoolkit/openvino) toolkit and
its `openvino_genai.WhisperPipeline` API — the only stack that uniformly
covers CPU, Intel Arc iGPU, and Intel NPU for Whisper. WhisperX remains the
default, unchanged path for CUDA/CPU users; the two backends live side by
side behind the existing `AbstractTranscriber` protocol.

What was added:

- `audio2report/hardware.py` — `detect_hardware()` enumerates CPU / NVIDIA
  CUDA (via torch) / Intel GPU / Intel NPU (via OpenVINO's `Core()`) and
  reports what's actually present.
- `audio2report/utils.py` — `pick_device()` is now backend-aware: "auto"
  mode only ever recommends a device the configured backend can actually
  use (priority: CUDA > Intel GPU > Intel NPU > CPU), and an explicit
  device request that isn't available or isn't supported by the backend
  raises a clear `RuntimeError` instead of silently falling back.
- `audio2report/transcription/openvino_backend.py` — `OpenVINOTranscriber`,
  using pre-converted OpenVINO IR Whisper models from Hugging Face (no local
  model conversion step required). `TranscriptionConfig.model` is
  interpreted as an HF repo id for this backend. Word-level forced
  alignment (WhisperX's separate `align()` step) is intentionally not run
  for this backend — the rest of the pipeline only consumes segment-level
  timestamps, so this isn't a functional gap.
- `audio2report/transcription/_diarize.py` — the diarization call, shared
  by both backends (diarization has no Intel GPU/NPU support of its own; it
  always resolves to cuda/cpu independently of the ASR device). Calls
  `pyannote.audio` directly rather than going through WhisperX's
  `whisperx.diarize` wrapper, so diarization works with just the `diarize`
  extra — the `openvino` backend has no reason to depend on the (much
  heavier, CUDA-torch-pulling) `whisperx` package at all.
- New CLI command `audio2report hardware` — prints detected devices and,
  with `--config`, checks whether a config file's requested device is
  actually usable, exiting non-zero with a clear message if not.
- Streamlit UI: a "Detected hardware" panel at the top of the sidebar,
  extended Device/Backend selectors, and a pre-run warning if the selected
  device isn't available.
- New presets `configs/intel_gpu.yaml` and `configs/intel_npu.yaml`.
- `Dockerfile.intel` + `docker-compose.yml`'s `audio2report-intel` service
  (Intel GPU/Level-Zero userspace runtime, `/dev/dri` passthrough; NPU
  passthrough via `/dev/accel` is included but commented out by default).
- New `intel` optional-dependency extra (`openvino`, `openvino-genai`,
  `openvino-tokenizers`, `huggingface_hub`) — deliberately kept out of the
  `full` extra, since it pulls in Intel-specific expectations that are
  irrelevant on non-Intel machines.

Two pre-existing, unrelated issues were found and fixed while building and
verifying this:

- `pyproject.toml` had `typer[all]>=0.12`, but Typer dropped the `all` extra
  in newer releases (rich/shellingham are now core deps) — the resolver
  warned on every install. Fixed to plain `typer>=0.12`.
- None of `Dockerfile.cpu`/`.gpu`/`.intel` ever `COPY`'d the `LICENSE` file
  into the build context, but `pyproject.toml` declares
  `license = { file = "LICENSE" }`, which hatchling's metadata build
  requires — every one of these Dockerfiles was actually broken (`docker
  build` failed at the `pip install` step). Fixed by adding `LICENSE` to
  each Dockerfile's `COPY` line.
- `Dockerfile.intel` additionally pins CPU-only `torch`/`torchaudio` wheels
  before installing the `diarize` extra — otherwise `pip` resolves
  `pyannote.audio`'s torch dependency to the default CUDA-enabled build,
  which is dead weight on a machine that will never have an NVIDIA GPU.
  This took the built image from 6.76GB down to 2.76GB.
- `Dockerfile.intel`'s base image moved from Ubuntu 22.04 to 24.04 — current
  Intel compute-runtime releases require glibc ≥ 2.38 (Ubuntu 22.04 ships
  2.35) and only publish Ubuntu 24.04-targeted packages. This also required
  switching to Ubuntu 24.04's native Python 3.12 (no `python3.11` package
  available) and setting `PIP_BREAK_SYSTEM_PACKAGES=1` (Ubuntu 24.04 enforces
  PEP 668 on the system Python). None of this is WSL2-specific — it's a
  correctness fix for the image on any host, since the original October-2024
  driver release predates support for this hardware generation entirely.
- `Dockerfile.intel` originally installed `.[intel,diarize]` (no `whisperx`),
  but the diarization code path imported `whisperx.diarize` directly — so
  the documented flagship example (`--diarize` with `configs/intel_gpu.yaml`,
  which has `diarization.enabled: true`) would have crashed immediately with
  "whisperx is not installed". Fixed by rewriting `_diarize.py` to call
  `pyannote.audio` directly (see above) instead of re-adding the much
  heavier `whisperx` package purely for one wrapper class. Verified inside
  the built image: running with `--diarize` and no HF token now correctly
  fails with "no Hugging Face token was provided" instead of an
  `ImportError` — full inference wasn't re-verified since that needs a
  user-provided, terms-accepted HF token for the gated diarization model.
  (`pyannote.audio`'s own audio-decoding fallback also emits a harmless
  `torchcodec`/CUDA-library warning on CPU-only torch — cosmetic, not a
  failure; it appears before the real success/error outcome either way.)

### What was hardware-verified, and where

This phase was implemented and tested directly on the target hardware: a
Framework 13 Pro "Series 3" laptop (Intel Core Ultra X7 358H), running
inside WSL2.

- **CPU (OpenVINO backend)** — fully verified. The complete non-mocked
  pipeline (real model download, real OpenVINO inference, real `ffmpeg`
  normalization) ran end to end both directly and inside the built
  `audio2report:intel` Docker image, producing correct transcription output.
- **Intel Arc iGPU — fully verified, including inside Docker.** Getting
  this working surfaced two real, non-obvious requirements, both now baked
  into `Dockerfile.intel`/`docker-compose.yml` and documented there:
  1. **Driver version matters a lot.** The compute-runtime release initially
     used (October 2024) installed cleanly but silently enumerated CPU-only
     — it predates driver support for this Panther Lake-generation iGPU.
     Upgrading to a current release (26.27.39122.11, mid-2026) fixed
     detection immediately. If a "supported" Intel GPU doesn't show up in
     `audio2report hardware`, suspect a stale compute-runtime release before
     anything else.
  2. **WSL2 needs the device node *and* the host shim libraries.** WSL2 has
     no `/dev/dri` — GPU compute goes through `/dev/dxg`
     (Microsoft/Intel's documented "WDDM GPU Paravirtualization" path, see
     [compute-runtime's WSL.md](https://github.com/intel/compute-runtime/blob/master/documentation/WSL.md)).
     Passing `--device /dev/dxg` alone is not sufficient in Docker: the
     container also needs the WSL-provided `libdxcore.so`/`libd3d12*.so`
     shim libraries, which live on the host at `/usr/lib/wsl/lib` and are
     **not** installed by any apt package — they have to be bind-mounted in
     (`-v /usr/lib/wsl:/usr/lib/wsl`). Without that mount, the container
     sees `/dev/dxg` but the driver silently falls back to CPU, exactly
     like the stale-driver symptom above — easy to misdiagnose as the same
     issue.
  3. **First GPU access after a driver install/upgrade is slow** — several
     minutes, looking exactly like a hang (0% CPU, no output) — before the
     Windows host finishes whatever one-time setup it does for the
     paravirtualized context. Subsequent runs are fast (a few seconds to
     load a small model). Don't kill it prematurely on the very first run
     after installing/upgrading the driver; do treat a hang on the *second*
     or later run as a real problem.
  4. Bare-metal Linux was **not** re-verified in this session (no such
     hardware was available) — only the WSL2 path above was hardware-tested.
     The bare-metal path (`/dev/dri`, no `/usr/lib/wsl` mount needed) should
     work by the same mechanism — `openvino.Core()` enumeration is identical
     either way, only the driver's transport differs — but this is an
     inference from how the stack is documented to work, not something this
     session directly confirmed. Treat it as untested until someone runs
     `audio2report hardware` on real bare-metal Intel Arc hardware.
- **Intel NPU** — WSL2 does **not** expose an NPU device (no `/dev/accel`,
  no NPU kernel driver reachable from inside WSL2 as of this writing). The
  NPU code path is fully unit-tested with mocked OpenVINO devices, but has
  **not** been verified against real NPU hardware. To verify on bare-metal
  Linux:
  1. Confirm `/dev/accel/accel0` exists (Linux NPU kernel driver installed).
  2. Confirm the Intel NPU userspace driver is installed (see
     `intel/linux-npu-driver` releases; `Dockerfile.intel` pins a known-good
     version as a reference).
  3. `python3 -c "import openvino; print(openvino.Core().available_devices)"`
     should include `NPU`.
  4. `audio2report hardware` should report `npu: available`.
  5. Run a real transcription: `audio2report single <file> --backend openvino
     --device npu --out /tmp/npu_test`, and confirm `run_meta.json` shows
     `backend: openvino, device: npu` (not a silent fallback).

### GPU-accelerated diarization via PyTorch XPU

**Status: implemented and verified working (2026-08-10).** Diarization was
found to be entirely CPU-bound (the Intel iGPU sat nearly idle at 1% while
CPU pinned near 100% across ~10 cores during a real batch run), dominating
wall-clock time for any diarization-enabled run — the fast GPU transcription
(~11.7x realtime, confirmed on the full 6.6-hour real dataset) was never the
bottleneck; diarization was, overwhelmingly.

**Fix**: `audio2report/transcription/_diarize.py` now has
`resolve_diarization_device()`, which prefers `cuda` > `xpu` (Intel GPU, via
PyTorch's native XPU backend — separate infrastructure from OpenVINO, which
the ASR backend uses) > `cpu`. `OpenVINOTranscriber` uses this instead of
its old CUDA-only check. `pyannote.audio`'s ops turned out to work fine on
XPU with no unimplemented-op crashes — verified with a real
`pyannote/speaker-diarization-community-1` pipeline moved to
`torch.device("xpu")` and run against real audit recordings, both directly
and through the full `audio2report single --diarize` CLI path.

**Getting XPU working required two things beyond installing `torch+xpu`,
neither obvious from PyTorch's own install docs:**

1. **A genuinely separate system package from what OpenVINO needs.**
   OpenVINO's GPU plugin only needed `intel-opencl-icd`/`libze-intel-gpu1`
   (the Intel-specific Level-Zero driver/ICD, already covered in the iGPU
   section above). PyTorch's Unified Runtime follows the standard
   Level-Zero loader-discovery spec properly and additionally needs
   `libze_loader.so` — the generic Level-Zero loader library, from a
   **different** package (`level-zero`/`libze1`, not
   `intel-level-zero-gpu`). Without it, `torch.xpu.is_available()` silently
   returns `False` with a "XPU device count is zero" warning — no error
   pointing at the real cause. Confirmed via `SYCL_UR_TRACE=1`: the
   Level-Zero adapter (`libur_adapter_level_zero.so.0`) loads fine, it's
   the underlying driver loader that's missing.
   Install (Ubuntu 24.04, matching `Dockerfile.intel`'s other driver
   packages):
   ```
   curl -L -O https://github.com/oneapi-src/level-zero/releases/download/v1.32.0/libze1_1.32.0+u24.04_amd64.deb
   sudo dpkg -i libze1_1.32.0+u24.04_amd64.deb
   ```
2. **A `torch+xpu` build, from a separate pip index**, not the default
   PyPI wheel:
   ```
   pip install --index-url https://download.pytorch.org/whl/xpu torch==2.11.0+xpu torchaudio==2.11.0+xpu
   ```
   (2.11.0 chosen as the highest version with a matching `torchaudio+xpu`
   build at the time of writing; check
   `download.pytorch.org/whl/xpu/torch/` and `.../torchaudio/` for current
   versions.) This is **not** wired into `pyproject.toml`'s `diarize`
   extra — a different package index can't be expressed cleanly as a
   standard extras dependency, and defaulting every `diarize`-extra install
   to a niche wheel index would be surprising for non-Intel users. It's a
   manual, documented opt-in step for now.

**Real timing data** (Framework 13 Pro Series 3, isolated `pyannote.audio`
diarization calls, apples-to-apples on the same files):

| File | Audio duration | CPU | XPU |
|---|---|---|---|
| Short | 152s | 63.7s (2.4x realtime) | 166.9s (0.9x realtime) |
| Long | 1800s (30 min) | 757.4s (2.4x realtime) | 194.2s (**9.3x realtime**) |

CPU throughput is consistent (~2.4x realtime) regardless of file length.
XPU has a large **one-time warm-up cost** (~164s, almost certainly first-use
kernel JIT compilation) but then runs at roughly **60x realtime**
steady-state on an isolated repeat of the *same* file — a clear net loss on
a single short clip, a **3.9x wall-clock win** on the 30-minute file in
isolation. That warm-up cost was also observed to persist **across separate
process invocations** in this small-scale test — a second, independent
`python3` run of the *same* file finished in ~28s instead of ~164s, and a
third only needed a few seconds. The exact caching mechanism wasn't
conclusively identified (no on-disk SYCL/Level-Zero kernel cache directory
was found under `~/.cache` in this WSL2 session, despite the speedup
persisting across processes — most likely a host-side driver cache,
analogous to a GPU shader cache, invisible from inside the Linux
filesystem).

**⚠ That small-scale extrapolation did not hold at real full-batch scale —
correcting it here rather than letting it stand.** Running the actual
39-file, 6.6-hour real audit dataset with diarization enabled (dual-mic
pipeline, transcription + diarization together) took **99m55s total**,
versus **33m56s for the identical batch with transcription only** — so
diarization added **~66 minutes** across the full dataset. Compared to the
~165 minutes a CPU-only run would take at CPU's measured ~2.4x-realtime
rate, that's a real, meaningful **~2.5x speedup**, not the near-60x
steady-state the 2-file isolated test suggested. Most likely explanation:
the "warm-up" only avoids recompiling kernels for a *specific tensor
shape*, and real files vary hugely in length (2.5s to 30 minutes across
this dataset) — each distinct audio length likely triggers its own
one-time compile, so the fixed cost gets paid repeatedly across a batch of
varied-length files rather than just once. **Takeaway: trust the full-batch
number (~2.5x) for planning, not the same-file-repeated number (~60x) — the
latter measures a best case that real, varied-length batches don't hit.**
This also means the "cache the diarization pipeline object across files"
optimization noted earlier is worth more than the small-scale test implied,
since a meaningful part of the added time is plausibly Python/model-load
overhead repeated 39 times, not just GPU kernel compute — not confirmed,
just a more-plausible-now hypothesis for a future profiling pass.

**Done**: `torch+xpu` is wired into `Dockerfile.intel` via a `TORCH_VARIANT`
build arg (`libze1` was already installed unconditionally for NPU support,
so no new system package was needed there), exposed as a separate,
discoverable `audio2report-intel-xpu` docker-compose service rather than a
build-arg someone would need to already know about — see README's "Which
Docker setup do I need?" section. `audio2report hardware` also now reports
the diarization device (not just the ASR device).

**Not yet done**: a corresponding real NPU timing comparison, or
investigating whether `torch+xpu` can also see/use
the NPU (unlikely — NPU access typically goes through a different runtime
entirely, not Level-Zero-over-dxg).

## Phase 2 — Windows-native support (this repo, future session)

**Status: not started.** audio2report should eventually run natively on
Windows (not just under WSL2), where Intel's NPU driver and OpenVINO NPU
plugin support is currently more mature than on Linux.

The device/backend vocabulary introduced in Phase 1
(`TranscriptionConfig.backend`, `.device` ∈ `{cpu, cuda, gpu, npu}`) was
deliberately kept OS-agnostic, so this phase should mostly need:

- A Windows-specific branch inside `audio2report/hardware.py`'s detection
  logic (OpenVINO's device enumeration API itself is cross-platform, but
  driver/runtime *presence* checks differ).
- Windows install documentation (no `Dockerfile` equivalent applies to a
  native Windows install).
- CI matrix additions if Windows CI runners are added.

No new config surface should be required — this is additive to the existing
`hardware.py`/backend design, not a redesign.

## Phase 3 — Enterprise code-signed Windows build (separate private repo, future)

**Status: not started, out of scope for this repo.** A code-signed Windows
build for use on managed/enterprise machines (subject to code-signing
requirements this public repo doesn't need to meet) will live in a
**separate, private repository**, not here. This repo's only relationship
to that future work is that Phase 2's native-Windows code path — once it
exists — is what would get packaged and signed there. No action item exists
in this repo beyond keeping Phase 2's extension points clean.

## Appendix — Intel-optimized local LLM servers

The report-generation step (`audio2report/llm/ollama_provider.py`,
`audio2report/llm/openai_provider.py`) is a plain HTTP client — it works
with anything that speaks the Ollama API or an OpenAI-compatible
`/v1/chat/completions` endpoint, wherever that server happens to run.
`LLMConfig.base_url`/`provider` already support pointing at any such server
today; **no code change in this repo is implied here.**

For Intel GPU/NPU-accelerated report generation, point `llm.base_url` at a
locally-running Intel-optimized server, for example:

- Ollama built with IPEX-LLM (Intel Extension for PyTorch) for Arc iGPU
  acceleration.
- An OpenVINO GenAI-based text-generation server exposing an
  OpenAI-compatible API.
- llama.cpp built with the SYCL backend for Intel GPU acceleration.

These are informational pointers, not a supported/tested configuration —
evaluate whichever fits your setup and point `audio2report report
--base-url <server>` at it.
