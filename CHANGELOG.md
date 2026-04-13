# Changelog

All notable changes to audio2report are documented here.

---

## [0.1.0] — 2026-04-12

Initial open-source release, refactored from a proof-of-concept monolithic script.

### Added

**M1 — Structural refactor**
- Modular pip-installable package (`audio2report/`) with Pydantic v2 config, Typer CLI
- `dual` and `single` pipeline commands
- `config init` / `config show` commands
- Hatchling build backend; optional extras: `whisperx`, `diarize`, `llm`, `ui`, `full`
- `AbstractTranscriber` protocol; `WhisperXTranscriber` with lazy model loading (loaded once per run, not per file)
- O(n²) duplicate matching replaced groundwork for bisect optimisation

**Bug fixes (from original script)**
- Fixed `postprocess_segments_for_llm` double-definition — second definition silently shadowed the first
- Fixed `is_very_short` defined after the function that calls it
- Fixed `compute_type="float16"` silent crash on CPU — now auto-downgrades to `int8`

**M2 — Performance + UX**
- Bisect time-windowed deduplication: O(n²) → O(n log n) (300×300 segments: ~25 ms)
- Bigram Jaccard pre-filter in anchor detection: eliminates ~95 % of A×B comparisons
- Rich progress bars integrated with shared `Console` instance (no interleaving with log output)
- `--dry-run`, `--verbose`/`-v`, `--quiet`/`-q` flags on `dual` and `single`
- Three preset configs: `configs/default.yaml`, `configs/cpu_fast.yaml`, `configs/gpu_full.yaml`

**M3 — Testing + Docker + CI**
- 124-test pytest suite (unit + integration); no GPU, no ffmpeg, no WhisperX required
  - ffmpeg/ffprobe monkeypatched with shutil.copy / constant return
  - WhisperX bypassed via pre-written JSON cache files
  - Integration tests build real `DualMicPipeline` instances end-to-end
- `Dockerfile.cpu` (python:3.11-slim), `Dockerfile.gpu` (nvidia/cuda:12.1.0-cudnn8)
- `docker-compose.yml` with volume mounts
- GitHub Actions CI: Python 3.10 / 3.11 / 3.12 matrix

**M4 — LLM report generation**
- `audio2report report` command
- `--report`, `--llm-provider`, `--llm-model`, `--llm-base-url` flags on `dual` / `single`
- `OllamaProvider` — native `/api/generate` REST API via `urllib` (no extra deps)
- `OpenAIProvider` — works with OpenAI, LM Studio, llama.cpp server, Ollama OpenAI compat
- `AbstractLLMProvider` protocol + `get_provider()` factory
- Built-in `audit_report` prompt template (6 structured sections)
- Transcript truncation with tail-preserving logic and truncation notice
- Streaming token output to terminal via `sys.stdout`
- 43 unit tests for all LLM module components

**M5 — Streamlit UI + README + release**
- `audio2report-ui` entry point launching Streamlit web interface
- Three-tab UI: Run Pipeline, Transcript viewer, Report generator
- Live subprocess log streaming in-browser
- Speaker-colour-coded transcript segments with search and suppressed-segment toggle
- Download buttons for cleaned transcript, CSV, and report
- Comprehensive README with architecture diagram, CLI reference, config table
- GitHub Actions release workflow (PyPI publish on version tag)

---

[0.1.0]: https://github.com/alex8642/audio2report/releases/tag/v0.1.0
