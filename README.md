# audio2report

[![CI](https://github.com/alex8642/audio2report/actions/workflows/ci.yml/badge.svg)](https://github.com/alex8642/audio2report/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> Convert dual-microphone meeting recordings into clean, deduplicated transcripts and LLM-generated audit reports — fully local, no cloud required.

---

## What makes it different

Most transcription tools treat each audio file independently.  **audio2report** is built for the common scenario where two lapel microphones (or two phones placed on a table) capture the same meeting from different positions.  Every utterance near the table is picked up by both mics, producing hundreds of near-identical duplicate segments when you transcribe them separately.

audio2report solves this with **dual-mic cross-talk deduplication**:

1. Transcribe each channel independently with [WhisperX](https://github.com/m-bain/whisperX).
2. Detect shared anchor utterances to estimate the clock offset between the two recorders.
3. Use a bisect-based time-windowed sweep (O(n log n)) to match cross-mic duplicates by text similarity + timestamp.
4. Keep the louder copy (higher RMS dBFS), suppress the other.
5. Post-process the merged timeline: remove acknowledgement overlaps, merge same-speaker fragments.

The result is a single, clean transcript where every word appears exactly once, with speaker attribution and timestamps throughout.

```
Channel A ──┐
            ├──► Transcribe ──► Align offsets ──► Deduplicate ──► Clean ──► Report
Channel B ──┘    (WhisperX)     (anchor text)     (O(n log n))   (merge)   (LLM)
```

---

## Features

- **Dual-mic deduplication** — text similarity + timestamp alignment + RMS scoring
- **Single-mic mode** — clean post-processing of any single recording folder
- **WhisperX ASR** — fast batched transcription with word-level alignment
- **Optional diarization** — speaker role labels via pyannote.audio
- **LLM report generation** — Ollama, OpenAI, LM Studio, llama.cpp (streaming)
- **Transcription cache** — skip re-transcription on repeat runs
- **Streamlit UI** — web interface for non-CLI users
- **Docker** — CPU and GPU images included
- **Zero cloud dependencies** — everything runs locally

---

## Installation

### From source (recommended while in early development)

```bash
git clone https://github.com/alex8642/audio2report.git
cd audio2report

# Core pipeline only (no ASR, no LLM)
pip install -e .

# Full install — ASR + diarization + LLM + UI
pip install -e ".[full]"
```

### Optional extras

| Extra | Installs | Use when |
|---|---|---|
| `whisperx` | WhisperX + faster-whisper | You want local ASR transcription |
| `diarize` | pyannote.audio ≥ 3.0 | You want speaker role labelling |
| `llm` | openai ≥ 1.0 | You want OpenAI / LM Studio / llama.cpp |
| `ui` | streamlit ≥ 1.30 | You want the web interface |
| `full` | all of the above | Everything |
| `dev` | pytest, ruff, mypy | Development |

### ffmpeg (required for audio normalisation)

```bash
# Ubuntu / Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

---

## Quick Start

### 1. Organise your recordings

audio2report expects a **root folder** containing exactly two sub-folders — one per microphone channel:

```
meeting_2024_01_15/
├── alice_lapel/          ← channel A (the "prime" is derived from folder name)
│   ├── rec_001.m4a
│   └── rec_002.m4a
└── bob_lapel/            ← channel B
    ├── rec_001.m4a
    └── rec_002.m4a
```

Multi-file folders are supported — files are sorted alphabetically and concatenated into a single timeline per channel.  Any format that ffmpeg can decode is accepted (m4a, mp3, wav, ogg, …).

### 2. Run the pipeline

```bash
audio2report dual meeting_2024_01_15/ --output-dir outputs/
```

That's it. The first run takes a few minutes (WhisperX downloads its model once). Subsequent runs on the same files are instant thanks to the transcription cache.

### 3. Inspect the outputs

```
outputs/
├── canonical_transcript.json   ← full segment trace (primary output)
├── cleaned_llm_input.txt       ← clean plain-text transcript
├── canonical_transcript.csv    ← spreadsheet-friendly
├── alignment_anchors.json      ← diagnostic: offset estimation
├── pair_matches.json           ← diagnostic: duplicate pairs
└── run_meta.json               ← run statistics
```

### 4. Generate a report (optional)

```bash
# Using Ollama (local, free)
ollama pull llama3
audio2report dual meeting_2024_01_15/ --output-dir outputs/ --report

# Or from an existing transcript
audio2report report outputs/canonical_transcript.json
```

The report is saved as `outputs/report.md`.

### 5. View in the UI (optional)

```bash
pip install "audio2report[ui]"
audio2report-ui
```

---

## CLI Reference

### `audio2report dual`

Process a root folder containing two channel sub-folders.

```
audio2report dual ROOT [OPTIONS]

Arguments:
  ROOT    Path to the root folder (must contain exactly 2 sub-folders)

Options:
  --output-dir, -o PATH   Where to write outputs [default: ROOT/audio2report_out]
  --config, -c PATH       YAML config file [default: built-in defaults]
  --report                Generate an LLM report after transcription
  --llm-provider TEXT     Override config: LLM provider (ollama|openai)
  --llm-model TEXT        Override config: LLM model name
  --llm-base-url TEXT     Override config: LLM server URL
  --dry-run               Show discovered files and cache status, then exit
  --verbose, -v           Enable DEBUG logging
  --quiet, -q             Suppress INFO logging (errors only)
  --help                  Show this message and exit
```

### `audio2report single`

Process a folder of recordings from a single microphone.

```
audio2report single FOLDER [OPTIONS]

Options: (same as dual, minus the dual-mic-specific flags)
```

### `audio2report report`

Generate a report from an existing transcript file.

```
audio2report report TRANSCRIPT [OPTIONS]

Arguments:
  TRANSCRIPT   Path to canonical_transcript.json or any .txt transcript

Options:
  --provider TEXT       LLM provider (ollama|openai) [default: ollama]
  --model TEXT          Model name [default: llama3]
  --base-url TEXT       Server URL [default: http://localhost:11434]
  --api-key TEXT        API key (or set OPENAI_API_KEY env var)
  --template TEXT       Prompt template name or path [default: audit_report]
  --output-dir, -o PATH Where to save report.md [default: TRANSCRIPT parent dir]
  --no-stream           Disable streaming output to terminal
```

### `audio2report config`

```bash
audio2report config init          # write default config to ./audio2report.yaml
audio2report config show          # print the resolved config (with current overrides)
```

---

## Configuration

Copy `configs/default.yaml` and pass it with `--config`:

```bash
audio2report dual meeting/ --config my_config.yaml
```

### Full reference

| Section | Key | Default | Description |
|---|---|---|---|
| *(root)* | `mode` | `dual` | `dual` or `single` |
| *(root)* | `cache` | `true` | Skip re-transcription when per-file JSON exists |
| `audio` | `inter_file_gap_sec` | `0.5` | Gap inserted between sequential files |
| `audio` | `min_duration_sec` | `1.0` | Files shorter than this are skipped |
| `transcription` | `model` | `large-v3` | WhisperX model size |
| `transcription` | `language` | `null` | Force language (null = auto-detect) |
| `transcription` | `compute_type` | `float16` | `float16` (GPU) or `int8` (CPU) — auto-downgraded |
| `transcription` | `device` | `null` | `cuda` or `cpu` (null = auto-detect) |
| `diarization` | `enabled` | `false` | Enable pyannote.audio speaker roles |
| `diarization` | `hf_token` | `null` | HuggingFace token (or `HF_TOKEN` env var) |
| `alignment` | `anchor_sim_threshold` | `0.90` | Min text similarity to use as clock anchor |
| `alignment` | `min_anchor_text_len` | `25` | Min text length for anchor candidates |
| `deduplication` | `enabled` | `true` | Enable cross-mic duplicate suppression |
| `deduplication` | `sim_threshold` | `0.86` | Min text similarity to suppress a duplicate |
| `deduplication` | `time_tolerance_sec` | `2.5` | Max timestamp gap (after alignment) |
| `deduplication` | `min_text_len` | `18` | Min text length for dedup candidates |
| `output` | `formats` | `[json,csv,txt]` | Output formats |
| `output` | `include_suppressed` | `true` | Include suppressed segments in JSON/CSV |
| `llm` | `enabled` | `false` | Auto-run report after pipeline |
| `llm` | `provider` | `ollama` | `ollama` or `openai` |
| `llm` | `model` | `llama3` | Model name |
| `llm` | `base_url` | `http://localhost:11434` | Server URL |
| `llm` | `max_transcript_chars` | `50000` | Truncate transcript if longer |
| `llm` | `stream` | `true` | Stream response tokens to terminal |

### Preset configs

| File | Best for |
|---|---|
| `configs/default.yaml` | GPU with large-v3, no diarization |
| `configs/cpu_fast.yaml` | CPU-only machines (medium model, int8) |
| `configs/gpu_full.yaml` | GPU with diarization enabled |

---

## Streamlit UI

The UI provides a browser-based interface for running the pipeline and viewing results — no command line required.

```bash
pip install "audio2report[ui]"
audio2report-ui                   # opens http://localhost:8501
```

Features:
- Folder path inputs with dry-run preview
- Config form (model, language, diarization, LLM settings)
- Live pipeline log stream
- Transcript viewer with speaker-colour-coded segments
- One-click report generation with inline Markdown preview
- Download buttons for transcript, CSV, and report

---

## Docker

### CPU

```bash
docker compose run --rm audio2report-cpu \
  audio2report dual /data/meeting/ --output-dir /data/outputs/
```

### GPU (requires NVIDIA Container Toolkit)

```bash
docker compose run --rm audio2report-gpu \
  audio2report dual /data/meeting/ --output-dir /data/outputs/
```

Mount your data with `-v /path/to/your/meetings:/data`.

### Build from scratch

```bash
docker build -f Dockerfile.cpu -t audio2report:cpu .
docker build -f Dockerfile.gpu -t audio2report:gpu .
```

---

## Architecture

```
audio2report/
├── cli/
│   └── main.py              Typer CLI — dual, single, report, config commands
├── pipeline/
│   ├── dual.py              DualMicPipeline — full 8-stage pipeline
│   └── single.py            SingleMicPipeline — single-channel variant
├── ingestion/
│   └── audio_files.py       Discover, sort, and normalise audio files via ffmpeg
├── transcription/
│   ├── base.py              AbstractTranscriber protocol
│   └── whisperx_backend.py  WhisperX implementation (lazy model loading + cache)
├── diarization/
│   └── roles.py             pyannote.audio speaker-role assignment
├── alignment/
│   └── anchors.py           Clock offset estimation via shared anchor utterances
├── deduplication/
│   └── matching.py          O(n log n) bisect-windowed cross-mic deduplication
├── postprocessing/
│   └── cleanup.py           Ack-suppression + same-speaker fragment merging
├── output/
│   └── writers.py           JSON, CSV, TXT, run_meta writers
├── llm/
│   ├── base.py              AbstractLLMProvider protocol + get_provider() factory
│   ├── ollama_provider.py   Ollama (urllib, no extra deps)
│   ├── openai_provider.py   OpenAI-compatible (openai package)
│   ├── report.py            Prompt assembly, truncation, streaming
│   └── templates/
│       └── audit_report.txt Built-in prompt template
├── ui/
│   └── app.py               Streamlit web interface
├── config.py                Pydantic v2 config models + YAML loader
├── models.py                Shared dataclasses (SegmentRecord, RunMeta, …)
├── utils.py                 Text similarity, RMS, device detection
└── _log.py                  Shared Rich console + logging setup
```

### Key design decisions

**Shared clock offset is unknown at dedup time.**  The two recorders start at different wall-clock times.  audio2report estimates the offset by finding utterances that appear on both channels with high text similarity (anchors), computing the timestamp deltas, and taking the MAD-filtered median.  This requires no hardware sync signal.

**Bigram Jaccard pre-filter in anchor detection.**  Since the offset is unknown during anchor search, we can't use a time window — we must compare all A×B pairs.  A bigram Jaccard check eliminates ~95 % of pairs before the expensive SequenceMatcher call, keeping anchor detection fast even for hour-long meetings.

**Bisect time-window in deduplication.**  Once the offset is known, we sort B-segment start times and use `bisect_left` / `bisect_right` to find only the candidates within ±`time_tolerance_sec`.  300×300 segments runs in ~25 ms.

**WhisperX model loaded once.**  The `WhisperXTranscriber` instance caches the loaded model; it is not reloaded per file.

**Single shared Rich console.**  RichHandler and Rich Progress share the same `Console` instance (`_log.get_console()`) so progress bars and log lines never interleave.

---

## Development

```bash
git clone https://github.com/alex8642/audio2report.git
cd audio2report
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check audio2report/ tests/

# Type-check
mypy audio2report/
```

The test suite (167 tests) uses `pytest` with monkeypatching to avoid requiring ffmpeg, WhisperX, or a GPU:
- ffmpeg calls are stubbed with `shutil.copy`
- WhisperX is bypassed via pre-written JSON cache files
- LLM providers are mocked with `unittest.mock`

---

## Contributing

1. Fork and create a feature branch.
2. Write tests for new behaviour.
3. Run `pytest tests/ -v` — all tests must pass.
4. Open a PR against `main`.

Bug reports and feature requests are welcome via [GitHub Issues](https://github.com/alex8642/audio2report/issues).

---

## License

MIT — see [LICENSE](LICENSE).
