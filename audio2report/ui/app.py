"""audio2report Streamlit UI.

Launch with:
    audio2report-ui                       # via installed entry point
    streamlit run audio2report/ui/app.py  # directly
"""
from __future__ import annotations

import hashlib
import io
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Temp-directory lifecycle
#
# Streamlit reruns this whole script in a fresh namespace on every
# interaction, so plain module-level tracking (e.g. a set + atexit hook)
# only ever "remembers" whatever the *most recent* rerun created — anything
# from earlier clicks, earlier sessions, or a killed/crashed process is
# unreachable and leaks. Instead of trying to track and clean up on exit,
# every run dirs go under one dedicated folder, and each new session sweeps
# it for anything stale before doing its own work — self-healing regardless
# of how the previous process ended (clean exit, SIGKILL, crash, refresh).
# ---------------------------------------------------------------------------

_TEMP_ROOT = Path(tempfile.gettempdir()) / "audio2report_ui"
_STALE_AFTER_SECONDS = 30 * 60


def _sweep_stale_temp_dirs() -> None:
    if not _TEMP_ROOT.is_dir():
        return
    cutoff = time.time() - _STALE_AFTER_SECONDS
    for entry in _TEMP_ROOT.iterdir():
        try:
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry, ignore_errors=True)
        except FileNotFoundError:
            pass  # removed concurrently, nothing to do


def _rm_temp_dir(path: str | None) -> None:
    if path:
        shutil.rmtree(path, ignore_errors=True)


def _files_signature(files) -> str:
    """Content hash of a set of uploaded files — identical uploads always
    map to the same output dir, so the per-file transcription cache
    (out_root/per_file_whisperx_json/) can actually be reused across
    reruns instead of starting from an empty dir every click."""
    h = hashlib.sha256()
    for f in sorted(files, key=lambda uf: uf.name):
        h.update(f.name.encode())
        h.update(f.getvalue())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Entry point (called by the audio2report-ui console script)
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch Streamlit with this file as the app."""
    app_path = Path(__file__).resolve()
    extra_args = sys.argv[1:]
    # Streamlit's 200MB default upload cap is too small for hour-long
    # dual-mic recordings; raise it unless the user already set one.
    if not any(a.startswith("--server.maxUploadSize") for a in extra_args):
        extra_args = ["--server.maxUploadSize", "2048"] + extra_args
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app_path),
             "--server.headless", "false"]
            + extra_args
        )
    )


# ---------------------------------------------------------------------------
# Streamlit app
#
# Streamlit sets __name__ = "__main__" when executing a script file, so this
# block runs during `streamlit run app.py` but NOT when the module is imported
# (e.g. to access the `main` entry point above).
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        import streamlit as st
    except ImportError:
        print(
            "Streamlit is not installed.\n"
            "Install it with:  pip install audio2report"
        )
        sys.exit(1)

    import json

    st.set_page_config(
        page_title="audio2report",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Session state initialisation ─────────────────────────────────────────

    def _init_state() -> None:
        defaults = {
            "log_lines": [],
            "pipeline_done": False,
            "transcript_segments": None,
            "report_text": None,
            "run_meta": None,
            "output_files": {},       # {filename: bytes} — populated after each run
            "_tmp_input_dir": None,
            "_tmp_output_dir": None,
            "_run_active": False,     # a subprocess is currently running, being polled
            "_run_kind": None,        # "dry_run" | "run" — which button started it
            "_run_cancelled": False,  # distinguishes a Cancel click from a pipeline error
            "_proc": None,
            "_reader_thread": None,
            "_log_queue": None,
            "_progress_file": None,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    _init_state()

    # Once per session (not once per rerun): sweep leftovers from any
    # previous process that never got to clean up after itself.
    if not st.session_state.get("_swept_stale_temp_dirs"):
        _sweep_stale_temp_dirs()
        st.session_state["_swept_stale_temp_dirs"] = True

    # ── Sidebar — configuration ───────────────────────────────────────────────

    with st.sidebar:
        st.title("🎙️ audio2report")
        st.caption("Dual-mic transcription & report generation")
        st.divider()

        st.subheader("🔧 Detected hardware")
        from audio2report.hardware import detect_hardware

        @st.cache_data(ttl=30, show_spinner=False)
        def _cached_detect_hardware():
            # Streamlit reruns the whole script on every widget interaction;
            # OpenVINO device enumeration is too slow to redo on every one,
            # so cache it briefly instead of calling detect_hardware() live.
            return detect_hardware()

        _hw_report = _cached_detect_hardware()
        _hw_labels = {"cpu": "CPU", "cuda": "NVIDIA CUDA", "gpu": "Intel GPU", "npu": "Intel NPU"}
        with st.expander(
            f"Auto will use: **{_hw_labels.get(_hw_report.recommended, _hw_report.recommended)}**",
            expanded=False,
        ):
            for _d in _hw_report.devices:
                _icon = "✅" if _d.available else "⬜"
                _label = _hw_labels.get(_d.id, _d.id)
                _extra = _d.name or _d.detail or ""
                st.caption(f"{_icon} **{_label}** — {_extra}" if _extra else f"{_icon} **{_label}**")
        st.divider()

        st.subheader("Mode")
        mode = st.radio("Pipeline mode", ["dual", "single"], horizontal=True)

        st.subheader("Transcription")
        backend = st.selectbox(
            "Backend", ["auto", "whisperx", "openvino", "torch_whisper"], index=0
        )

        from audio2report.utils import resolve_backend_and_device

        @st.cache_data(ttl=30, show_spinner=False)
        def _cached_resolve_auto():
            return resolve_backend_and_device("auto", None)

        if backend == "auto":
            try:
                _auto_backend, _auto_device = _cached_resolve_auto()
                st.caption(f"Auto will use: **{_auto_backend}** on **{_auto_device}**")
            except RuntimeError as exc:
                st.error(f"⚠️ {exc}")
                _auto_backend, _auto_device = "whisperx", "cpu"  # widget fallback only
            effective_backend = _auto_backend
        else:
            effective_backend = backend

        language = st.text_input("Language (blank = auto-detect)", value="")
        device = st.selectbox("Device", ["auto", "cpu", "cuda", "gpu", "npu"], index=0)
        compute_type = st.selectbox("Compute type", ["auto", "float16", "int8"], index=0)

        if device != "auto":
            _selected_info = _hw_report.get(device)
            if _selected_info is not None and not _selected_info.available:
                st.warning(
                    f"⚠️ '{_hw_labels.get(device, device)}' was not detected on this "
                    f"system — the run will fail at pipeline start. "
                    f"({_selected_info.detail or 'unavailable'})"
                )
        if backend in ("whisperx", "torch_whisper") and device in ("gpu", "npu"):
            st.warning(
                f"⚠️ The {backend} backend cannot use Intel GPU/NPU — "
                "switch Backend to 'openvino' above."
            )

        @st.cache_data(ttl=30, show_spinner=False)
        def _cached_pick_device(backend_: str) -> str:
            from audio2report.utils import pick_device
            return pick_device(None, backend_)

        if device == "auto":
            try:
                effective_device = _auto_device if backend == "auto" else _cached_pick_device(effective_backend)
            except RuntimeError:
                effective_device = "cpu"  # widget fallback only
        else:
            effective_device = device

        # torch_whisper uses the same short model names as whisperx (both
        # ultimately load an openai-whisper-compatible checkpoint) — only
        # openvino's HF-repo-id convention needs a different input.
        whisper_model = st.selectbox(
            "Whisper model",
            ["auto", "large-v3", "large-v2", "medium", "small", "base", "tiny"],
            index=0,
        ) if effective_backend != "openvino" else st.text_input(
            "OpenVINO model (HF repo id, or 'auto')", value="auto"
        )

        if whisper_model == "auto":
            from audio2report.utils import resolve_model

            @st.cache_data(ttl=30, show_spinner=False)
            def _cached_resolve_model(backend_: str, device_: str) -> str:
                return resolve_model("auto", backend_, device_)

            st.caption(f"Auto will use: **{_cached_resolve_model(effective_backend, effective_device)}**")

        st.subheader("Diarization")
        diarize = st.checkbox("Enable speaker diarization", value=True)
        hf_token = ""
        if diarize:
            hf_token = st.text_input("HuggingFace token", type="password")
            import os as _os

            if not hf_token and not _os.environ.get("HF_TOKEN"):
                st.warning(
                    "⚠️ No Hugging Face token found (field is blank and "
                    "HF_TOKEN is not set) — diarization will be skipped "
                    "automatically at run time."
                )

        st.subheader("Output")
        cache = st.checkbox("Use transcription cache", value=True)
        out_formats = st.multiselect(
            "Output formats", ["json", "csv", "txt"], default=["json", "csv", "txt"]
        )

        st.subheader("LLM Report")
        llm_enabled = st.checkbox("Generate report after pipeline", value=False)
        llm_provider = st.selectbox("Provider", ["ollama", "openai"], index=0)
        llm_model = st.text_input("Model", value="llama3")
        llm_base_url = st.text_input("Server URL", value="http://localhost:11434")
        llm_api_key = ""
        if llm_provider == "openai":
            llm_api_key = st.text_input("API key", type="password")
        llm_stream = st.checkbox("Stream response", value=True)

        st.divider()
        st.caption("audio2report v0.1.0")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_inline_config(tmp_dir: Path) -> Path:
        """Write a temporary YAML config reflecting the current sidebar settings."""
        import yaml

        device_val = None if device == "auto" else device
        compute_val = "float16" if compute_type == "auto" else compute_type

        cfg = {
            "mode": mode,
            "cache": cache,
            "transcription": {
                "backend": backend,
                "model": whisper_model,
                "language": language.strip() or None,
                "device": device_val,
                "compute_type": compute_val,
            },
            "diarization": {
                "enabled": diarize,
                "hf_token": hf_token or None,
            },
            "output": {
                "formats": out_formats,
                "include_suppressed": True,
            },
            "llm": {
                "enabled": llm_enabled,
                "provider": llm_provider,
                "model": llm_model,
                "base_url": llm_base_url,
                "stream": llm_stream,
            },
        }
        config_path = tmp_dir / "ui_config.yaml"
        config_path.write_text(yaml.dump(cfg, allow_unicode=True), encoding="utf-8")
        return config_path

    def _load_outputs(output_dir: str) -> None:
        """Read pipeline output into session state. Bytes cached for download buttons."""
        out = Path(output_dir)

        transcript_json = out / "canonical_transcript.json"
        if transcript_json.exists():
            data = json.loads(transcript_json.read_text(encoding="utf-8"))
            st.session_state["transcript_segments"] = data.get("segments", [])

        meta_json = out / "run_meta.json"
        if meta_json.exists():
            st.session_state["run_meta"] = json.loads(
                meta_json.read_text(encoding="utf-8")
            )

        report_md = out / "report.md"
        if report_md.exists():
            st.session_state["report_text"] = report_md.read_text(encoding="utf-8")

        # Cache all downloadable files as bytes so downloads survive temp-dir cleanup
        download_names = [
            "canonical_transcript.json",
            "canonical_transcript.csv",
            "canonical_transcript.txt",
            "cleaned_llm_input.txt",
            "alignment_anchors.json",
            "run_meta.json",
            "report.md",
        ]
        files: dict[str, bytes] = {}
        for name in download_names:
            p = out / name
            if p.exists():
                files[name] = p.read_bytes()
        st.session_state["output_files"] = files

    def _reader_thread_fn(proc: subprocess.Popen, q: queue.Queue) -> None:
        """Background thread: pushes subprocess output lines into a
        thread-safe queue. Never touches st.session_state directly — only
        the main script thread does that, by draining the queue on each
        rerun. This is what makes the run non-blocking: without it,
        reading proc.stdout in the main script thread would freeze the
        whole Streamlit app (including a Cancel button) for the run's
        full duration."""
        assert proc.stdout is not None
        for line in proc.stdout:
            q.put(line.rstrip())
        q.put(None)  # sentinel: stdout closed

    def _start_subprocess(cmd: list[str], progress_file: Path, run_kind: str) -> None:
        """Launch *cmd* as a non-blocking background subprocess and stash
        everything needed to poll it in session_state."""
        st.session_state["log_lines"] = []
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        q: queue.Queue = queue.Queue()
        thread = threading.Thread(target=_reader_thread_fn, args=(proc, q), daemon=True)
        thread.start()
        st.session_state["_proc"] = proc
        st.session_state["_reader_thread"] = thread
        st.session_state["_log_queue"] = q
        st.session_state["_progress_file"] = progress_file
        st.session_state["_run_kind"] = run_kind
        st.session_state["_run_active"] = True
        st.session_state["_run_cancelled"] = False

    def _drain_log_queue(log_area) -> None:
        q = st.session_state["_log_queue"]
        while True:
            try:
                line = q.get_nowait()
            except queue.Empty:
                break
            if line is not None:
                st.session_state["log_lines"].append(line)
        if st.session_state["log_lines"]:
            log_area.code("\n".join(st.session_state["log_lines"][-200:]), language=None)

    def _read_progress() -> dict | None:
        """Last event written by the CLI's --progress-file sink, if any."""
        pf = st.session_state.get("_progress_file")
        if not pf or not pf.exists():
            return None
        try:
            lines = pf.read_text(encoding="utf-8").strip().splitlines()
            return json.loads(lines[-1]) if lines else None
        except (json.JSONDecodeError, OSError):
            return None  # caught it mid-write; try again next poll

    # ── Main tabs ─────────────────────────────────────────────────────────────

    tab_run, tab_transcript, tab_report = st.tabs(
        ["▶ Run Pipeline", "📄 Transcript", "📋 Report"]
    )

    # ── Tab 1: Run Pipeline ───────────────────────────────────────────────────

    with tab_run:
        st.header("Run Pipeline")

        # ── File upload ───────────────────────────────────────────────────────
        if mode == "dual":
            st.caption(
                "Upload WAV files from each microphone. "
                "Channel A = interviewer mic, Channel B = subject mic."
            )
            col_a, col_b = st.columns(2)
            with col_a:
                uploaded_a = st.file_uploader(
                    "🎤 Channel A — Interviewer",
                    type=["wav"],
                    accept_multiple_files=True,
                    key="uploader_a",
                )
            with col_b:
                uploaded_b = st.file_uploader(
                    "🎤 Channel B — Subject",
                    type=["wav"],
                    accept_multiple_files=True,
                    key="uploader_b",
                )
            files_ready = bool(uploaded_a and uploaded_b)
            uploaded_single = []
        else:
            uploaded_single = st.file_uploader(
                "🎙 Audio file(s) — WAV",
                type=["wav"],
                accept_multiple_files=True,
                key="uploader_single",
            )
            uploaded_a = uploaded_b = []
            files_ready = bool(uploaded_single)

        run_active = st.session_state["_run_active"]

        if not files_ready:
            st.caption(
                "Upload at least one WAV file for each channel to enable Dry Run / Run."
                if mode == "dual"
                else "Upload at least one WAV file to enable Dry Run / Run."
            )

        dry_col, run_col, _ = st.columns([1, 1, 4])
        dry_run_clicked = dry_col.button(
            "🔍 Dry Run", use_container_width=True, disabled=not files_ready or run_active
        )
        run_clicked = run_col.button(
            "▶ Run", type="primary", use_container_width=True, disabled=not files_ready or run_active
        )

        log_area = st.empty()
        progress_area = st.empty()

        if run_active:
            _drain_log_queue(log_area)
            event = _read_progress()
            if event and event.get("total"):
                fraction = event["completed"] / event["total"]
                if event["completed"] > 0:
                    remaining_sec = event["elapsed_sec"] * (event["total"] - event["completed"]) / event["completed"]
                    eta_text = f" • ETA {remaining_sec:.0f}s"
                else:
                    eta_text = ""
                progress_area.progress(
                    fraction, text=f"{event['completed']}/{event['total']} files{eta_text}"
                )

            proc = st.session_state["_proc"]
            poll = proc.poll()

            if poll is None:
                if st.button("⏹ Cancel"):
                    proc.terminate()
                    for _attempt in range(10):  # up to ~5s grace period
                        if proc.poll() is not None:
                            break
                        time.sleep(0.5)
                    if proc.poll() is None:
                        proc.kill()
                    st.session_state["_run_cancelled"] = True
                    st.rerun()
                else:
                    time.sleep(1)
                    st.rerun()
            else:
                st.session_state["_run_active"] = False
                rc = poll
                run_kind = st.session_state["_run_kind"]
                cancelled = st.session_state["_run_cancelled"]
                tmp_output = st.session_state["_tmp_output_dir"]

                if cancelled:
                    st.warning("Run cancelled.")
                elif run_kind == "run":
                    if rc == 0:
                        st.success("Pipeline completed successfully.")
                        st.session_state["pipeline_done"] = True
                        _load_outputs(tmp_output)
                    else:
                        st.error(f"Pipeline exited with code {rc}. Check the log above.")
                elif rc != 0:
                    st.error(f"Dry run exited with code {rc}. Check the log above.")

        elif dry_run_clicked or run_clicked:
            content_key = _files_signature(
                list(uploaded_a) + list(uploaded_b) if mode == "dual" else list(uploaded_single)
            )
            tmp_input = str(_TEMP_ROOT / f"a2r_in_{content_key}")
            tmp_output = str(_TEMP_ROOT / f"a2r_out_{content_key}")

            # Input is always rewritten fresh — cheap, and guarantees it
            # matches exactly what's currently uploaded.
            _rm_temp_dir(tmp_input)
            Path(tmp_input).mkdir(parents=True)

            # Output is only wiped when caching is off. Otherwise this is
            # the *same* path as any previous run of this exact upload
            # (path is content-addressed), so out_root/per_file_whisperx_json/
            # from an earlier click is still there and the pipeline's own
            # "if cfg.cache and json_path.exists()" check can actually hit —
            # e.g. Dry Run then Run, or Run again after only toggling
            # diarization/report options, skips re-transcribing.
            if not cache:
                _rm_temp_dir(tmp_output)
            Path(tmp_output).mkdir(parents=True, exist_ok=True)

            st.session_state["_tmp_input_dir"] = tmp_input
            st.session_state["_tmp_output_dir"] = tmp_output
            st.session_state["output_files"] = {}

            if mode == "dual":
                dir_a = Path(tmp_input) / "channel_a prime"
                dir_b = Path(tmp_input) / "channel_b prime"
                dir_a.mkdir()
                dir_b.mkdir()
                for f in uploaded_a:
                    (dir_a / f.name).write_bytes(f.getvalue())
                for f in uploaded_b:
                    (dir_b / f.name).write_bytes(f.getvalue())
                input_path = tmp_input
            else:
                for f in uploaded_single:
                    (Path(tmp_input) / f.name).write_bytes(f.getvalue())
                input_path = tmp_input

            # Not a context manager: this needs to outlive the current
            # script rerun, since the subprocess now runs in the
            # background and reads the config file well after this
            # function returns. Reclaimed later by the same 30-minute
            # stale-dir sweep as everything else under _TEMP_ROOT.
            cfg_tmp = Path(tempfile.mkdtemp(dir=str(_TEMP_ROOT), prefix="a2r_cfg_"))
            cfg_path = _build_inline_config(cfg_tmp)
            progress_file = Path(tmp_output) / "progress.jsonl"
            cmd = [
                sys.executable, "-m", "audio2report.cli.main",
                mode, input_path,
                "--out", tmp_output,
                "--config", str(cfg_path),
                "--progress-file", str(progress_file),
            ]
            if dry_run_clicked:
                cmd.append("--dry-run")
            if run_clicked and llm_enabled:
                cmd += [
                    "--report",
                    "--llm-provider", llm_provider,
                    "--llm-model", llm_model,
                    "--llm-base-url", llm_base_url,
                ]

            _start_subprocess(cmd, progress_file, run_kind="run" if run_clicked else "dry_run")
            st.rerun()

        elif st.session_state["log_lines"]:
            log_area.code(
                "\n".join(st.session_state["log_lines"][-200:]), language=None
            )

        # ── Downloads ─────────────────────────────────────────────────────────
        # st.download_button triggers a full script rerun on every click (like
        # any Streamlit widget) — with one button per output file, downloading
        # several in a row meant repeatedly watching the page dim and recover.
        # A single zip means at most one rerun instead of up to seven.
        output_files = st.session_state.get("output_files") or {}
        if output_files:
            st.divider()
            st.subheader("Downloads")
            _MIME = {
                "json": "application/json",
                "csv": "text/csv",
                "txt": "text/plain",
                "md": "text/markdown",
            }

            zip_buf = io.BytesIO()
            with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
                for fname, data in output_files.items():
                    zf.writestr(fname, data)
            st.download_button(
                "⬇ Download all outputs (.zip)",
                data=zip_buf.getvalue(),
                file_name="audio2report_output.zip",
                mime="application/zip",
                type="primary",
            )

            with st.expander("Download individual files instead"):
                cols = st.columns(4)
                for col_idx, (fname, data) in enumerate(output_files.items()):
                    ext = fname.rsplit(".", 1)[-1]
                    cols[col_idx % 4].download_button(
                        f"⬇ {fname}",
                        data=data,
                        file_name=fname,
                        mime=_MIME.get(ext, "application/octet-stream"),
                        use_container_width=True,
                        key=f"dl_{fname}",
                    )

            with st.expander("What are these files?"):
                st.markdown(
                    """
| File | What it is |
|---|---|
| `canonical_transcript.json` | Full segment list with speaker labels, timestamps, confidence scores, and suppression flags. Primary output — use this for any downstream processing. |
| `canonical_transcript.csv` | Same data in spreadsheet format. Open in Excel or Google Sheets to sort, filter, or annotate. |
| `canonical_transcript.txt` | Plain-text human-readable transcript. Useful for quick review or copy-pasting. |
| `cleaned_llm_input.txt` | Deduplicated transcript with suppressed segments removed. This is what the pipeline feeds to the LLM — paste it into any chat AI for summarisation. |
| `alignment_anchors.json` | Diagnostic file showing the shared utterances used to estimate the clock offset between the two microphones. Useful for debugging alignment issues. |
| `run_meta.json` | Pipeline run statistics: segment counts, suppression rate, estimated mic offset, device, model, and timing. |
| `report.md` | LLM-generated audit report in Markdown. Only present if you enabled report generation. |
"""
                )

        # ── Run metadata panel ─────────────────────────────────────────────────
        if st.session_state["run_meta"]:
            st.divider()
            st.subheader("Run statistics")
            meta = st.session_state["run_meta"]
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total segments", meta.get("total_segments", "—"))
            m2.metric("Kept", meta.get("kept_segments", "—"))
            m3.metric("Suppressed", meta.get("suppressed_segments", "—"))
            m4.metric("Anchors", meta.get("anchor_count", "—"))
            m5.metric("Duplicate pairs", meta.get("pair_match_count", "—"))

            offset = meta.get("estimated_offset_b_minus_a_sec")
            if offset is not None:
                st.caption(
                    f"Clock offset B−A: **{offset:+.3f} s** | "
                    f"Backend: {meta.get('backend', '—')} | "
                    f"Device: {meta.get('device', '—')} | "
                    f"Model: {meta.get('model', '—')}"
                )

    # ── Tab 2: Transcript ─────────────────────────────────────────────────────

    with tab_transcript:
        st.header("Transcript")

        segments = st.session_state.get("transcript_segments")

        if not segments:
            st.info("Run the pipeline first to view the transcript here.")
        else:
            ctrl1, ctrl2, _ = st.columns([2, 2, 2])
            show_suppressed = ctrl1.checkbox("Show suppressed segments", value=False)
            search_text = ctrl2.text_input("Search", placeholder="Filter by text…")

            # Speaker colour map
            speakers = sorted(
                {s.get("speaker_final") or "UNKNOWN" for s in segments}
            )
            _COLOURS = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                "#9467bd", "#8c564b", "#e377c2", "#17becf",
            ]
            speaker_colour = {
                sp: _COLOURS[i % len(_COLOURS)] for i, sp in enumerate(speakers)
            }

            def _ts(sec: float) -> str:
                h = int(sec // 3600)
                m = int((sec % 3600) // 60)
                s = sec % 60
                return f"{h:02d}:{m:02d}:{s:06.3f}"

            visible = [
                s for s in segments
                if (show_suppressed or s.get("keep", True))
                and (
                    not search_text
                    or search_text.lower() in (s.get("text") or "").lower()
                )
            ]

            st.caption(f"Showing {len(visible)} of {len(segments)} segments")

            for seg in visible:
                keep = seg.get("keep", True)
                speaker = seg.get("speaker_final") or "UNKNOWN"
                colour = speaker_colour.get(speaker, "#888888")
                start = seg.get("global_start_sec", 0.0)
                end = seg.get("global_end_sec", 0.0)
                text = seg.get("text") or ""
                flags = seg.get("flags") or []

                opacity = "1.0" if keep else "0.4"
                strike = "text-decoration: line-through;" if not keep else ""
                flag_html = (
                    f' <span style="color:#e07b00;font-size:0.78em">⚑ {", ".join(flags)}</span>'
                    if flags else ""
                )
                suppressed_label = (
                    ' <span style="color:#888;font-size:0.78em">(suppressed)</span>'
                    if not keep else ""
                )

                st.markdown(
                    f'<div style="border-left:4px solid {colour};padding:4px 10px;'
                    f'margin-bottom:4px;opacity:{opacity};">'
                    f'<span style="color:{colour};font-weight:bold">{speaker}</span> '
                    f'<span style="color:#888;font-size:0.82em">'
                    f'[{_ts(start)} → {_ts(end)}]</span>'
                    f'{flag_html}'
                    f'<br><span style="{strike}">{text}</span>'
                    f'{suppressed_label}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.divider()
            output_files = st.session_state.get("output_files") or {}
            dl1, dl2 = st.columns(2)

            if "cleaned_llm_input.txt" in output_files:
                dl1.download_button(
                    "⬇ cleaned_llm_input.txt",
                    data=output_files["cleaned_llm_input.txt"],
                    file_name="cleaned_llm_input.txt",
                    mime="text/plain",
                )

            if "canonical_transcript.csv" in output_files:
                dl2.download_button(
                    "⬇ canonical_transcript.csv",
                    data=output_files["canonical_transcript.csv"],
                    file_name="canonical_transcript.csv",
                    mime="text/csv",
                )

    # ── Tab 3: Report ─────────────────────────────────────────────────────────

    with tab_report:
        st.header("LLM Report")

        report_text = st.session_state.get("report_text")
        output_files = st.session_state.get("output_files") or {}

        if report_text:
            st.success("Report loaded from pipeline output.")
            st.markdown(report_text)
            st.divider()
            st.download_button(
                "⬇ Download report.md",
                data=output_files.get("report.md", report_text.encode("utf-8")),
                file_name="report.md",
                mime="text/markdown",
            )
        else:
            st.info(
                "No report found yet.  "
                "Either enable **Generate report after pipeline** in the sidebar and "
                "re-run, or generate a report from an existing transcript below."
            )

        st.divider()
        st.subheader("Generate report from transcript")

        tmp_out = st.session_state.get("_tmp_output_dir") or ""
        default_transcript = (
            str(Path(tmp_out) / "canonical_transcript.json") if tmp_out else ""
        )
        transcript_path = st.text_input(
            "Transcript file",
            value=default_transcript,
            placeholder="/path/to/canonical_transcript.json",
        )

        generate_clicked = st.button("📋 Generate report", type="primary")

        if generate_clicked:
            if not transcript_path or not Path(transcript_path).exists():
                st.error("Transcript file not found.")
            else:
                report_out_dir = str(Path(transcript_path).parent)
                report_file = str(Path(report_out_dir) / "report.md")
                cmd = [
                    sys.executable, "-m", "audio2report.cli.main",
                    "report", transcript_path,
                    "--provider", llm_provider,
                    "--model", llm_model,
                    "--base-url", llm_base_url,
                    "--out", report_file,
                ]
                if llm_api_key:
                    cmd += ["--api-key", llm_api_key]
                if not llm_stream:
                    cmd.append("--no-stream")

                with st.spinner("Generating report…"):
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                if result.returncode == 0:
                    report_path = Path(report_out_dir) / "report.md"
                    if report_path.exists():
                        report_bytes = report_path.read_bytes()
                        st.session_state["report_text"] = report_bytes.decode("utf-8")
                        files = dict(st.session_state.get("output_files") or {})
                        files["report.md"] = report_bytes
                        st.session_state["output_files"] = files
                        st.rerun()
                    else:
                        st.warning("Command succeeded but report.md was not found.")
                        if result.stdout:
                            st.text(result.stdout)
                else:
                    st.error("Report generation failed.")
                    st.code(
                        result.stderr or result.stdout or "(no output)", language=None
                    )
