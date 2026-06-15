"""audio2report Streamlit UI.

Launch with:
    audio2report-ui                       # via installed entry point
    streamlit run audio2report/ui/app.py  # directly
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Entry point (called by the audio2report-ui console script)
# ---------------------------------------------------------------------------

def main() -> None:
    """Launch Streamlit with this file as the app."""
    app_path = Path(__file__).resolve()
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "streamlit", "run", str(app_path),
             "--server.headless", "false"]
            + sys.argv[1:]
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
            "Install it with:  pip install 'audio2report[ui]'"
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
            "output_dir": None,
            "transcript_segments": None,
            "report_text": None,
            "run_meta": None,
            "_tmp_input_dir": None,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    _init_state()

    # ── Sidebar — configuration ───────────────────────────────────────────────

    with st.sidebar:
        st.title("🎙️ audio2report")
        st.caption("Dual-mic transcription & report generation")
        st.divider()

        st.subheader("Mode")
        mode = st.radio("Pipeline mode", ["dual", "single"], horizontal=True)

        st.subheader("Transcription")
        whisper_model = st.selectbox(
            "Whisper model",
            ["large-v3", "large-v2", "medium", "small", "base", "tiny"],
            index=0,
        )
        language = st.text_input("Language (blank = auto-detect)", value="")
        device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
        compute_type = st.selectbox("Compute type", ["auto", "float16", "int8"], index=0)

        st.subheader("Diarization")
        diarize = st.checkbox("Enable speaker diarization", value=False)
        hf_token = ""
        if diarize:
            hf_token = st.text_input("HuggingFace token", type="password")

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

    def _stream_subprocess(cmd: list[str], log_area) -> int:
        """Run *cmd*, streaming stdout/stderr into a Streamlit code block."""
        st.session_state["log_lines"] = []
        log_text = log_area.code("Starting…", language=None)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            st.session_state["log_lines"].append(line.rstrip())
            visible = "\n".join(st.session_state["log_lines"][-200:])
            log_text.code(visible, language=None)

        try:
            proc.wait(timeout=3600)
        except subprocess.TimeoutExpired:
            proc.kill()
            st.error("Pipeline timed out after 60 minutes.")
        return proc.returncode

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

        output_dir_input = st.text_input(
            "Output directory",
            value=str(Path.home() / "audio2report_out"),
            help="Transcript, CSV, and report files are written here.",
        )

        dry_col, run_col, _ = st.columns([1, 1, 4])
        dry_run_clicked = dry_col.button("🔍 Dry Run", use_container_width=True)
        run_clicked = run_col.button("▶ Run", type="primary", use_container_width=True)

        log_area = st.empty()

        if dry_run_clicked or run_clicked:
            if not files_ready:
                if mode == "dual":
                    st.error("Upload at least one WAV file for each channel.")
                else:
                    st.error("Upload at least one WAV file.")
            else:
                # Clean up previous temp input dir from a prior run
                prev_tmp = st.session_state.get("_tmp_input_dir")
                if prev_tmp and Path(prev_tmp).exists():
                    shutil.rmtree(prev_tmp, ignore_errors=True)

                tmp_input = tempfile.mkdtemp(prefix="a2r_in_")
                st.session_state["_tmp_input_dir"] = tmp_input

                if mode == "dual":
                    dir_a = Path(tmp_input) / "channel_a"
                    dir_b = Path(tmp_input) / "channel_b"
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

                out_dir = output_dir_input.strip() or str(Path.home() / "audio2report_out")
                Path(out_dir).mkdir(parents=True, exist_ok=True)

                with tempfile.TemporaryDirectory() as cfg_tmp:
                    cfg_path = _build_inline_config(Path(cfg_tmp))
                    cmd = [
                        sys.executable, "-m", "audio2report.cli.main",
                        mode, input_path,
                        "--output-dir", out_dir,
                        "--config", str(cfg_path),
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
                        if llm_api_key:
                            cmd += ["--llm-api-key", llm_api_key]

                    with st.spinner("Running…"):
                        rc = _stream_subprocess(cmd, log_area)

                if run_clicked:
                    if rc == 0:
                        st.success("Pipeline completed successfully.")
                        st.session_state["pipeline_done"] = True
                        st.session_state["output_dir"] = out_dir
                        _load_outputs(out_dir)
                    else:
                        st.error(f"Pipeline exited with code {rc}. Check the log above.")

        elif st.session_state["log_lines"]:
            log_area.code(
                "\n".join(st.session_state["log_lines"][-200:]), language=None
            )

        # Run metadata panel
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
            out_dir_path = st.session_state.get("output_dir") or ""

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
            dl1, dl2 = st.columns(2)

            if out_dir_path:
                txt_path = Path(out_dir_path) / "cleaned_llm_input.txt"
                if txt_path.exists():
                    dl1.download_button(
                        "⬇ cleaned_llm_input.txt",
                        data=txt_path.read_bytes(),
                        file_name="cleaned_llm_input.txt",
                        mime="text/plain",
                    )

                csv_path = Path(out_dir_path) / "canonical_transcript.csv"
                if csv_path.exists():
                    dl2.download_button(
                        "⬇ canonical_transcript.csv",
                        data=csv_path.read_bytes(),
                        file_name="canonical_transcript.csv",
                        mime="text/csv",
                    )

    # ── Tab 3: Report ─────────────────────────────────────────────────────────

    with tab_report:
        st.header("LLM Report")

        report_text = st.session_state.get("report_text")
        out_dir_path = st.session_state.get("output_dir") or ""

        if report_text:
            st.success("Report loaded from pipeline output.")
            st.markdown(report_text)
            st.divider()
            st.download_button(
                "⬇ Download report.md",
                data=report_text.encode("utf-8"),
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

        default_transcript = (
            str(Path(out_dir_path) / "canonical_transcript.json")
            if out_dir_path else ""
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
                _report_out = str(Path(transcript_path).parent)
                cmd = [
                    sys.executable, "-m", "audio2report.cli.main",
                    "report", transcript_path,
                    "--provider", llm_provider,
                    "--model", llm_model,
                    "--base-url", llm_base_url,
                    "--output-dir", _report_out,
                ]
                if llm_api_key:
                    cmd += ["--api-key", llm_api_key]
                if not llm_stream:
                    cmd.append("--no-stream")

                with st.spinner("Generating report…"):
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

                if result.returncode == 0:
                    report_file = Path(_report_out) / "report.md"
                    if report_file.exists():
                        st.session_state["report_text"] = report_file.read_text(
                            encoding="utf-8"
                        )
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
