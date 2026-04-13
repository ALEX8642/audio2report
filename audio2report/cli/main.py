"""audio2report CLI — built with Typer.

Commands
--------
dual    Two mic folders → aligned, deduplicated transcript
single  One file or folder → transcript
report  Existing transcript → LLM-generated audit report
config  Utilities for managing config files
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel

import audio2report
from audio2report._log import set_log_level
from audio2report.config import Config, load_config
from audio2report.output.writers import write_all_outputs

load_dotenv()

app = typer.Typer(
    name="audio2report",
    help=(
        "[bold]audio2report[/bold] — Dual-mic cross-talk deduplication "
        "and transcription pipeline."
    ),
    rich_markup_mode="rich",
    no_args_is_help=True,
)

console = Console()

_VERBOSE_HELP = "Enable DEBUG-level logging."
_QUIET_HELP   = "Suppress INFO messages; show warnings and errors only."


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _apply_overrides(cfg: Config, **kwargs) -> Config:
    data = cfg.model_dump()

    mapping = {
        "model":        ("transcription", "model"),
        "language":     ("transcription", "language"),
        "compute_type": ("transcription", "compute_type"),
        "batch_size":   ("transcription", "batch_size"),
        "device":       ("transcription", "device"),
        "hf_token":     ("diarization",   "hf_token"),
        "diarize":      ("diarization",   "enabled"),
    }

    for key, value in kwargs.items():
        if value is None:
            continue
        if key == "no_cache":
            data["cache"] = not value
            continue
        if key == "formats" and value:
            data["output"]["formats"] = list(value)
            continue
        dest = mapping.get(key)
        if dest:
            section, field = dest
            data[section][field] = value

    return Config.model_validate(data)


def _apply_verbosity(verbose: bool, quiet: bool) -> None:
    if verbose:
        set_log_level(logging.DEBUG)
    elif quiet:
        set_log_level(logging.WARNING)


def _run_report_step(
    out_root: Path,
    cfg: Config,
    meta=None,
) -> None:
    """Called after a pipeline run when --report is set."""
    from audio2report.llm.report import generate_report, load_transcript_text

    # Prefer the clean LLM input; fall back to canonical JSON
    clean_txt = out_root / "cleaned_llm_input.txt"
    canon_json = out_root / "canonical_transcript.json"
    if clean_txt.exists():
        transcript_text = load_transcript_text(clean_txt)
    elif canon_json.exists():
        transcript_text = load_transcript_text(canon_json)
    else:
        console.print("[red]No transcript file found for report generation.[/red]")
        return

    console.rule("[bold cyan]Generating Report[/bold cyan]")
    console.print(
        f"Provider: [bold]{cfg.llm.provider}[/bold]  "
        f"Model: [bold]{cfg.llm.model}[/bold]\n"
    )

    try:
        report_text = generate_report(
            transcript_text,
            cfg.llm,
            meta=meta,
            stream_to_stdout=cfg.llm.stream,
        )
    except RuntimeError as exc:
        console.print(f"\n[red]LLM error:[/red] {exc}")
        return

    report_path = out_root / "report.md"
    prompt_path = out_root / "report_prompt.txt"

    report_path.write_text(report_text, encoding="utf-8")

    # Save prompt for transparency
    from audio2report.llm.report import build_prompt
    prompt = build_prompt(transcript_text, cfg.llm, meta=meta)
    prompt_path.write_text(prompt, encoding="utf-8")

    console.print(f"\n[green]Report saved:[/green] {report_path}")


# ---------------------------------------------------------------------------
# dual command
# ---------------------------------------------------------------------------

@app.command()
def dual(
    root: Path = typer.Argument(..., help="Root folder containing exactly two 'prime' subfolders."),
    out: Path = typer.Option(Path("audio2report_output"), "--out", "-o", help="Output directory."),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file."),
    model: Optional[str] = typer.Option(None, help="WhisperX model (e.g. large-v3, medium)."),
    language: Optional[str] = typer.Option(None, help="Force language code (e.g. en). Default: auto-detect."),
    compute_type: Optional[str] = typer.Option(None, help="Compute type: float16 | int8 | float32."),
    batch_size: Optional[int] = typer.Option(None, help="WhisperX batch size."),
    device: Optional[str] = typer.Option(None, help="Device: cuda | cpu."),
    hf_token: Optional[str] = typer.Option(None, envvar="HF_TOKEN", help="Hugging Face token for diarization."),
    diarize: bool = typer.Option(False, "--diarize/--no-diarize", help="Enable speaker diarization."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Re-transcribe even if cached JSON exists."),
    formats: Optional[List[str]] = typer.Option(None, "--format", help="Output formats: json, csv, txt (repeatable)."),
    generate_report: bool = typer.Option(False, "--report", help="Generate an LLM audit report after transcription."),
    llm_provider: Optional[str] = typer.Option(None, "--llm-provider", help="LLM provider: ollama | openai."),
    llm_model: Optional[str] = typer.Option(None, "--llm-model", help="LLM model name."),
    llm_base_url: Optional[str] = typer.Option(None, "--llm-base-url", help="LLM API base URL."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed without running."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help=_VERBOSE_HELP),
    quiet: bool = typer.Option(False, "--quiet", "-q", help=_QUIET_HELP),
) -> None:
    """
    [bold]Dual-mic pipeline[/bold]: align two microphone folders, deduplicate cross-talk,
    and produce an attributed transcript.

    Examples:

        audio2report dual ./recordings --out ./output --diarize

        audio2report dual ./recordings --config configs/gpu_full.yaml

        audio2report dual ./recordings --report --llm-provider ollama --llm-model llama3

        audio2report dual ./recordings --dry-run
    """
    from audio2report.pipeline.dual import DualMicPipeline

    _apply_verbosity(verbose, quiet)

    cfg = load_config(config_file)
    cfg = _apply_overrides(
        cfg,
        model=model, language=language, compute_type=compute_type,
        batch_size=batch_size, device=device, hf_token=hf_token,
        diarize=diarize, no_cache=no_cache, formats=formats,
    )
    # Apply LLM overrides
    if llm_provider or llm_model or llm_base_url or generate_report:
        llm_data = cfg.llm.model_dump()
        if llm_provider:
            llm_data["provider"] = llm_provider
        if llm_model:
            llm_data["model"] = llm_model
        if llm_base_url:
            llm_data["base_url"] = llm_base_url
        if generate_report:
            llm_data["enabled"] = True
        from audio2report.config import LLMConfig
        cfg = cfg.model_copy(update={"llm": LLMConfig.model_validate(llm_data)})

    if not quiet:
        console.print(
            Panel(
                f"[bold]audio2report[/bold] v{audio2report.__version__}  •  mode: [cyan]dual[/cyan]\n"
                f"root : {root.resolve()}\n"
                f"out  : {out.resolve()}\n"
                f"model: {cfg.transcription.model}  "
                f"device: {cfg.transcription.device or 'auto'}  "
                f"diarize: {cfg.diarization.enabled}  "
                f"cache: {cfg.cache}",
                title="Run config",
                border_style="dim",
            )
        )

    pipeline = DualMicPipeline(cfg)

    if dry_run:
        pipeline.dry_run(root.resolve(), out.resolve())
        return

    try:
        result = pipeline.run(root.resolve(), out.resolve())
        write_all_outputs(result, out.resolve(), cfg.output)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    meta = result.meta
    if not quiet:
        console.print(
            Panel(
                f"[green]Done.[/green]\n"
                f"Anchors: {meta.anchor_count}  •  "
                f"Offset: {meta.estimated_offset_b_minus_a_sec:+.3f} s\n"
                f"Segments: {meta.total_segments} total  •  "
                f"{meta.kept_segments} kept  •  "
                f"{meta.suppressed_segments} suppressed\n"
                f"Output: [bold]{out.resolve()}[/bold]",
                title="Result",
                border_style="green",
            )
        )

    if cfg.llm.enabled:
        _run_report_step(out.resolve(), cfg, meta=meta)


# ---------------------------------------------------------------------------
# single command
# ---------------------------------------------------------------------------

@app.command()
def single(
    input_path: Path = typer.Argument(..., metavar="INPUT", help="Audio file or folder."),
    out: Path = typer.Option(Path("audio2report_output"), "--out", "-o", help="Output directory."),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file."),
    prime_name: Optional[str] = typer.Option(None, "--speaker", help="Speaker name for attribution."),
    model: Optional[str] = typer.Option(None, help="WhisperX model name."),
    language: Optional[str] = typer.Option(None, help="Force language code."),
    compute_type: Optional[str] = typer.Option(None, help="Compute type: float16 | int8 | float32."),
    batch_size: Optional[int] = typer.Option(None, help="WhisperX batch size."),
    device: Optional[str] = typer.Option(None, help="Device: cuda | cpu."),
    hf_token: Optional[str] = typer.Option(None, envvar="HF_TOKEN"),
    diarize: bool = typer.Option(False, "--diarize/--no-diarize"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    formats: Optional[List[str]] = typer.Option(None, "--format"),
    generate_report: bool = typer.Option(False, "--report", help="Generate an LLM report after transcription."),
    llm_provider: Optional[str] = typer.Option(None, "--llm-provider"),
    llm_model: Optional[str] = typer.Option(None, "--llm-model"),
    llm_base_url: Optional[str] = typer.Option(None, "--llm-base-url"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be processed without running."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help=_VERBOSE_HELP),
    quiet: bool = typer.Option(False, "--quiet", "-q", help=_QUIET_HELP),
) -> None:
    """
    [bold]Single-mic pipeline[/bold]: transcribe one audio file or folder.

    Examples:

        audio2report single meeting.mp3 --out ./output

        audio2report single ./recordings_folder --diarize --speaker Alice

        audio2report single meeting.mp3 --report --llm-provider ollama
    """
    from audio2report.pipeline.single import SingleMicPipeline

    _apply_verbosity(verbose, quiet)

    cfg = load_config(config_file)
    cfg = _apply_overrides(
        cfg,
        model=model, language=language, compute_type=compute_type,
        batch_size=batch_size, device=device, hf_token=hf_token,
        diarize=diarize, no_cache=no_cache, formats=formats,
    )
    if llm_provider or llm_model or llm_base_url or generate_report:
        llm_data = cfg.llm.model_dump()
        if llm_provider:  llm_data["provider"]  = llm_provider
        if llm_model:     llm_data["model"]      = llm_model
        if llm_base_url:  llm_data["base_url"]   = llm_base_url
        if generate_report: llm_data["enabled"]  = True
        from audio2report.config import LLMConfig
        cfg = cfg.model_copy(update={"llm": LLMConfig.model_validate(llm_data)})

    if dry_run:
        from rich.table import Table
        table = Table(title=f"Dry run — {input_path}", show_lines=True)
        table.add_column("File")
        table.add_column("Status", style="yellow")
        if input_path.is_dir():
            from audio2report.ingestion.discovery import list_audio_files
            for f in list_audio_files(input_path):
                table.add_row(f.name, "will transcribe")
        else:
            table.add_row(input_path.name, "will transcribe")
        console.print(table)
        return

    if not quiet:
        console.print(
            Panel(
                f"[bold]audio2report[/bold] v{audio2report.__version__}  •  mode: [cyan]single[/cyan]\n"
                f"input: {input_path.resolve()}\n"
                f"out  : {out.resolve()}",
                title="Run config",
                border_style="dim",
            )
        )

    try:
        pipeline = SingleMicPipeline(cfg)
        result = pipeline.run(input_path.resolve(), out.resolve(), prime_name=prime_name)
        write_all_outputs(result, out.resolve(), cfg.output)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    meta = result.meta
    if not quiet:
        console.print(
            Panel(
                f"[green]Done.[/green]  "
                f"{meta.kept_segments} segment(s)  •  "
                f"Output: [bold]{out.resolve()}[/bold]",
                title="Result",
                border_style="green",
            )
        )

    if cfg.llm.enabled:
        _run_report_step(out.resolve(), cfg, meta=meta)


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------

@app.command()
def report(
    transcript: Path = typer.Argument(
        ...,
        help=(
            "Path to transcript file.  Accepts:\n"
            "  canonical_transcript.json   (full trace with speaker labels)\n"
            "  cleaned_llm_input.txt       (post-processed plain text)"
        ),
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o",
        help="Output path for the report (default: report.md next to the transcript).",
    ),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="YAML config file."),
    provider: Optional[str] = typer.Option(None, "--provider", help="LLM provider: ollama | openai."),
    llm_model: Optional[str] = typer.Option(None, "--model", help="LLM model name."),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="LLM API base URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", envvar="OPENAI_API_KEY"),
    template: Optional[str] = typer.Option(None, "--template", help="Template name or path to .txt file."),
    no_stream: bool = typer.Option(False, "--no-stream", help="Disable streaming output."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help=_VERBOSE_HELP),
    quiet: bool = typer.Option(False, "--quiet", "-q", help=_QUIET_HELP),
) -> None:
    """
    [bold]Generate an LLM audit report[/bold] from an existing transcript.

    Examples:

        # Using local Ollama (default)
        audio2report report ./output/canonical_transcript.json

        # Using OpenAI
        audio2report report ./output/cleaned_llm_input.txt \\
            --provider openai --model gpt-4o --base-url https://api.openai.com/v1

        # Custom template
        audio2report report ./output/cleaned_llm_input.txt --template ./my_template.txt
    """
    from audio2report.llm.report import build_prompt, generate_report, load_transcript_text

    _apply_verbosity(verbose, quiet)

    cfg = load_config(config_file)

    # Apply report-specific overrides
    llm_data = cfg.llm.model_dump()
    if provider:   llm_data["provider"] = provider
    if llm_model:  llm_data["model"]    = llm_model
    if base_url:   llm_data["base_url"] = base_url
    if api_key:    llm_data["api_key"]  = api_key
    if template:   llm_data["prompt_template"] = template
    if no_stream:  llm_data["stream"]   = False
    llm_data["enabled"] = True

    from audio2report.config import LLMConfig
    cfg = cfg.model_copy(update={"llm": LLMConfig.model_validate(llm_data)})

    # Resolve output path
    report_out = out or transcript.parent / "report.md"
    prompt_out = report_out.parent / "report_prompt.txt"

    if not quiet:
        console.print(
            Panel(
                f"Transcript : {transcript}\n"
                f"Provider   : [bold]{cfg.llm.provider}[/bold]  "
                f"Model: [bold]{cfg.llm.model}[/bold]\n"
                f"Report out : {report_out}",
                title="[bold]audio2report[/bold] — report",
                border_style="dim",
            )
        )

    # Load transcript
    try:
        transcript_text = load_transcript_text(transcript)
    except Exception as exc:
        console.print(f"[red]Failed to load transcript:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not transcript_text.strip():
        console.print("[yellow]Warning:[/yellow] Transcript appears empty.")

    console.rule("[bold cyan]Generating Report[/bold cyan]")

    try:
        report_text = generate_report(
            transcript_text,
            cfg.llm,
            stream_to_stdout=cfg.llm.stream,
        )
    except RuntimeError as exc:
        console.print(f"\n[red]LLM error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    # Save report and prompt
    report_out.write_text(report_text, encoding="utf-8")
    prompt = build_prompt(transcript_text, cfg.llm)
    prompt_out.write_text(prompt, encoding="utf-8")

    if not quiet:
        console.print(
            Panel(
                f"[green]Report saved:[/green] {report_out}\n"
                f"Prompt saved: {prompt_out}",
                border_style="green",
            )
        )


# ---------------------------------------------------------------------------
# config subcommand group
# ---------------------------------------------------------------------------

config_app = typer.Typer(name="config", help="Config file utilities.", no_args_is_help=True)
app.add_typer(config_app)


@config_app.command("init")
def config_init(
    output: Path = typer.Option(Path("config.yaml"), "--output", "-o", help="Destination path."),
    preset: str = typer.Option("default", "--preset", help="Preset: default | cpu_fast | gpu_full."),
) -> None:
    """Generate a starter YAML config file from a built-in preset."""
    import shutil
    configs_dir = Path(__file__).parent.parent.parent / "configs"
    src = configs_dir / f"{preset}.yaml"
    if not src.exists():
        console.print(f"[red]Preset '{preset}' not found.[/red]  "
                      f"Available: default, cpu_fast, gpu_full")
        raise typer.Exit(code=1)
    shutil.copy(src, output)
    console.print(f"[green]Config written to {output}[/green]")


@config_app.command("show")
def config_show(
    config_file: Optional[Path] = typer.Argument(None, help="YAML config to display."),
) -> None:
    """Print the effective config as JSON."""
    import json as _json
    cfg = load_config(config_file)
    console.print_json(_json.dumps(cfg.model_dump(), indent=2))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    app()


if __name__ == "__main__":
    main()
