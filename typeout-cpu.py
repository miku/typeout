#!/usr/bin/env -S uv run
# /// script
# dependencies = [
#     "click",
#     "rich",
#     "yt-dlp",
#     "ffmpeg-python",
#     "openai-whisper",
#     "torch",
#     "torchaudio",
#     "transformers>=4.52,!=5.0.*,!=5.1.*",
#     "soundfile",
#     "librosa",
#     "sentencepiece",
#     "protobuf",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# torchaudio = { index = "pytorch-cpu" }
# ///

"""
typeout-cpu.py - CPU backend for typeout transcription.

Transcribe audio/video to text using Whisper or Cohere Transcribe (CPU).

Input can be a local audio/video file (any format ffmpeg supports),
a URL, or a YouTube video ID.
"""

import click
import sys
import os
import re
import hashlib
import tempfile
import shutil
from pathlib import Path
from rich.console import Console
import yt_dlp
import ffmpeg
import whisper

console = Console(stderr=True)

APP_NAME = "typeout"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_data_dir() -> Path:
    xdg = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    d = Path(xdg) / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_cache_dir() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    d = Path(xdg) / APP_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Cache keys
# ---------------------------------------------------------------------------

def source_key(input_source: str, input_type: str) -> str:
    if input_type == "url":
        return hashlib.sha256(input_source.encode()).hexdigest()[:16]
    p = Path(input_source).resolve()
    st = p.stat()
    raw = f"{p}:{st.st_mtime_ns}:{st.st_size}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def transcript_key(src_key: str, model: str, lang: str = "en") -> str:
    raw = f"{src_key}:{model}:{lang}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Input detection
# ---------------------------------------------------------------------------

def detect_input_type(input_str: str) -> str:
    """Classify input as file, url, or youtube_id."""
    if os.path.exists(input_str):
        return "file"
    if re.match(r"^https?://", input_str):
        return "url"
    if re.match(r"^[A-Za-z0-9_-]{11}$", input_str):
        return "youtube_id"
    console.print(f"[red]Not a file, URL, or YouTube ID:[/red] {input_str}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Media info
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def print_media_info(input_path: str, input_type: str):
    try:
        if input_type == "url":
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                info = ydl.extract_info(input_path, download=False)
            if info.get("title"):
                console.print(f"[dim]Title:[/dim] {info['title']}")
            parts = []
            if info.get("duration"):
                parts.append(format_duration(info["duration"]))
            if info.get("ext"):
                parts.append(info["ext"])
            if info.get("resolution") and info["resolution"] != "audio only":
                parts.append(info["resolution"])
            if parts:
                console.print(f"[dim]{', '.join(parts)}[/dim]")
        else:
            probe = ffmpeg.probe(input_path)
            fmt = probe.get("format", {})
            parts = []
            if fmt.get("duration"):
                parts.append(format_duration(float(fmt["duration"])))
            if fmt.get("format_long_name"):
                parts.append(fmt["format_long_name"])
            for s in probe.get("streams", []):
                if s.get("codec_type") == "video":
                    w, h = s.get("width"), s.get("height")
                    if w and h:
                        parts.append(f"{w}x{h}")
                    if s.get("nb_frames") and s["nb_frames"] != "N/A":
                        parts.append(f"{s['nb_frames']} frames")
                    break
            if parts:
                console.print(f"[dim]{', '.join(parts)}[/dim]")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Audio extraction & normalization
# ---------------------------------------------------------------------------

def normalize_audio(input_path: str, output_path: str) -> str:
    """Convert any media file to 16 kHz mono WAV via ffmpeg."""
    try:
        (
            ffmpeg.input(input_path)
            .output(output_path, ar="16000", ac=1)
            .run(quiet=True, overwrite_output=True)
        )
        return output_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if e.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg error: {stderr}")


def download_url(url: str, output_path: str) -> str:
    """Download audio from a URL via yt-dlp, then normalize."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as raw:
        raw_path = raw.name
    try:
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": raw_path[:-4],
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # yt-dlp may append to the base name
        base = raw_path[:-4].split("/")[-1]
        downloaded = raw_path
        for f in os.listdir(os.path.dirname(raw_path)):
            if f.startswith(base) and f.endswith(".wav"):
                downloaded = os.path.join(os.path.dirname(raw_path), f)
                break
        return normalize_audio(downloaded, output_path)
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)


def prepare_audio(input_source: str, input_type: str, output_path: str) -> str:
    """Get normalized 16 kHz mono WAV from any source."""
    console.print("[dim]Preparing audio...[/dim]")
    if input_type == "url":
        return download_url(input_source, output_path)
    return normalize_audio(input_source, output_path)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = {
    "tiny": {"type": "whisper", "description": "Whisper tiny (fastest, lowest accuracy)"},
    "base": {"type": "whisper", "description": "Whisper base (default)"},
    "small": {"type": "whisper", "description": "Whisper small"},
    "medium": {"type": "whisper", "description": "Whisper medium"},
    "large": {"type": "whisper", "description": "Whisper large (slowest, highest accuracy)"},
    "distil-large-v3": {
        "type": "distil-whisper",
        "pretrained": "distil-whisper/distil-large-v3",
        "multilingual": True,
        "description": "Distil-Whisper large-v3 (~750MB, 6x faster than Whisper large)",
    },
    "distil-medium.en": {
        "type": "distil-whisper",
        "pretrained": "distil-whisper/distil-medium.en",
        "multilingual": False,
        "description": "Distil-Whisper medium English-only (~400MB, fast)",
    },
    "cohere-transcribe": {
        "type": "cohere",
        "pretrained": "CohereLabs/cohere-transcribe-03-2026",
        "description": "Cohere Transcribe 2B (14 languages, high accuracy)",
    },
}

DEFAULT_MODEL = "base"


def print_models():
    console.print("\n[bold]Available models:[/bold]\n")
    for name, cfg in MODELS.items():
        default = " [green](default)[/green]" if name == DEFAULT_MODEL else ""
        console.print(f"  [bold]{name}[/bold]{default}")
        console.print(f"    {cfg['description']}")
    console.print()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

def transcribe(audio_path: str, model_name: str, lang: str = "en") -> str:
    """Transcribe audio using the specified model."""
    model_cfg = MODELS.get(model_name, {"type": "whisper"})

    if model_cfg.get("type") == "distil-whisper":
        return _transcribe_distil_whisper(audio_path, model_cfg, lang)

    if model_cfg.get("type") == "cohere":
        return _transcribe_cohere(audio_path, model_cfg, lang)

    # Whisper (default)
    data_dir = get_data_dir() / "whisper"
    data_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Loading Whisper model:[/dim] {model_name}")
    model = whisper.load_model(model_name, download_root=str(data_dir))
    console.print("[dim]Transcribing...[/dim]")
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]


def _transcribe_distil_whisper(audio_path: str, model_cfg: dict, lang: str) -> str:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cpu"
    console.print(f"[dim]Loading model:[/dim] {model_cfg['pretrained']}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_cfg["pretrained"], dtype=torch.float32,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_cfg["pretrained"])

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch.float32,
        device=device,
    )

    generate_kwargs = {}
    if model_cfg.get("multilingual"):
        generate_kwargs["language"] = lang

    console.print("[dim]Transcribing...[/dim]")
    result = pipe(audio_path, chunk_length_s=30, return_timestamps=True,
                  generate_kwargs=generate_kwargs)
    return result["text"]


def _transcribe_cohere(audio_path: str, model_cfg: dict, lang: str) -> str:
    """Transcribe using Cohere Transcribe model (CPU)."""
    import torch

    device = "cpu"
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    data_dir = get_data_dir() / "models" / model_cfg["pretrained"].replace("/", "_")
    data_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"[dim]Loading model:[/dim] {model_cfg['pretrained']}")
    processor = AutoProcessor.from_pretrained(
        model_cfg["pretrained"], trust_remote_code=True, cache_dir=str(data_dir)
    )
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_cfg["pretrained"], trust_remote_code=True, cache_dir=str(data_dir)
    ).to(device)
    model.eval()

    console.print("[dim]Transcribing...[/dim]")
    texts = model.transcribe(
        processor=processor,
        audio_files=[audio_path],
        language=lang,
    )
    return texts[0]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("input_source", required=False)
@click.option("--model", default="base",
              help="Model name (use --list-models to see all)")
@click.option("--lang", default="en",
              help="Language code (e.g., en, es, fr, de, ja, zh, ko, ar, vi, nl, pl, pt, el, it)")
@click.option("--output", "-o", type=click.Path(), help="Write transcript to file")
@click.option("--no-cache", is_flag=True, help="Bypass cache")
@click.option("--clear-cache", is_flag=True, help="Remove all cached data")
@click.option("--list-models", is_flag=True, help="List available models")
@click.option("--check", is_flag=True, help="Check external tools")
@click.version_option(version=os.environ.get("TYPEOUT_VERSION", "dev"))
def cli(input_source, model, output, no_cache, clear_cache, list_models, check, lang):
    """Transcribe audio or video to text (CPU).

    INPUT_SOURCE can be a local file (any format ffmpeg supports),
    a URL, or a YouTube video ID. Use --list-models to see available models.
    """
    if list_models:
        print_models()
        return

    if clear_cache:
        shutil.rmtree(get_cache_dir(), ignore_errors=True)
        console.print("[green]Cache cleared.[/green]")
        return

    if check:
        tools = {"ffmpeg": shutil.which("ffmpeg")}
        for name, path in tools.items():
            s = "[green]ok[/green]" if path else "[red]missing[/red]"
            console.print(f"  {name}: {s}")
        if not all(tools.values()):
            sys.exit(1)
        return

    if not input_source:
        console.print("[red]Please provide an input file, URL, or YouTube ID.[/red]")
        raise SystemExit(1)

    # Resolve input
    input_type = detect_input_type(input_source)
    if input_type == "youtube_id":
        input_source = f"https://www.youtube.com/watch?v={input_source}"
        input_type = "url"

    console.print(f"[blue]Input:[/blue] {input_source}")
    print_media_info(input_source, input_type)

    # Cache lookup
    cache_dir = get_cache_dir()
    src_key = source_key(input_source, input_type)
    t_key = transcript_key(src_key, model, lang)
    transcript_cache = cache_dir / "transcripts" / f"{t_key}.txt"

    if not no_cache and transcript_cache.exists():
        text = transcript_cache.read_text()
        console.print("[dim]cached[/dim]")
        if output:
            Path(output).write_text(text)
            console.print(f"[green]Saved:[/green] {output}")
        else:
            print(text)
        return

    # Audio cache (URLs only — local files are already on disk)
    audio_cache = cache_dir / "audio" / f"{src_key}.wav"
    tmp_audio = None

    if not no_cache and input_type == "url" and audio_cache.exists():
        console.print("[dim]Using cached audio[/dim]")
        audio_path = str(audio_cache)
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_audio = tmp.name
        audio_path = prepare_audio(input_source, input_type, tmp_audio)
        if input_type == "url":
            audio_cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audio_path, audio_cache)

    try:
        text = transcribe(audio_path, model, lang)

        transcript_cache.parent.mkdir(parents=True, exist_ok=True)
        transcript_cache.write_text(text)

        if output:
            Path(output).write_text(text)
            console.print(f"[green]Saved:[/green] {output}")
        else:
            print(text)
    finally:
        if tmp_audio and os.path.exists(tmp_audio):
            os.remove(tmp_audio)


if __name__ == "__main__":
    cli()
