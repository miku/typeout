#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "click",
#     "rich",
#     "yt-dlp",
#     "ffmpeg-python",
#     "nemo_toolkit[asr]",
# ]
# ///

"""
videototext-gpu.py - Convert video/audio to text using NVIDIA Canary-Qwen-2.5B.

Usage:
    videototext-gpu.py <input> [options]

Input can be a local file path, a URL, or a YouTube video ID.
Requires an NVIDIA GPU.
"""

import warnings
import os

# Silence NeMo and dependency warnings before any imports
warnings.filterwarnings("ignore")
os.environ["NEMO_LOG_LEVEL"] = "ERROR"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
import logging

logging.disable(logging.CRITICAL)

import click
import sys
import re
import hashlib
import tempfile
import shutil
from pathlib import Path
from rich.console import Console
import yt_dlp
import ffmpeg

console = Console(stderr=True)


def get_data_dir() -> Path:
    """Get XDG-compliant data directory for videototext."""
    xdg = os.environ.get("XDG_DATA_HOME", os.path.expanduser("~/.local/share"))
    data_dir = Path(xdg) / "videototext"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_cache_dir() -> Path:
    """Get XDG-compliant cache directory for videototext."""
    xdg = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    cache_dir = Path(xdg) / "videototext"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def source_key(input_source: str, input_type: str) -> str:
    """Compute a cache key for the input source."""
    if input_type == "url":
        return hashlib.sha256(input_source.encode()).hexdigest()[:16]
    else:
        p = Path(input_source).resolve()
        st = p.stat()
        raw = f"{p}:{st.st_mtime_ns}:{st.st_size}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


def transcript_key(src_key: str, backend: str, model: str) -> str:
    """Compute a cache key for a transcript (source + backend + model)."""
    raw = f"{src_key}:{backend}:{model}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def detect_input_type(input_str: str) -> str:
    """Detect whether input is a file path, URL, or YouTube ID."""
    if os.path.exists(input_str):
        return "file"
    if re.match(r"^https?://", input_str):
        return "url"
    # YouTube video IDs are 11 characters: alphanumeric, dash, underscore
    if re.match(r"^[A-Za-z0-9_-]{11}$", input_str):
        return "youtube_id"
    console.print(
        f"[red]✗ File not found and not recognized as URL or YouTube ID:[/red] {input_str}"
    )
    sys.exit(1)


def format_duration(seconds: float) -> str:
    """Format seconds into H:MM:SS or M:SS."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def print_media_info(input_path: str, input_type: str):
    """Print basic media information if available."""
    try:
        if input_type == "url":
            with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
                info = ydl.extract_info(input_path, download=False)
            parts = []
            if info.get("title"):
                console.print(f"[dim]Title:[/dim] {info['title']}")
            if info.get("duration"):
                parts.append(f"duration {format_duration(info['duration'])}")
            if info.get("ext"):
                parts.append(f"format {info['ext']}")
            if info.get("resolution") and info["resolution"] != "audio only":
                parts.append(info["resolution"])
            if parts:
                console.print(f"[dim]{', '.join(parts)}[/dim]")
        else:
            probe = ffmpeg.probe(input_path)
            parts = []
            fmt = probe.get("format", {})
            if fmt.get("duration"):
                parts.append(f"duration {format_duration(float(fmt['duration']))}")
            if fmt.get("format_long_name"):
                parts.append(fmt["format_long_name"])
            for stream in probe.get("streams", []):
                if stream.get("codec_type") == "video":
                    w, h = stream.get("width"), stream.get("height")
                    if w and h:
                        parts.append(f"{w}x{h}")
                    if stream.get("nb_frames") and stream["nb_frames"] != "N/A":
                        parts.append(f"{stream['nb_frames']} frames")
                    if stream.get("r_frame_rate"):
                        num, den = stream["r_frame_rate"].split("/")
                        if int(den) > 0:
                            parts.append(f"{int(num)/int(den):.1f} fps")
                    break
            if parts:
                console.print(f"[dim]{', '.join(parts)}[/dim]")
    except Exception:
        pass


def normalize_audio(input_path: str, output_path: str) -> str:
    """Convert audio to 16kHz mono WAV."""
    try:
        stream = ffmpeg.input(input_path)
        stream = ffmpeg.output(stream, output_path, ar="16000", ac=1)
        ffmpeg.run(stream, quiet=True, overwrite_output=True)
        return output_path
    except ffmpeg.Error as e:
        stderr = e.stderr.decode() if e.stderr else "Unknown error"
        raise RuntimeError(f"FFmpeg error: {stderr}")


def extract_audio(input_path: str, output_path: str) -> str:
    """Extract audio from video file or URL as 16kHz mono WAV."""
    console.print(f"[dim]Extracting audio...[/dim]")

    if input_path.startswith(("http://", "https://")):
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
                ydl.download([input_path])

            # Find the downloaded file
            base = raw_path[:-4].split("/")[-1]
            downloaded = raw_path
            for f in os.listdir(os.path.dirname(raw_path) or "."):
                if f.startswith(base) and f.endswith(".wav"):
                    downloaded = os.path.join(os.path.dirname(raw_path) or ".", f)
                    break

            return normalize_audio(downloaded, output_path)
        finally:
            if os.path.exists(raw_path):
                os.remove(raw_path)
    else:
        return normalize_audio(input_path, output_path)


CHUNK_SECONDS = 30


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds via ffprobe."""
    probe = ffmpeg.probe(audio_path)
    return float(probe["format"]["duration"])


def split_audio(audio_path: str, chunk_dir: str, chunk_sec: int = CHUNK_SECONDS) -> list[str]:
    """Split audio into fixed-length chunks, return list of chunk paths."""
    duration = get_audio_duration(audio_path)
    chunks = []
    for start in range(0, int(duration) + 1, chunk_sec):
        chunk_path = os.path.join(chunk_dir, f"chunk_{start:06d}.wav")
        (
            ffmpeg.input(audio_path, ss=start, t=chunk_sec)
            .output(chunk_path, ar="16000", ac=1)
            .run(quiet=True, overwrite_output=True)
        )
        chunks.append(chunk_path)
    return chunks


def transcribe(audio_path: str) -> str:
    """Transcribe audio using NVIDIA Canary-Qwen-2.5B, chunking long files."""
    from nemo.collections.speechlm2.models import SALM

    console.print("[dim]Loading Canary-Qwen-2.5B model...[/dim]")
    model = SALM.from_pretrained("nvidia/canary-qwen-2.5b")
    model = model.cuda().eval()

    duration = get_audio_duration(audio_path)
    if duration <= CHUNK_SECONDS:
        console.print("[dim]Transcribing...[/dim]")
        answer_ids = model.generate(
            prompts=[
                [
                    {
                        "role": "user",
                        "content": f"Transcribe the following: {model.audio_locator_tag}",
                        "audio": [audio_path],
                    }
                ]
            ],
            max_new_tokens=1024,
        )
        return model.tokenizer.ids_to_text(answer_ids[0].cpu())

    # Long audio: split into chunks
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

    chunk_dir = tempfile.mkdtemp(prefix="vtt_chunks_")
    try:
        chunks = split_audio(audio_path, chunk_dir)
        parts = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[dim]{task.description}[/dim]"),
            BarColumn(),
            TextColumn("[dim]{task.completed}/{task.total}[/dim]"),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Transcribing", total=len(chunks))
            for chunk_path in chunks:
                answer_ids = model.generate(
                    prompts=[
                        [
                            {
                                "role": "user",
                                "content": f"Transcribe the following: {model.audio_locator_tag}",
                                "audio": [chunk_path],
                            }
                        ]
                    ],
                    max_new_tokens=1024,
                )
                text = model.tokenizer.ids_to_text(answer_ids[0].cpu())
                if text.strip():
                    parts.append(text.strip())
                progress.advance(task)
        return " ".join(parts)
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)


@click.command()
@click.argument("input_source", required=False)
@click.option("--output", "-o", type=click.Path(), help="Output text file")
@click.option("--no-cache", is_flag=True, help="Bypass cache")
@click.option("--clear-cache", is_flag=True, help="Remove all cached audio and transcripts")
@click.option("--check", is_flag=True, help="Check for required external tools")
@click.version_option(version="0.2.0")
def cli(input_source, output, no_cache, clear_cache, check):
    """Convert video/audio to text using NVIDIA Canary-Qwen-2.5B.

    INPUT_SOURCE can be a local file path, a URL, or a YouTube video ID.
    Requires an NVIDIA GPU.
    """
    if clear_cache:
        cache_dir = get_cache_dir()
        shutil.rmtree(cache_dir, ignore_errors=True)
        console.print("[green]Cache cleared.[/green]")
        return

    if check:
        tools = {
            "ffmpeg": shutil.which("ffmpeg"),
            "lame": shutil.which("lame"),
            "nvidia-smi": shutil.which("nvidia-smi"),
        }
        console.print("\n[bold]External Tools[/bold]")
        for tool, path in tools.items():
            status = "[green]✓[/green]" if path else "[red]✗[/red]"
            location = path if path else "[dim]not found[/dim]"
            console.print(f"  {status} {tool:12} {location}")
        missing = [k for k, v in tools.items() if not v]
        if missing:
            console.print(f"\n[yellow]Missing:[/yellow] {', '.join(missing)}")
            sys.exit(1)
        else:
            console.print("\n[green]All tools installed.[/green]")
        return

    if not input_source:
        console.print("[red]✗ Please provide an input file, URL, or YouTube ID.[/red]")
        raise SystemExit(1)

    input_type = detect_input_type(input_source)

    if input_type == "youtube_id":
        input_source = f"https://www.youtube.com/watch?v={input_source}"
        input_type = "url"

    label = "file" if input_type == "file" else "URL"
    console.print(f"[blue]Processing {label}:[/blue] {input_source}")
    print_media_info(input_source, input_type)

    cache_dir = get_cache_dir()
    src_key = source_key(input_source, input_type)
    t_key = transcript_key(src_key, "canary", "canary-qwen-2.5b")

    # Check transcript cache
    transcript_cache = cache_dir / "transcripts" / f"{t_key}.txt"
    if not no_cache and transcript_cache.exists():
        text = transcript_cache.read_text()
        console.print("[dim]Using cached transcript[/dim]")
        if output:
            Path(output).write_text(text)
            console.print(f"[green]✓ Saved to:[/green] {output}")
        else:
            print(text)
        return

    # Check audio cache (URLs only)
    audio_cache = cache_dir / "audio" / f"{src_key}.wav"
    if not no_cache and input_type == "url" and audio_cache.exists():
        console.print("[dim]Using cached audio[/dim]")
        audio_path = str(audio_cache)
        tmp_audio = None
    else:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_audio = tmp.name
        audio_path = extract_audio(input_source, tmp_audio)
        # Cache the downloaded audio for URLs
        if input_type == "url":
            audio_cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audio_path, audio_cache)

    try:
        text = transcribe(audio_path)

        # Cache the transcript
        transcript_cache.parent.mkdir(parents=True, exist_ok=True)
        transcript_cache.write_text(text)

        if output:
            Path(output).write_text(text)
            console.print(f"[green]✓ Saved to:[/green] {output}")
        else:
            print(text)
    finally:
        if tmp_audio and os.path.exists(tmp_audio):
            os.remove(tmp_audio)


if __name__ == "__main__":
    cli()
