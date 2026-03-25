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
videototext.py - Convert video/audio to text using OpenAI Whisper (CPU).

Usage:
    videototext.py <input> [options]

Input can be a local file path, a URL, or a YouTube video ID.
For GPU-accelerated transcription, see videototext-gpu.py.
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


def transcribe_with_whisper(audio_path: str, model_name: str = "base") -> str:
    """Transcribe audio using OpenAI Whisper."""
    data_dir = get_data_dir() / "whisper"
    data_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Loading Whisper model:[/dim] {model_name}")
    model = whisper.load_model(model_name, download_root=str(data_dir))
    console.print("[dim]Transcribing...[/dim]")
    result = model.transcribe(audio_path, fp16=False)
    return result["text"]



@click.command()
@click.argument("input_source", required=False)
@click.option(
    "--model",
    default="base",
    help="Whisper model size (tiny, base, small, medium, large)",
)
@click.option("--output", "-o", type=click.Path(), help="Output text file")
@click.option("--no-cache", is_flag=True, help="Bypass cache")
@click.option("--clear-cache", is_flag=True, help="Remove all cached audio and transcripts")
@click.option("--check", is_flag=True, help="Check for required external tools")
@click.version_option(version="0.2.0")
def cli(input_source, model, output, no_cache, clear_cache, check):
    """Convert video/audio to text using OpenAI Whisper (CPU).

    INPUT_SOURCE can be a local file path, a URL, or a YouTube video ID.
    """
    if clear_cache:
        cache_dir = get_cache_dir()
        shutil.rmtree(cache_dir, ignore_errors=True)
        console.print("[green]Cache cleared.[/green]")
        return

    if check:
        tools = {"ffmpeg": shutil.which("ffmpeg"), "lame": shutil.which("lame")}
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
    t_key = transcript_key(src_key, "whisper", model)

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
        text = transcribe_with_whisper(audio_path, model)

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
