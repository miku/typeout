#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "click",
#     "rich",
#     "yt-dlp",
#     "ffmpeg-python",
#     "openai-whisper",
#     "nemo_toolkit[asr]",
#     "transformers>=4.52,!=5.0.*,!=5.1.*",
#     "soundfile",
#     "librosa",
#     "sentencepiece",
#     "protobuf",
# ]
# ///

"""
typeout-gpu.py - GPU backend for typeout transcription.

Transcribe audio/video to text using NVIDIA NeMo ASR or Cohere Transcribe (GPU).

Input can be a local audio/video file (any format ffmpeg supports),
a URL, or a YouTube video ID. Requires an NVIDIA GPU.
"""

import warnings
import os

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

APP_NAME = "typeout"
CHUNK_SECONDS = 30

MODELS = {
    "canary-1b-v2": {
        "pretrained": "nvidia/canary-1b-v2",
        "api": "asr",
        "multilingual": True,
        "description": "1B multilingual (25 languages), translation support",
    },
    "canary-qwen-2.5b": {
        "pretrained": "nvidia/canary-qwen-2.5b",
        "api": "salm",
        "multilingual": True,
        "description": "2.5B speech-language model, highest quality",
    },
    "parakeet-0.6b": {
        "pretrained": "nvidia/parakeet-tdt-0.6b-v2",
        "api": "asr",
        "multilingual": False,
        "description": "600M English-only, fast and lightweight",
    },
    "cohere-transcribe": {
        "pretrained": "CohereLabs/cohere-transcribe-03-2026",
        "api": "cohere",
        "multilingual": True,
        "description": "2B multilingual (14 languages), high accuracy",
    },
    "distil-large-v3": {
        "pretrained": "distil-whisper/distil-large-v3",
        "api": "distil-whisper",
        "multilingual": True,
        "description": "Distil-Whisper large-v3 (~750MB, 6x faster than Whisper large)",
    },
    "distil-medium.en": {
        "pretrained": "distil-whisper/distil-medium.en",
        "api": "distil-whisper",
        "multilingual": False,
        "description": "Distil-Whisper medium English-only (~400MB, fast)",
    },
    "tiny": {
        "pretrained": "tiny",
        "api": "whisper",
        "multilingual": True,
        "description": "Whisper tiny, fastest, lowest accuracy",
    },
    "base": {
        "pretrained": "base",
        "api": "whisper",
        "multilingual": True,
        "description": "Whisper base, good balance of speed and accuracy",
    },
    "small": {
        "pretrained": "small",
        "api": "whisper",
        "multilingual": True,
        "description": "Whisper small, moderate accuracy",
    },
    "medium": {
        "pretrained": "medium",
        "api": "whisper",
        "multilingual": True,
        "description": "Whisper medium, high accuracy",
    },
    "large": {
        "pretrained": "large",
        "api": "whisper",
        "multilingual": True,
        "description": "Whisper large, highest accuracy",
    },
}

DEFAULT_MODEL = "canary-1b-v2"


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


def transcript_key(src_key: str, model_name: str, lang: str, target_lang: str) -> str:
    raw = f"{src_key}:{model_name}:{lang}:{target_lang}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Input detection
# ---------------------------------------------------------------------------

def detect_input_type(input_str: str) -> str:
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
    console.print("[dim]Preparing audio...[/dim]")
    if input_type == "url":
        return download_url(input_source, output_path)
    return normalize_audio(input_source, output_path)


# ---------------------------------------------------------------------------
# Audio chunking
# ---------------------------------------------------------------------------

def get_audio_duration(audio_path: str) -> float:
    probe = ffmpeg.probe(audio_path)
    return float(probe["format"]["duration"])


def split_audio(audio_path: str, chunk_dir: str) -> list[str]:
    duration = get_audio_duration(audio_path)
    chunks = []
    for start in range(0, int(duration) + 1, CHUNK_SECONDS):
        chunk_path = os.path.join(chunk_dir, f"chunk_{start:06d}.wav")
        (
            ffmpeg.input(audio_path, ss=start, t=CHUNK_SECONDS)
            .output(chunk_path, ar="16000", ac=1)
            .run(quiet=True, overwrite_output=True)
        )
        chunks.append(chunk_path)
    return chunks


# ---------------------------------------------------------------------------
# Transcription backends
# ---------------------------------------------------------------------------

def _transcribe_salm_chunk(model, audio_path: str) -> str:
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


def _transcribe_asr_chunk(model, audio_path: str, lang: str, target_lang: str, multilingual: bool) -> str:
    kwargs = {}
    if multilingual:
        kwargs["source_lang"] = lang
        kwargs["target_lang"] = target_lang
    output = model.transcribe([audio_path], batch_size=1, verbose=False, **kwargs)
    result = output[0]
    return result.text if hasattr(result, "text") else str(result)


def _transcribe_cohere(audio_path: str, model_cfg: dict, lang: str) -> str:
    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    device = "cuda:0"
    processor = AutoProcessor.from_pretrained(model_cfg["pretrained"], trust_remote_code=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_cfg["pretrained"], trust_remote_code=True
    ).to(device)
    model.eval()

    console.print("[dim]Transcribing...[/dim]")
    texts = model.transcribe(
        processor=processor,
        audio_files=[audio_path],
        language=lang,
    )
    return texts[0]


def _transcribe_whisper(audio_path: str, model_name: str, lang: str) -> str:
    import whisper

    data_dir = get_data_dir() / "whisper"
    data_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"[dim]Loading Whisper model:[/dim] {model_name}")
    model = whisper.load_model(model_name, download_root=str(data_dir))
    console.print("[dim]Transcribing...[/dim]")
    result = model.transcribe(audio_path, language=lang)
    return result["text"]


def _transcribe_distil_whisper(audio_path: str, model_cfg: dict, lang: str) -> str:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

    device = "cuda:0"
    console.print(f"[dim]Loading model:[/dim] {model_cfg['pretrained']}")
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_cfg["pretrained"], dtype=torch.float16,
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_cfg["pretrained"])

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch.float16,
        device=device,
    )

    generate_kwargs = {}
    if model_cfg.get("multilingual"):
        generate_kwargs["language"] = lang

    console.print("[dim]Transcribing...[/dim]")
    result = pipe(audio_path, chunk_length_s=30, return_timestamps=True,
                  generate_kwargs=generate_kwargs)
    return result["text"]


def transcribe(audio_path: str, model_name: str, lang: str, target_lang: str) -> str:
    """Load model, chunk if needed, transcribe, return text."""
    model_cfg = MODELS[model_name]

    console.print(f"[dim]Loading model:[/dim] {model_cfg['pretrained']}")

    if model_cfg["api"] == "whisper":
        return _transcribe_whisper(audio_path, model_cfg["pretrained"], lang)

    if model_cfg["api"] == "distil-whisper":
        return _transcribe_distil_whisper(audio_path, model_cfg, lang)

    if model_cfg["api"] == "cohere":
        return _transcribe_cohere(audio_path, model_cfg, lang)

    if model_cfg["api"] == "salm":
        from nemo.collections.speechlm2.models import SALM
        model = SALM.from_pretrained(model_cfg["pretrained"]).cuda().eval()
        do_chunk = lambda path: _transcribe_salm_chunk(model, path)
    else:
        from nemo.collections.asr.models import ASRModel
        model = ASRModel.from_pretrained(model_name=model_cfg["pretrained"]).cuda().eval()
        do_chunk = lambda path: _transcribe_asr_chunk(
            model, path, lang, target_lang, model_cfg["multilingual"]
        )

    duration = get_audio_duration(audio_path)
    if duration <= CHUNK_SECONDS:
        console.print("[dim]Transcribing...[/dim]")
        return do_chunk(audio_path).strip()

    # Long audio — chunk and show progress
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn

    chunk_dir = tempfile.mkdtemp(prefix="avt_chunks_")
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
                text = do_chunk(chunk_path)
                if text.strip():
                    parts.append(text.strip())
                progress.advance(task)
        return " ".join(parts)
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Model listing
# ---------------------------------------------------------------------------

def print_models():
    console.print("\n[bold]Available models:[/bold]\n")
    for name, cfg in MODELS.items():
        default = " [green](default)[/green]" if name == DEFAULT_MODEL else ""
        lang = "multilingual" if cfg["multilingual"] else "English only"
        console.print(f"  [bold]{name}[/bold]{default}")
        console.print(f"    {cfg['description']}")
        console.print(f"    [dim]{cfg['pretrained']} \u2022 {lang}[/dim]")
    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.argument("input_source", required=False)
@click.option("--model", "model_name", type=click.Choice(list(MODELS.keys())),
              default=DEFAULT_MODEL, help="ASR model")
@click.option("--lang", default="en", help="Source language (multilingual models)")
@click.option("--target-lang", default=None, help="Target language for translation (defaults to --lang)")
@click.option("--output", "-o", type=click.Path(), help="Write transcript to file")
@click.option("--no-cache", is_flag=True, help="Bypass cache")
@click.option("--clear-cache", is_flag=True, help="Remove all cached data")
@click.option("--list-models", is_flag=True, help="List available models")
@click.option("--check", is_flag=True, help="Check external tools")
@click.version_option(version=os.environ.get("TYPEOUT_VERSION", "dev"))
def cli(input_source, model_name, lang, target_lang, output, no_cache, clear_cache, list_models, check):
    """Transcribe audio or video to text using NVIDIA NeMo ASR (GPU).

    INPUT_SOURCE can be a local file (any format ffmpeg supports),
    a URL, or a YouTube video ID. Requires an NVIDIA GPU.
    """
    if list_models:
        print_models()
        return

    if clear_cache:
        shutil.rmtree(get_cache_dir(), ignore_errors=True)
        console.print("[green]Cache cleared.[/green]")
        return

    if check:
        tools = {"ffmpeg": shutil.which("ffmpeg"), "nvidia-smi": shutil.which("nvidia-smi")}
        for name, path in tools.items():
            s = "[green]ok[/green]" if path else "[red]missing[/red]"
            console.print(f"  {name}: {s}")
        if not all(tools.values()):
            sys.exit(1)
        return

    if not input_source:
        console.print("[red]Please provide an input file, URL, or YouTube ID.[/red]")
        raise SystemExit(1)

    if target_lang is None:
        target_lang = lang

    model_cfg = MODELS[model_name]
    if not model_cfg["multilingual"] and lang != "en":
        console.print(f"[yellow]{model_name} is English-only, ignoring --lang {lang}[/yellow]")
        lang = "en"
        target_lang = "en"
    if model_cfg["api"] in ("cohere", "whisper", "distil-whisper") and target_lang != lang:
        console.print(f"[yellow]{model_name} does not support translation, ignoring --target-lang {target_lang}[/yellow]")
        target_lang = lang

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
    t_key = transcript_key(src_key, model_name, lang, target_lang)
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

    # Audio cache (URLs only)
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
        text = transcribe(audio_path, model_name, lang, target_lang)

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
