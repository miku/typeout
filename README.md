# typeout

Transcribe audio or video to text using ASR (Automatic Speech Recognition).

A single self-contained script that auto-detects GPU availability and uses the appropriate backend.

| Backend | Hardware | Models |
|---|---|---|
| OpenAI Whisper | CPU/GPU | tiny, base, small, medium, large |
| Distil-Whisper | CPU/GPU | distil-large-v3, distil-medium.en |
| Cohere Transcribe 2B | CPU/GPU | 14 languages |
| NVIDIA NeMo ASR | GPU | Canary-1B-v2, Canary-Qwen-2.5B, Parakeet-0.6B |

Input can be a local file (any format ffmpeg supports — mp3, wav, flac, mp4, mkv, webm, ...), a URL, or a YouTube video ID.

## Features

- **Single command** — auto-detects GPU and uses the right backend
- Accepts audio or video in any format ffmpeg can read
- Transcribe YouTube videos by URL or video ID
- Multiple models: Whisper, Cohere Transcribe, NeMo ASR
- Multilingual transcription (14 languages with Cohere, 25 with NeMo Canary)
- Speech translation (NeMo Canary models only)
- Caches downloaded audio and transcripts for instant repeat lookups
- Transcript on stdout, diagnostics on stderr (pipe-friendly)

## Installation

The only prerequisite is [uv](https://docs.astral.sh/uv/). The script is self-contained and manages all Python dependencies automatically on first run.

```bash
# Install uv
cargo install uv  # or: pip install uv

# Download typeout
curl -O https://raw.githubusercontent.com/tir/code/refs/heads/main/miku/avtotext/typeout
# or
wget https://raw.githubusercontent.com/tir/code/refs/heads/main/miku/avtotext/typeout

# Make executable and put in PATH
chmod +x typeout
mv typeout ~/.local/bin/
```

That's it. No virtualenvs, no `pip install`, no setup.py.

## Usage

```bash
# Transcribe a local file
typeout recording.mp3
typeout lecture.mp4

# Write to file
typeout podcast.flac -o transcript.txt

# Use different Whisper model (CPU)
typeout recording.mp3 --model small

# Use Cohere Transcribe (CPU/GPU, requires Hugging Face login)
typeout recording.mp3 --model cohere-transcribe --lang en
typeout lecture.mp4 --model cohere-transcribe --lang ja

# GPU models (auto-detected on systems with NVIDIA GPU)
typeout recording.mp3 --model canary-qwen-2.5b
typeout lecture.mp4 --model parakeet-0.6b

# Use Whisper on GPU (fp16 acceleration)
typeout recording.mp3 --model whisper-large

# Multilingual: set source language
typeout interview.wav --lang de

# Translation: German audio to English text (NeMo Canary only)
typeout interview.wav --lang de --target-lang en

# From a URL or YouTube ID
typeout "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
typeout dQw4w9WgXcQ

# Check external tools
typeout --check

# Clear cache
typeout --clear-cache

# List GPU models
typeout --list-models
```

## Models

### CPU (auto-selected)

| Model | Size | Languages | Notes |
|---|---|---|---|
| `base` (default) | ~140MB | multilingual | Whisper, good balance |
| `tiny` | ~40MB | multilingual | Whisper, fastest |
| `small` | ~460MB | multilingual | Whisper |
| `medium` | ~1.5GB | multilingual | Whisper |
| `large` | ~2.9GB | multilingual | Whisper, highest accuracy |
| `distil-large-v3` | ~750MB | multilingual | Distil-Whisper, 6x faster than large |
| `distil-medium.en` | ~400MB | English only | Distil-Whisper, fast |
| `cohere-transcribe` | ~4GB | 14 languages | High accuracy, requires HF login |

### GPU (auto-selected when NVIDIA GPU detected)

| Model | Size | Languages | Notes |
|---|---|---|---|
| `canary-1b-v2` (default) | 1B | 25 languages | NeMo, multilingual, translation |
| `canary-qwen-2.5b` | 2.5B | multilingual | NeMo, highest quality, SLM |
| `parakeet-0.6b` | 600M | English only | NeMo, fast and lightweight |
| `cohere-transcribe` | 2B | 14 languages | Cohere, high accuracy |
| `distil-large-v3` | ~750MB | multilingual | Distil-Whisper, 6x faster than large |
| `distil-medium.en` | ~400MB | English only | Distil-Whisper, fast |
| `whisper-tiny` | ~40MB | multilingual | Whisper, fastest |
| `whisper-base` | ~140MB | multilingual | Whisper, good balance |
| `whisper-small` | ~460MB | multilingual | Whisper |
| `whisper-medium` | ~1.5GB | multilingual | Whisper |
| `whisper-large` | ~2.9GB | multilingual | Whisper, highest accuracy |

**Cohere Transcribe setup** (gated model):
```bash
# 1. Accept terms at: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
# 2. Login to Hugging Face
huggingface-cli login
```

## How it works

The `typeout` script is an **amalgamation** — it contains both CPU and GPU Python scripts embedded within it. On first run:

1. Detects if NVIDIA GPU is available (`nvidia-smi`)
2. Extracts the appropriate Python script to `~/.cache/typeout/`
3. Runs it with `uv run`, which installs dependencies automatically

Subsequent runs reuse the extracted script (unless you upgrade typeout).

## Caching

Downloaded audio (for URLs) and transcripts are cached in `~/.cache/typeout/` (respects `$XDG_CACHE_HOME`).

- **URLs**: keyed by URL — same URL hits cache instantly
- **Local files**: keyed by path + modification time + size — cache invalidates on edit
- **Transcripts**: keyed by source + model + language — different models/languages get separate entries

Use `--no-cache` to bypass, `--clear-cache` to remove all cached data.

## Dependencies

- [uv](https://docs.astral.sh/uv/) — runs the script and manages Python dependencies
- `ffmpeg` — audio extraction and normalization (check with `--check`)
- `nvidia-smi` — GPU detection (auto-detected)
- `huggingface-cli login` — required for Cohere Transcribe (gated model)

Models are cached in `~/.local/share/typeout/` (respects `$XDG_DATA_HOME`).
