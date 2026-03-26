# avtotext

Transcribe audio or video to text using ASR (Automatic Speech Recognition).

Two self-contained scripts, pick the one that fits your hardware:

| Script | Backend | Hardware |
|---|---|---|
| `avtotext.py` | OpenAI Whisper, Cohere Transcribe | CPU |
| `avtotext-gpu.py` | NVIDIA NeMo ASR, Cohere Transcribe | NVIDIA GPU |

Input can be a local file (any format ffmpeg supports: mp3, wav, flac, mp4,
mkv, webm, ...), a URL, or a YouTube video ID.

## Features

- Accepts audio or video in any format ffmpeg can read
- Transcribe YouTube videos by URL or video ID
- Multiple models: Whisper (CPU/GPU), Cohere Transcribe 2B (CPU/GPU), NeMo ASR (GPU)
- Multilingual transcription (14 languages with Cohere, 25 with NeMo Canary)
- Speech translation (NeMo Canary models only)
- Caches downloaded audio and transcripts for instant repeat lookups
- Transcript on stdout, diagnostics on stderr (pipe-friendly)

## Installation

The only prerequisite is [uv](https://docs.astral.sh/uv/). Each script is
self-contained and manages all Python dependencies automatically on first run
via [uv script](https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies).

```bash
# Install uv
cargo install uv  # or: pip install uv, or: curl -LsSf https://astral.sh/uv/install.sh | sh

# Copy the script(s) somewhere in PATH and make executable
cp avtotext.py ~/.local/bin/
chmod +x ~/.local/bin/avtotext.py
```

That's it. No virtualenvs, no `pip install`, no setup.py.

## Usage

### CPU (Whisper, Cohere Transcribe)

```bash
# Transcribe a local audio or video file (default: Whisper base)
avtotext.py recording.mp3
avtotext.py lecture.mp4

# Use Cohere Transcribe (requires Hugging Face login)
avtotext.py recording.mp3 --model cohere-transcribe --lang en
avtotext.py lecture.mp4 --model cohere-transcribe --lang ja

# Pipe-friendly: transcript on stdout, progress on stderr
avtotext.py interview.wav > transcript.txt

# Write to file
avtotext.py podcast.flac -o transcript.txt

# Use a different Whisper model (tiny, base, small, medium, large)
avtotext.py recording.mp3 --model small

# Transcribe from a URL
avtotext.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# YouTube video ID (11 characters)
avtotext.py dQw4w9WgXcQ

# Check external tools
avtotext.py --check
```

**Cohere Transcribe setup** (gated model):
```bash
# 1. Accept terms at: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026
# 2. Login to Hugging Face
huggingface-cli login
```

### GPU (NVIDIA [NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/intro.html), Cohere Transcribe)

```bash
# List available models
avtotext-gpu.py --list-models

# Transcribe with default model (canary-1b-v2)
avtotext-gpu.py recording.mp3

# Cohere Transcribe (requires Hugging Face login)
avtotext-gpu.py recording.mp3 --model cohere-transcribe --lang en

# Choose a specific NeMo model
avtotext-gpu.py lecture.mp4 --model canary-qwen-2.5b
avtotext-gpu.py lecture.mp4 --model parakeet-0.6b

# Multilingual: set source language
avtotext-gpu.py interview.wav --lang de

# Translation: German audio to English text (NeMo Canary only)
avtotext-gpu.py interview.wav --lang de --target-lang en

# From a URL
avtotext-gpu.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

#### Available GPU models

| Model | Size | Languages | Notes |
|---|---|---|
| `canary-1b-v2` (default) | 1B | 25 languages | NeMo, multilingual, translation |
| `canary-qwen-2.5b` | 2.5B | multilingual | NeMo, highest quality, SLM |
| `parakeet-0.6b` | 600M | English only | NeMo, fast and lightweight |
| `cohere-transcribe` | 2B | 14 languages | Cohere, high accuracy |

## Caching

Downloaded audio (for URLs) and transcripts are cached in
`$XDG_CACHE_HOME/avtotext/` (defaults to `~/.cache/avtotext/`). Both scripts
share the same cache directory.

- **URLs**: keyed by URL: same URL hits cache instantly
- **Local files**: keyed by path + modification time + size — cache invalidates on edit
- **Transcripts**: keyed by source + model + language — different models/languages get separate entries

Use `--no-cache` to bypass, `--clear-cache` to remove all cached data.

## Dependencies

The only thing you need to install is `uv` and `ffmpeg`. Everything else is
managed automatically.

- [uv](https://docs.astral.sh/uv/) — runs the scripts and manages Python dependencies
- `ffmpeg` — audio extraction and normalization (check with `--check`)
- `nvidia-smi` — GPU script only
- `huggingface-cli login` — required for Cohere Transcribe (gated model)

Models are cached in `$XDG_DATA_HOME/avtotext/` (defaults to `~/.local/share/avtotext/`).
