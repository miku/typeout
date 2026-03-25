#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["nemo_toolkit[asr]"]
# ///

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from nemo.collections.asr.models import ASRModel


def get_cache_dir():
    """Get the cache directory using XDG_CACHE_HOME"""
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return Path(cache_home) / "radioscript"


def get_transcription_file(input_file):
    """Get the corresponding transcription file path"""
    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{input_file.stem}.json"


def convert_to_wav(input_path, output_path):
    """
    Convert input audio to mono WAV using ffmpeg.
    Targeting 16kHz is recommended for most ASR models.
    """
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # Overwrite output file if exists
                "-i",
                str(input_path),
                "-ac",
                "1",  # Convert to Mono
                "-ar",
                "16000",  # Resample to 16000Hz (Standard for ASR)
                "-vn",  # Disable video recording
                str(output_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed for {input_path}") from e
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please ensure ffmpeg is installed and in your PATH."
        )


def transcribe_file(input_file):
    """Transcribe a single file"""
    # Check if transcription already exists
    trans_file = get_transcription_file(input_file)
    if trans_file.exists():
        print(f"Skipping {input_file.name} (already transcribed)", file=sys.stderr)
        return

    # Create temporary file for the JSON output (Atomic write pattern)
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as tmp_json:
        temp_json_path = Path(tmp_json.name)

        try:
            # Create a temporary directory for the intermediate WAV file
            with tempfile.TemporaryDirectory() as temp_audio_dir:
                temp_wav_path = Path(temp_audio_dir) / "temp_audio.wav"

                # Convert MP3 to WAV
                # print(f"Converting {input_file.name} to wav...")
                convert_to_wav(input_file, temp_wav_path)

                # Load model (lazy loading, done only once)
                if "asr_ast_model" not in globals():
                    global asr_ast_model
                    print("Loading Canary model...", file=sys.stderr)
                    asr_ast_model = ASRModel.from_pretrained(
                        model_name="nvidia/canary-1b-v2"
                    )

                # Transcribe using the temporary WAV file
                print(f"Transcribing {input_file.name}...", file=sys.stderr)
                output = asr_ast_model.transcribe(
                    [str(temp_wav_path)],
                    source_lang="de",
                    target_lang="de",
                    timestamps=True,
                )

                # Write to temporary JSON file
                if output and len(output) > 0:
                    tss = output[0].timestamp
                    json_data = {
                        "tss": tss,
                        "text": output[0].text,
                    }
                    json.dump(json_data, tmp_json)
                else:
                    json.dump({"tss": [], "text": ""}, tmp_json)

            # Atomically move JSON to final location
            temp_json_path.replace(trans_file)
            print(f"Finished {input_file.name}", file=sys.stderr)

        except Exception as e:
            # Clean up JSON temp file on error
            temp_json_path.unlink(missing_ok=True)
            raise e


def main():
    """Main function to process filtered MP3 files"""
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py <recording_directory>")
        sys.exit(1)

    recording_dir = Path(sys.argv[1])

    if not recording_dir.exists():
        print(f"Directory not found: {recording_dir}")
        sys.exit(1)

    # Regex pattern breakdown:
    # ^stream-      : Starts with "stream-"
    # [^-]+         : NAME (one or more characters that are not hyphens)
    # -             : Separator
    # [^-]+         : ID (one or more characters that are not hyphens)
    # -             : Separator
    # \d{8}         : DATE (exactly 8 digits)
    # -             : Separator
    # \d{6}         : TIME (exactly 6 digits)
    # \.mp3$        : Ends with ".mp3"
    filename_pattern = re.compile(r"^stream-[^.]+-[^-]+-\d{8}-\d{6}\.mp3$")

    # Find all MP3 files recursively
    all_mp3_candidates = recording_dir.rglob("*.mp3")

    # Filter files based on the regex pattern
    mp3_files = [f for f in all_mp3_candidates if filename_pattern.match(f.name)]

    if not mp3_files:
        print(
            "No MP3 files matching the pattern 'stream-NAME-ID-DATE-STARTTIME.mp3' found."
        )
        return

    print(f"Found {len(mp3_files)} matching MP3 files.")

    # Process each file
    for mp3_file in mp3_files:
        try:
            transcribe_file(mp3_file)
        except Exception as e:
            print(f"Error processing {mp3_file.name}: {e}")


if __name__ == "__main__":
    main()
