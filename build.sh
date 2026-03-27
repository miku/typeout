#!/bin/bash
# build.sh - Generate the typeout amalgamation from CPU and GPU scripts

set -e

TYPEOUT="typeout"
TYPEOUT_CPU="typeout-cpu.py"
TYPEOUT_GPU="typeout-gpu.py"
VERSION="0.3.0"

echo "Building $TYPEOUT..."

cat > "$TYPEOUT" << 'HEADER'
#!/bin/bash
# typeout - Transcribe audio/video to text
# Amalgamation: contains both CPU and GPU versions, selects automatically
HEADER

echo "# Version $VERSION" >> "$TYPEOUT"
echo "# https://github.com/tir/code/miku/typeout" >> "$TYPEOUT"
echo "" >> "$TYPEOUT"

cat >> "$TYPEOUT" << 'BOILERPLATE'
set -o pipefail

# Check for uv
if ! command -v uv &>/dev/null; then
    echo "error: uv is required but not installed" >&2
    echo "install: cargo install uv  # or: pip install uv" >&2
    echo "  https://docs.astral.sh/uv/" >&2
    exit 1
fi

# Detect GPU
has_gpu() {
    [[ -d /proc/driver/nvidia ]] && command -v nvidia-smi &>/dev/null
}

# Get cache directory for extracted scripts
get_script_dir() {
    local base="${XDG_CACHE_HOME:-$HOME/.cache}/typeout"
    mkdir -p "$base"
    printf '%s' "$base"
}

# Extract and run the appropriate script
main() {
    local script_dir
    local script_file
    local backend

    script_dir=$(get_script_dir)

    if has_gpu; then
        backend="gpu"
        script_file="$script_dir/typeout-gpu.py"
    else
        backend="cpu"
        script_file="$script_dir/typeout-cpu.py"
    fi

    # Extract script if not present or version mismatch
    if [[ ! -f "$script_file" ]] || ! grep -q "Version 0.3.0" "$script_file" 2>/dev/null; then
        extract_"$backend"_script "$script_file"
    fi

    exec uv run "$script_file" "$@"
}

# Extract CPU version of the script
extract_cpu_script() {
    cat > "$1" << 'CPUSCRIPT'
BOILERPLATE

cat "$TYPEOUT_CPU" >> "$TYPEOUT"

cat >> "$TYPEOUT" << 'FOOTER1'

CPUSCRIPT
}

# Extract GPU version of the script
extract_gpu_script() {
    cat > "$1" << 'GPUSCRIPT'
FOOTER1

cat "$TYPEOUT_GPU" >> "$TYPEOUT"

cat >> "$TYPEOUT" << 'FOOTER2'

GPUSCRIPT
}

main "$@"
FOOTER2

chmod +x "$TYPEOUT"
echo "Done: $TYPEOUT"
