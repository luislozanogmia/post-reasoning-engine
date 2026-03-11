#!/bin/bash
# Build the Personal Synaptic Network (PSN)
# One command to set up everything.

set -e

echo "=== Post-Reasoning Engine: Personal Synaptic Network ==="
echo "Building your synthetic brain..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python 3.10+ required. Install from python.org"
    exit 1
fi

PYTHON=$(command -v python3 || command -v python)
echo "Using Python: $PYTHON"

# Create venv
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv .venv
fi

# Activate
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Download embedding model (first run only)
echo "Downloading embedding model (all-MiniLM-L6-v2, ~90MB)..."
python -c "
from transformers import AutoModel, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print('Downloading model...')
AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print('Done.')
"

# Create checkpoints directory
mkdir -p psn/checkpoints

# Verify
echo ""
echo "=== Build Complete ==="
echo ""
echo "Quick test:"
python -m psn store "This is my first thought in the network"
python -m psn recall "first thought"
echo ""
echo "Your PSN is ready. Start feeding it your thoughts:"
echo "  python -m psn store \"your thought here\""
echo "  python -m psn recall \"your cue here\""
echo "  python -m psn status"
echo ""
echo "To feed your conversations:"
echo "  python -m psn ingest --chatgpt path/to/export.zip"
echo "  python -m psn ingest --claude-json path/to/conversations.json"
