#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN=""
if command -v python3.10 >/dev/null 2>&1; then
  PYTHON_BIN="python3.10"
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="python3.11"
elif command -v python3.12 >/dev/null 2>&1; then
  PYTHON_BIN="python3.12"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

recreate_venv="0"
if [[ -d .venv ]]; then
  venv_py=".venv/bin/python"
  if [[ -x "$venv_py" ]]; then
    vver="$("$venv_py" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)"
    # Torch/torchaudio and many audio deps are not available for Python 3.13+ yet.
    if [[ "$vver" == "3.13" || "$vver" == "3.14" ]]; then
      recreate_venv="1"
    fi
  else
    recreate_venv="1"
  fi
fi

if [[ ! -d .venv || "$recreate_venv" == "1" ]]; then
  rm -rf .venv
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools

# Core backend deps
python -m pip install -r requirements.txt

# Real-model deps (minimal)
# - faster-whisper: local ASR (downloads small model on first use)
# - openai-whisper + torch: alternative ASR backend (torch uses MPS on Apple Silicon)
python -m pip install "torch>=2.2" "torchaudio>=2.2" openai-whisper faster-whisper sentencepiece speechbrain scikit-learn pyannote.audio funasr modelscope

echo "OK. Activate env with: source backend/.venv/bin/activate"
echo "Run E2E smoke with: python backend/scripts/e2e_hub_real_models.py"
