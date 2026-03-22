#!/usr/bin/env bash
# Local environment setup script for Racing Gym RL.
#
# WHY NOT "pip install -r requirements.txt":
# stable-baselines3 >= 1.5 declares gym>=0.21 in its PyPI metadata, but
# multi_car_racing requires gym==0.17.3.  pip's modern resolver refuses to
# install both simultaneously.  The packages DO work together at runtime —
# the workaround is to install gym and SB3 with --no-deps, then add each
# package's real transitive dependencies explicitly.
#
# Usage (from repo root, with venv activated):
#   chmod +x setup_local.sh && ./setup_local.sh

set -e  # abort on first error

# ── System dependency: swig ───────────────────────────────────────────────────
# box2d-py 2.3.8 has no pre-built wheel for macOS arm64 and must be compiled
# from source.  The build requires 'swig' (a C++ wrapper generator).
echo "==> Step 0: checking for swig (required to build box2d-py from source)"
if ! command -v swig &>/dev/null; then
    if command -v brew &>/dev/null; then
        echo "    swig not found — installing via Homebrew..."
        brew install swig
    else
        echo "ERROR: 'swig' is not installed and Homebrew is not available."
        echo "       Please install swig manually:"
        echo "         macOS:  brew install swig"
        echo "         Ubuntu: sudo apt-get install swig"
        exit 1
    fi
else
    echo "    swig found: $(swig -version 2>&1 | head -1)"
fi

echo "==> Step 1: core numeric / utility deps"
pip install "numpy>=1.22.0,<1.23.0"
pip install "scipy"
pip install "cloudpickle>=1.2.0,<1.7.0"

echo "==> Step 2: rendering deps"
pip install "pyglet==1.5.0"
pip install "future"        # required by pyglet 1.5.0

echo "==> Step 3: physics deps"
pip install "shapely==1.8.5.post1"
pip install "box2d-py==2.3.8"

echo "==> Step 4: gym==0.17.3 (--no-deps avoids pyglet/SB3 conflict)"
pip install "gym==0.17.3" --no-deps

echo "==> Step 5: stable-baselines3==1.5.0 (--no-deps avoids gym conflict)"
pip install "stable-baselines3[extra]==1.5.0" --no-deps
# pandas and seaborn are SB3 [extra] transitive deps skipped by --no-deps
pip install "pandas" "seaborn"

echo "==> Step 6: ML / viz / config deps"
pip install "torch>=2.0.0"
pip install "matplotlib>=3.7.0"
pip install "opencv-python>=4.8.0"
pip install "tensorboard>=2.13.0"
pip install "pyyaml>=6.0"

echo "==> Step 7: multi_car_racing (--no-deps avoids shapely/box2d build conflicts)"
pip install git+https://github.com/igilitschenski/multi_car_racing.git --no-deps

echo ""
echo "Done. Verify with:  python3 -c \"import train\""
