# Comprehensive Installation Prompt for Warp AI

Copy and paste this entire prompt into Warp AI to get help with installation:

---

**I need help installing Python packages for a reinforcement learning project. I'm encountering multiple dependency conflicts and build errors. Please help me diagnose and fix the installation issues.**

## Current Situation:

1. **Python Environment:**
   - Python 3.11+ (numpy 1.22.x not available)
   - Windows 10/11
   - Using pip (may need upgrade)

2. **Package Requirements:**
   - `numpy` - Need version compatible with Python 3.11+
   - `gymnasium>=0.29.1` - RL environment library
   - `stable-baselines3[extra]>=2.4.0` - PPO algorithm implementation
   - `pybullet>=3.2.0` - Physics engine (requires C++ build tools on Windows)
   - `racecar_gym` - Custom package from GitHub (may pin numpy==1.22.3)
   - `torch>=2.0.0` - PyTorch
   - Other packages: matplotlib, opencv-python, tensorboard, pyyaml

3. **Known Issues:**
   - `numpy==1.22.3` not available for Python 3.11+ (only 1.23.2+ available)
   - `racecar_gym` from GitHub may require `numpy==1.22.3` exactly
   - `pybullet` requires Microsoft Visual C++ Build Tools to compile from source
   - `stable-baselines3 2.0.0` requires `gymnasium==0.28.1`, but we need `gymnasium>=0.29.1`

## What I Need:

1. **Diagnose my current Python environment:**
   - Check Python version
   - Check pip version
   - Check which packages are already installed
   - Check what numpy versions are available
   - Check if C++ build tools are installed

2. **Create an installation plan that:**
   - Resolves all dependency conflicts
   - Uses compatible package versions
   - Handles the numpy version issue (Python 3.11+)
   - Handles pybullet installation (prefer pre-built wheels, fallback to build tools)
   - Installs racecar_gym without strict dependency checking if needed
   - Provides fallback options if primary method fails

3. **Generate step-by-step commands:**
   - Commands should be Windows PowerShell compatible
   - Include error handling and verification steps
   - Provide alternatives if primary method fails

## Specific Questions:

1. What's the best way to handle the numpy version conflict between Python 3.11+ and racecar_gym's requirement?
2. How can I install pybullet without building from source (to avoid C++ build tools)?
3. What's the correct order to install packages to minimize conflicts?
4. Should I use `--no-deps` for racecar_gym, or is there a better approach?
5. Are there alternative package sources (conda-forge, etc.) that might help?

## Expected Output:

Please provide:
- Diagnostic commands to run first
- A step-by-step installation script/commands
- Verification steps after installation
- Troubleshooting guide for common errors

Run the diagnostic script first: `python install_diagnostic.py`

Then provide a tailored installation plan based on the results.

---
