"""
Diagnostic script to check Python environment and suggest installation fixes.
Run this to get detailed information about your setup.
"""

import sys
import subprocess
import platform

def run_command(cmd):
    """Run a command and return output."""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), 1

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"\n{'='*70}")
    print("PYTHON VERSION CHECK")
    print(f"{'='*70}")
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    print(f"Python Executable: {sys.executable}")
    
    if version.major != 3:
        print("⚠️  ERROR: Python 3 is required!")
        return False
    if version.minor < 8:
        print("⚠️  WARNING: Python 3.8+ recommended")
        return False
    if version.minor >= 11:
        print("ℹ️  INFO: Python 3.11+ detected - numpy 1.22.x not available")
    print("✓ Python version OK")
    return True

def check_pip_version():
    """Check pip version."""
    print(f"\n{'='*70}")
    print("PIP VERSION CHECK")
    print(f"{'='*70}")
    stdout, stderr, code = run_command(f"{sys.executable} -m pip --version")
    if code == 0:
        print(f"✓ {stdout}")
        version_str = stdout.split()[1]
        major, minor = map(int, version_str.split('.')[:2])
        if major < 24:
            print(f"⚠️  WARNING: pip version {version_str} is outdated")
            print("   Consider upgrading: python -m pip install --upgrade pip")
        return True
    else:
        print(f"✗ Error checking pip: {stderr}")
        return False

def check_installed_packages():
    """Check which packages are already installed."""
    print(f"\n{'='*70}")
    print("INSTALLED PACKAGES CHECK")
    print(f"{'='*70}")
    
    packages_to_check = [
        'numpy', 'gymnasium', 'stable_baselines3', 
        'pybullet', 'torch', 'matplotlib', 'opencv-python', 
        'tensorboard', 'yaml', 'racecar_gym'
    ]
    
    installed = {}
    for pkg in packages_to_check:
        import_name = pkg.replace('-', '_')
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            installed[pkg] = version
            print(f"✓ {pkg}: {version}")
        except ImportError:
            print(f"✗ {pkg}: NOT INSTALLED")
            installed[pkg] = None
    
    return installed

def check_numpy_availability():
    """Check what numpy versions are available."""
    print(f"\n{'='*70}")
    print("NUMPY AVAILABILITY CHECK")
    print(f"{'='*70}")
    
    stdout, stderr, code = run_command(
        f"{sys.executable} -m pip index versions numpy"
    )
    
    if code == 0:
        print("Available numpy versions:")
        print(stdout[:500])  # First 500 chars
    else:
        # Try alternative method
        stdout, stderr, code = run_command(
            f"{sys.executable} -m pip install numpy==999.999.999 2>&1"
        )
        if "from versions:" in stderr:
            versions_line = [line for line in stderr.split('\n') if 'from versions:' in line]
            if versions_line:
                print("Available numpy versions:")
                print(versions_line[0])

def check_build_tools():
    """Check if C++ build tools are available."""
    print(f"\n{'='*70}")
    print("BUILD TOOLS CHECK")
    print(f"{'='*70}")
    
    if platform.system() == 'Windows':
        # Check for Visual Studio Build Tools
        import os
        vs_paths = [
            r"C:\Program Files (x86)\Microsoft Visual Studio",
            r"C:\Program Files\Microsoft Visual Studio",
        ]
        
        found = False
        for vs_path in vs_paths:
            if os.path.exists(vs_path):
                print(f"✓ Found Visual Studio at: {vs_path}")
                found = True
        
        if not found:
            print("✗ Visual C++ Build Tools not found")
            print("  Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
            print("  Select 'C++ build tools' workload during installation")
        else:
            print("✓ Build tools appear to be installed")
    else:
        print("ℹ️  Build tools check skipped (not Windows)")

def generate_installation_plan(installed):
    """Generate a recommended installation plan."""
    print(f"\n{'='*70}")
    print("RECOMMENDED INSTALLATION PLAN")
    print(f"{'='*70}")
    
    python_version = sys.version_info
    
    plan = []
    
    # Step 1: Upgrade pip
    plan.append({
        'step': 1,
        'command': f'{sys.executable} -m pip install --upgrade pip',
        'description': 'Upgrade pip to latest version'
    })
    
    # Step 2: Install numpy (compatible version)
    if python_version.minor >= 11:
        numpy_version = "numpy>=1.23.0,<2.0"
        plan.append({
            'step': 2,
            'command': f'pip install --prefer-binary "{numpy_version}"',
            'description': f'Install numpy {numpy_version} (Python 3.11+ compatible)'
        })
    else:
        numpy_version = "numpy>=1.22.0,<1.23.0"
        plan.append({
            'step': 2,
            'command': f'pip install --prefer-binary "{numpy_version}"',
            'description': f'Install numpy {numpy_version}'
        })
    
    # Step 3: Install core packages
    plan.append({
        'step': 3,
        'command': 'pip install --prefer-binary "gymnasium>=0.29.1" "stable-baselines3[extra]>=2.4.0" "matplotlib>=3.7.0" "opencv-python>=4.8.0" "tensorboard>=2.13.0" "pyyaml>=6.0" "torch>=2.0.0"',
        'description': 'Install core ML packages'
    })
    
    # Step 4: Install pybullet
    plan.append({
        'step': 4,
        'command': 'pip install --prefer-binary pybullet',
        'description': 'Install pybullet (may require C++ build tools if no wheel available)',
        'fallback': 'If this fails, install Visual C++ Build Tools or use: conda install -c conda-forge pybullet'
    })
    
    # Step 5: Install racecar-gym
    plan.append({
        'step': 5,
        'command': 'pip install git+https://github.com/axelbr/racecar_gym.git --no-deps',
        'description': 'Install racecar-gym without dependency checking (to avoid numpy conflict)'
    })
    
    # Print plan
    for item in plan:
        print(f"\nStep {item['step']}: {item['description']}")
        print(f"  Command: {item['command']}")
        if 'fallback' in item:
            print(f"  Fallback: {item['fallback']}")
    
    return plan

def main():
    """Run all diagnostic checks."""
    print("="*70)
    print("RACING GYM RL - INSTALLATION DIAGNOSTIC")
    print("="*70)
    
    # Run checks
    python_ok = check_python_version()
    pip_ok = check_pip_version()
    installed = check_installed_packages()
    check_numpy_availability()
    check_build_tools()
    
    # Generate plan
    plan = generate_installation_plan(installed)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Python: {'✓' if python_ok else '✗'}")
    print(f"Pip: {'✓' if pip_ok else '✗'}")
    print(f"Packages installed: {sum(1 for v in installed.values() if v is not None)}/{len(installed)}")
    
    print(f"\n{'='*70}")
    print("NEXT STEPS")
    print(f"{'='*70}")
    print("Run the commands from the installation plan above, or use:")
    print("  python install_diagnostic.py > install_log.txt")
    print("to save this output to a file.")

if __name__ == '__main__':
    main()
