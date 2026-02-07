"""
Automated installation script for Racing Gym RL project.
Installs multi_car_racing dependencies in correct order.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a command and handle errors."""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}")
    print("-"*70)
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n✗ FAILED: {description}")
        if check:
            print("Stopping installation. Please fix the error above.")
            sys.exit(1)
        else:
            print("Continuing despite error...")
    else:
        print(f"\n✓ SUCCESS: {description}")
    
    return result.returncode == 0

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    print(f"\n{'='*70}")
    print("ENVIRONMENT CHECK")
    print(f"{'='*70}")
    print(f"Python Version: {version.major}.{version.minor}.{version.micro}")
    print(f"Python Executable: {sys.executable}")
    
    if version.major != 3 or version.minor < 8:
        print("✗ ERROR: Python 3.8+ is required!")
        sys.exit(1)
    
    if version.minor >= 11:
        print("ℹ️  INFO: Python 3.11+ detected - will use numpy>=1.23.0")
        return ">=1.23.0,<2.0"
    else:
        print("ℹ️  INFO: Python <3.11 - will use numpy>=1.22.0,<1.23.0")
        return ">=1.22.0,<1.23.0"

def main():
    """Main installation process."""
    print("="*70)
    print("RACING GYM RL - AUTOMATED INSTALLATION")
    print("="*70)
    print("\nThis script will install all required packages in the correct order.")
    print("Make sure you have:")
    print("  - Python 3.8+ installed")
    print("  - Internet connection")
    print("\nPress Ctrl+C to cancel, or Enter to continue...")
    try:
        input()
    except KeyboardInterrupt:
        print("\nInstallation cancelled.")
        sys.exit(0)
    
    # Check Python version and determine numpy version
    numpy_version = check_python_version()
    
    # Step 1: Upgrade pip
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip to latest version"
    )
    
    # Step 2: Install numpy first (compatible version)
    run_command(
        f'pip install --prefer-binary "numpy{numpy_version}"',
        f"Installing numpy {numpy_version}"
    )
    
    # Step 3: Install core ML packages
    run_command(
        'pip install --prefer-binary "gym==0.17.3" "stable-baselines3[extra]==1.8.0"',
        "Installing gym and stable-baselines3"
    )
    
    # Step 4: Install visualization and utility packages
    run_command(
        'pip install --prefer-binary "matplotlib>=3.7.0" "opencv-python>=4.8.0" "tensorboard>=2.13.0" "pyyaml>=6.0" "pyglet==1.5.27"',
        "Installing visualization and utility packages"
    )
    
    # Step 5: Install PyTorch
    run_command(
        'pip install --prefer-binary "torch>=2.0.0"',
        "Installing PyTorch"
    )
    
    # Step 6: Install multi_car_racing without deps (we pin shapely/box2d explicitly)
    run_command(
        'pip install git+https://github.com/igilitschenski/multi_car_racing.git --no-deps',
        "Installing multi_car_racing (no-deps)"
    )
    
    # Step 8: Verify installation
    print(f"\n{'='*70}")
    print("VERIFICATION")
    print(f"{'='*70}")
    
    packages_to_check = {
        'numpy': 'numpy',
        'gym': 'gym',
        'gymnasium': 'gymnasium',
        'shimmy': 'shimmy',
        'stable_baselines3': 'stable_baselines3',
        'torch': 'torch',
        'gym_multi_car_racing': 'gym_multi_car_racing'
    }
    
    all_ok = True
    for display_name, import_name in packages_to_check.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'installed')
            print(f"✓ {display_name}: {version}")
        except ImportError as e:
            print(f"✗ {display_name}: NOT INSTALLED - {e}")
            all_ok = False
    
    # Final summary
    print(f"\n{'='*70}")
    print("INSTALLATION COMPLETE")
    print(f"{'='*70}")
    
    if all_ok:
        print("✓ All packages installed successfully!")
        print("\nNext steps:")
        print("  1. Run: python check_setup.py")
        print("  2. Run: python train.py --config config/multi_car_config.yaml")
    else:
        print("⚠️  Some packages failed to install. Please check errors above.")
        print("\nYou can try installing failed packages manually:")
        print("  pip install <package_name>")
    
    print("="*70)

if __name__ == '__main__':
    main()
