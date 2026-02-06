"""
Quick setup verification script to check if all dependencies are installed correctly.
"""

import sys

def check_imports():
    """Check if all required packages are installed."""
    print("Checking dependencies...")
    print("-" * 50)
    
    required_packages = {
        'gymnasium': 'gymnasium',
        'stable_baselines3': 'stable_baselines3',
        'numpy': 'numpy',
        'pybullet': 'pybullet',
        'torch': 'torch',
        'yaml': 'yaml',
        'racecar_gym': 'racecar_gym.envs.gym_api'
    }
    
    missing = []
    for name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING")
            missing.append(name)
    
    print("-" * 50)
    
    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\nAll dependencies installed!")
        return True

def check_config():
    """Check if config file exists."""
    import os
    from pathlib import Path
    
    config_path = Path('config/circle_config.yaml')
    if config_path.exists():
        print(f"\n✓ Config file found: {config_path}")
        return True
    else:
        print(f"\n✗ Config file not found: {config_path}")
        return False

def check_directories():
    """Check if required directories exist or can be created."""
    from pathlib import Path
    
    dirs = ['models', 'logs', 'results']
    print("\nChecking directories...")
    print("-" * 50)
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ {dir_name}/ exists")
        else:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ {dir_name}/ created")
            except Exception as e:
                print(f"✗ {dir_name}/ - Cannot create: {e}")
                return False
    
    return True

if __name__ == '__main__':
    print("=" * 50)
    print("RACING GYM RL - SETUP VERIFICATION")
    print("=" * 50)
    
    all_ok = True
    all_ok &= check_imports()
    all_ok &= check_config()
    all_ok &= check_directories()
    
    print("\n" + "=" * 50)
    if all_ok:
        print("✓ Setup verification complete! Ready to train.")
        print("\nRun training with:")
        print("  python train.py")
    else:
        print("✗ Setup incomplete. Please fix the issues above.")
    print("=" * 50)
