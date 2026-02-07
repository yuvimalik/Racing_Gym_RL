"""
Fix racecar_gym dataclass bug for Python 3.11+ compatibility.

This script patches the racecar_gym package to fix the mutable default value error.
"""

import os
import sys
import shutil
from pathlib import Path

def find_racecar_gym_path():
    """Find where racecar_gym is installed."""
    try:
        import racecar_gym
        return Path(racecar_gym.__file__).parent
    except ImportError:
        # Try to find it in site-packages
        import site
        for site_packages in site.getsitepackages():
            racecar_path = Path(site_packages) / 'racecar_gym'
            if racecar_path.exists():
                return racecar_path
        return None

def backup_file(file_path):
    """Create a backup of the file."""
    backup_path = file_path.with_suffix(file_path.suffix + '.bak')
    shutil.copy2(file_path, backup_path)
    print(f"✓ Backed up to: {backup_path}")
    return backup_path

def fix_specs_file(specs_path):
    """Fix the dataclass mutable default issue."""
    print(f"\nReading file: {specs_path}")
    
    with open(specs_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already fixed
    if 'default_factory' in content and 'field(default_factory' in content:
        print("✓ File already appears to be patched!")
        return True
    
    # Find the problematic dataclass
    # The error mentions line 33, so we need to find the dataclass with a VehicleSpec default
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Look for dataclass decorator
        if '@dataclass' in line or 'class ' in line and 'Spec' in line:
            # Check next few lines for the problematic field
            lookahead = min(10, len(lines) - i)
            for j in range(lookahead):
                check_line = lines[i + j]
                # Look for field with VehicleSpec() or similar mutable default
                if 'vehicle' in check_line.lower() and ':' in check_line:
                    # Check if it has a default value that's a class instance
                    if 'VehicleSpec()' in check_line or '= VehicleSpec(' in check_line:
                        # Replace with default_factory
                        original = check_line
                        # Extract the field name and type
                        if '=' in check_line:
                            parts = check_line.split('=')
                            field_def = parts[0].strip()
                            default_val = parts[1].strip()
                            
                            # Replace with default_factory
                            indent = len(check_line) - len(check_line.lstrip())
                            new_line = ' ' * indent + field_def + ': VehicleSpec = field(default_factory=VehicleSpec)'
                            fixed_lines.append(new_line)
                            print(f"  Fixed line {i+j+1}:")
                            print(f"    Old: {original.strip()}")
                            print(f"    New: {new_line.strip()}")
                            i += j + 1
                            break
                    elif 'vehicle:' in check_line.lower() and 'VehicleSpec' in check_line:
                        # Might be a type annotation without default
                        fixed_lines.append(line)
                        i += 1
                        break
        
        fixed_lines.append(line)
        i += 1
    
    # Write the fixed content
    backup_path = backup_file(specs_path)
    new_content = '\n'.join(fixed_lines)
    
    with open(specs_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"✓ Fixed file written to: {specs_path}")
    return True

def main():
    """Main patching function."""
    print("="*70)
    print("RACECAR_GYM PATCH SCRIPT")
    print("="*70)
    print("\nThis script fixes the dataclass mutable default bug in racecar_gym")
    print("that causes errors in Python 3.11+.\n")
    
    # Find racecar_gym installation
    racecar_path = find_racecar_gym_path()
    
    if not racecar_path:
        print("✗ ERROR: Could not find racecar_gym installation!")
        print("Please install racecar_gym first:")
        print("  pip install git+https://github.com/axelbr/racecar_gym.git --no-deps")
        sys.exit(1)
    
    print(f"✓ Found racecar_gym at: {racecar_path}")
    
    # Find specs.py
    specs_path = racecar_path / 'core' / 'specs.py'
    
    if not specs_path.exists():
        print(f"✗ ERROR: Could not find specs.py at {specs_path}")
        sys.exit(1)
    
    print(f"✓ Found specs.py at: {specs_path}")
    
    # Fix the file
    try:
        fix_specs_file(specs_path)
        print("\n" + "="*70)
        print("PATCH COMPLETE")
        print("="*70)
        print("\n✓ racecar_gym has been patched successfully!")
        print("\nYou can now try running your training script:")
        print("  python train.py --config config/circle_config.yaml")
        print("\nIf you encounter issues, the original file was backed up.")
    except Exception as e:
        print(f"\n✗ ERROR: Failed to patch file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
