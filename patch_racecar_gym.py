"""
Patch script to fix racecar_gym dataclass compatibility issue with Python 3.11+.

This fixes the ValueError: mutable default for field vehicle is not allowed.
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
        # Try common installation paths
        possible_paths = [
            Path(sys.prefix) / "Lib" / "site-packages" / "racecar_gym",
            Path(sys.executable).parent / "Lib" / "site-packages" / "racecar_gym",
        ]
        
        # Check Windows Store Python location
        if 'PythonSoftwareFoundation' in str(sys.executable):
            user_site = Path.home() / "AppData" / "Local" / "Packages" / "PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0" / "LocalCache" / "local-packages" / "Python311" / "site-packages" / "racecar_gym"
            if user_site.exists():
                return user_site
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None

def patch_specs_file(specs_path):
    """Patch the specs.py file to fix dataclass issue."""
    print(f"Reading file: {specs_path}")
    
    with open(specs_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already patched
    if 'default_factory' in content and 'from dataclasses import field' in content:
        print("✓ File appears to be already patched!")
        return True
    
    # Create backup
    backup_path = specs_path.with_suffix('.py.backup')
    shutil.copy2(specs_path, backup_path)
    print(f"✓ Backup created: {backup_path}")
    
    # Fix the import - add field to imports
    if 'from dataclasses import dataclass' in content:
        content = content.replace(
            'from dataclasses import dataclass',
            'from dataclasses import dataclass, field'
        )
        print("✓ Added 'field' to dataclass imports")
    elif 'import dataclass' in content:
        # Handle different import styles
        content = content.replace(
            'from dataclasses import dataclass',
            'from dataclasses import dataclass, field'
        )
    
    # Find and fix the problematic dataclass field
    # Look for patterns like: vehicle: VehicleSpec = VehicleSpec(...)
    # Need to change to: vehicle: VehicleSpec = field(default_factory=VehicleSpec)
    
    lines = content.split('\n')
    modified = False
    new_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Look for dataclass field definitions with VehicleSpec as default
        if 'vehicle:' in line and 'VehicleSpec' in line and '=' in line:
            # Check if it's using a mutable default
            if 'VehicleSpec(' in line or 'VehicleSpec()' in line:
                # Extract the field name and type
                # Pattern: vehicle: VehicleSpec = VehicleSpec(...)
                indent = len(line) - len(line.lstrip())
                field_name = line.split(':')[0].strip()
                
                # Replace with default_factory
                new_line = ' ' * indent + f'{field_name}: VehicleSpec = field(default_factory=VehicleSpec)'
                new_lines.append(new_line)
                print(f"✓ Fixed line {i+1}: {line.strip()} -> {new_line.strip()}")
                modified = True
                i += 1
                continue
        
        new_lines.append(line)
        i += 1
    
    if modified:
        # Write patched content
        content = '\n'.join(new_lines)
        with open(specs_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ File patched successfully!")
        return True
    else:
        # Try a more aggressive search and replace
        print("⚠️  Pattern matching failed, trying regex replacement...")
        
        import re
        # Pattern: field_name: VehicleSpec = VehicleSpec(...)
        pattern = r'(\s+)(\w+):\s*VehicleSpec\s*=\s*VehicleSpec\([^)]*\)'
        
        def replace_func(match):
            indent = match.group(1)
            field_name = match.group(2)
            return f'{indent}{field_name}: VehicleSpec = field(default_factory=VehicleSpec)'
        
        new_content = re.sub(pattern, replace_func, content)
        
        if new_content != content:
            with open(specs_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print("✓ File patched using regex!")
            return True
        else:
            print("✗ Could not find pattern to patch. Manual fix may be required.")
            print("\nPlease manually edit the file and change:")
            print("  vehicle: VehicleSpec = VehicleSpec(...)")
            print("to:")
            print("  vehicle: VehicleSpec = field(default_factory=VehicleSpec)")
            return False

def main():
    """Main patching function."""
    print("="*70)
    print("RACECAR_GYM PATCH SCRIPT")
    print("="*70)
    print("\nThis script fixes the dataclass compatibility issue with Python 3.11+")
    print("Error: ValueError: mutable default for field vehicle is not allowed")
    print("="*70 + "\n")
    
    # Find racecar_gym installation
    racecar_gym_path = find_racecar_gym_path()
    
    if racecar_gym_path is None:
        print("✗ ERROR: Could not find racecar_gym installation!")
        print("\nPlease install racecar_gym first:")
        print("  pip install git+https://github.com/axelbr/racecar_gym.git --no-deps")
        return False
    
    print(f"✓ Found racecar_gym at: {racecar_gym_path}")
    
    # Find specs.py
    specs_path = racecar_gym_path / "core" / "specs.py"
    
    if not specs_path.exists():
        print(f"✗ ERROR: Could not find specs.py at: {specs_path}")
        return False
    
    print(f"✓ Found specs.py at: {specs_path}")
    
    # Patch the file
    success = patch_specs_file(specs_path)
    
    if success:
        print("\n" + "="*70)
        print("PATCH COMPLETE")
        print("="*70)
        print("✓ racecar_gym has been patched successfully!")
        print("\nYou can now import racecar_gym:")
        print("  python -c 'import racecar_gym.envs.gym_api; print(\"Success!\")'")
        return True
    else:
        print("\n" + "="*70)
        print("PATCH FAILED")
        print("="*70)
        print("✗ Automatic patching failed. Manual intervention required.")
        return False

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
