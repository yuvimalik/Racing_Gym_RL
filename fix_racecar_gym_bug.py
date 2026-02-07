"""
Fix the racecar_gym dataclass bug for Python 3.11+ compatibility.
This script patches the specs.py file to use default_factory instead of mutable defaults.
"""

import os
import sys
import site
from pathlib import Path
import re

def find_racecar_gym_path():
    """Find racecar_gym installation path without importing it."""
    # Get all site-packages directories
    site_packages = site.getsitepackages()
    
    # Also check user site-packages
    user_site = site.getusersitepackages()
    if user_site:
        site_packages.append(user_site)
    
    # Common Windows Store Python location
    python_path = Path(sys.executable).parent.parent
    local_packages = python_path / "LocalCache" / "local-packages" / f"Python{sys.version_info.major}{sys.version_info.minor}" / "site-packages"
    if local_packages.exists():
        site_packages.append(str(local_packages))
    
    # Search for racecar_gym
    for sp_dir in site_packages:
        sp_path = Path(sp_dir)
        if not sp_path.exists():
            continue
        
        racecar_path = sp_path / "racecar_gym"
        if racecar_path.exists() and (racecar_path / "core" / "specs.py").exists():
            return racecar_path / "core" / "specs.py"
    
    # Try to find it by searching common locations
    common_paths = [
        Path(sys.executable).parent / "Lib" / "site-packages" / "racecar_gym" / "core" / "specs.py",
        Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "racecar_gym" / "core" / "specs.py",
    ]
    
    for path in common_paths:
        if path.exists():
            return path
    
    return None

def fix_specs_file(file_path):
    """Fix the dataclass mutable default issue in specs.py."""
    print(f"Reading file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern to find dataclass fields with VehicleSpec as default
    # Look for patterns like: vehicle: VehicleSpec = VehicleSpec(...)
    # or: vehicle: VehicleSpec = some_vehicle_instance
    
    # First, let's find the problematic dataclass
    # The error says line 33, so let's look around there
    
    lines = content.split('\n')
    fixed_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        # Look for dataclass decorator
        if '@dataclass' in line or 'class' in line and 'Spec' in line:
            # Check if this is a dataclass with mutable default
            # Look ahead a few lines for field definitions
            class_start = i
            class_lines = []
            indent_level = None
            
            # Find the class definition
            if 'class' in line:
                indent_level = len(line) - len(line.lstrip())
                class_lines.append(line)
                i += 1
                
                # Collect class body
                while i < len(lines):
                    current_line = lines[i]
                    if current_line.strip() == '':
                        class_lines.append(current_line)
                        i += 1
                        continue
                    
                    current_indent = len(current_line) - len(current_line.lstrip())
                    
                    # Check if we've left the class
                    if current_indent <= indent_level and current_line.strip() != '':
                        break
                    
                    class_lines.append(current_line)
                    i += 1
                
                # Check if this class has the problematic pattern
                class_content = '\n'.join(class_lines)
                
                # Look for field definitions with VehicleSpec as default
                # Pattern: field_name: VehicleSpec = VehicleSpec(...)
                pattern = r'(\s+)(\w+):\s*VehicleSpec\s*=\s*VehicleSpec\('
                match = re.search(pattern, class_content)
                
                if match:
                    print(f"Found problematic field at line {class_start + class_content[:match.start()].count(chr(10)) + 1}")
                    
                    # Fix it by using default_factory
                    # Replace: field: VehicleSpec = VehicleSpec(...)
                    # With: field: VehicleSpec = field(default_factory=lambda: VehicleSpec(...))
                    
                    # More careful replacement
                    field_pattern = r'(\s+)(\w+):\s*VehicleSpec\s*=\s*VehicleSpec\(([^)]*)\)'
                    
                    def replace_field(m):
                        indent = m.group(1)
                        field_name = m.group(2)
                        args = m.group(3)
                        return f'{indent}{field_name}: VehicleSpec = field(default_factory=lambda: VehicleSpec({args}))'
                    
                    fixed_class = re.sub(field_pattern, replace_field, class_content)
                    
                    # Check if we need to import field
                    if 'from dataclasses import' in fixed_class:
                        # Add field to imports
                        fixed_class = re.sub(
                            r'from dataclasses import ([^\\n]+)',
                            lambda m: f"from dataclasses import {m.group(1)}, field" if 'field' not in m.group(1) else m.group(0),
                            fixed_class
                        )
                    elif 'import dataclasses' in fixed_class:
                        # Use dataclasses.field
                        fixed_class = re.sub(
                            r'field\(default_factory=lambda: VehicleSpec',
                            r'dataclasses.field(default_factory=lambda: VehicleSpec',
                            fixed_class
                        )
                    else:
                        # Add import
                        fixed_class = re.sub(
                            r'from dataclasses import dataclass',
                            r'from dataclasses import dataclass, field',
                            fixed_class
                        )
                    
                    fixed_lines.extend(fixed_class.split('\n'))
                    continue
                else:
                    # No fix needed, add lines as-is
                    fixed_lines.extend(class_lines)
                    continue
        
        fixed_lines.append(line)
        i += 1
    
    fixed_content = '\n'.join(fixed_lines)
    
    # If content changed, write it back
    if fixed_content != original_content:
        # Create backup
        backup_path = file_path.with_suffix('.py.backup')
        print(f"Creating backup: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(original_content)
        
        # Write fixed content
        print(f"Writing fixed file: {file_path}")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(fixed_content)
        
        return True
    else:
        print("No changes needed or pattern not found.")
        return False

def main():
    """Main function."""
    print("="*70)
    print("RACECAR_GYM BUG FIX FOR PYTHON 3.11+")
    print("="*70)
    print(f"Python version: {sys.version}")
    print()
    
    # Find specs.py file
    specs_file = find_racecar_gym_path()
    
    if not specs_file:
        print("ERROR: Could not find racecar_gym/core/specs.py")
        print("\nTried searching in:")
        for sp in site.getsitepackages():
            print(f"  - {sp}")
        print("\nPlease manually locate the file and run:")
        print("  python fix_racecar_gym_bug.py <path_to_specs.py>")
        sys.exit(1)
    
    print(f"Found specs.py at: {specs_file}")
    print()
    
    # Fix the file
    try:
        fixed = fix_specs_file(specs_file)
        if fixed:
            print("\n" + "="*70)
            print("SUCCESS: File has been fixed!")
            print("="*70)
            print("A backup has been created at:")
            print(f"  {specs_file.with_suffix('.py.backup')}")
            print("\nYou can now try running your training script:")
            print("  python train.py")
        else:
            print("\n" + "="*70)
            print("INFO: File may already be fixed or pattern not found.")
            print("="*70)
            print("Try running your training script to see if it works:")
            print("  python train.py")
    except Exception as e:
        print(f"\nERROR: Failed to fix file: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    # Allow manual path specification
    if len(sys.argv) > 1:
        manual_path = Path(sys.argv[1])
        if manual_path.exists():
            fix_specs_file(manual_path)
        else:
            print(f"ERROR: File not found: {manual_path}")
            sys.exit(1)
    else:
        main()
