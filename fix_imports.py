#!/usr/bin/env python3
"""
Script to fix all imports after moving structure_net to src/
Changes imports from:
    from src.structure_net.xxx import yyy
    from src.structure_net.xxx import yyy
to:
    from src.structure_net.xxx import yyy
"""

import os
import re
from pathlib import Path

def fix_imports_in_file(filepath):
    """Fix imports in a single file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: from src.structure_net.xxx import yyy
    pattern1 = r'from structure_net\.'
    replacement1 = r'from src.structure_net.'
    content = re.sub(pattern1, replacement1, content)
    
    # Pattern 2: from src.structure_net.xxx import yyy (already correct, but might have src.src.)
    pattern2 = r'from src\.src\.structure_net\.'
    replacement2 = r'from src.structure_net.'
    content = re.sub(pattern2, replacement2, content)
    
    # Pattern 3: import src.structure_net.xxx
    pattern3 = r'import structure_net\.'
    replacement3 = r'import src.structure_net.'
    content = re.sub(pattern3, replacement3, content)
    
    # Pattern 4: Fix relative imports within structure_net
    # If file is in src/structure_net, use relative imports
    if 'src/structure_net/' in str(filepath):
        # Change from src.structure_net to relative imports
        # Count how many directories deep we are from src/structure_net
        rel_path = os.path.relpath(filepath, '/home/rabbit/structure_net/src/structure_net')
        depth = len(Path(rel_path).parents) - 1
        
        if depth > 0:
            # Replace imports within structure_net with relative imports
            # from src.structure_net.core.xxx import yyy -> from ..core.xxx import yyy
            pattern_internal = r'from src\.structure_net\.([a-zA-Z0-9_.]+) import'
            
            def replace_with_relative(match):
                module_path = match.group(1)
                parts = module_path.split('.')
                
                # Calculate relative path
                dots = '.' * (depth + 1)
                return f'from {dots}{module_path} import'
            
            content = re.sub(pattern_internal, replace_with_relative, content)
    
    if content != original_content:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix imports in all Python files."""
    root_dir = '/home/rabbit/structure_net'
    fixed_files = []
    
    # Fix files in all directories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip hidden directories and __pycache__
        dirnames[:] = [d for d in dirnames if not d.startswith('.') and d != '__pycache__']
        
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(dirpath, filename)
                if fix_imports_in_file(filepath):
                    fixed_files.append(filepath)
                    print(f"Fixed imports in: {filepath}")
    
    print(f"\nTotal files fixed: {len(fixed_files)}")
    
    # Special handling for neural_architecture_lab since it's in src/ not src/structure_net/
    nal_dir = os.path.join(root_dir, 'src/neural_architecture_lab')
    if os.path.exists(nal_dir):
        print("\nFixing Neural Architecture Lab imports...")
        for filename in os.listdir(nal_dir):
            if filename.endswith('.py'):
                filepath = os.path.join(nal_dir, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                
                # Fix relative imports from NAL to structure_net
                content = re.sub(r'from \.\.', r'from src.structure_net.', content)
                content = re.sub(r'from \.\.\.(.*) import', r'from src.structure_net.\1 import', content)
                
                with open(filepath, 'w') as f:
                    f.write(content)
                print(f"Fixed NAL imports in: {filepath}")

if __name__ == "__main__":
    main()