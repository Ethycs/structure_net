#!/usr/bin/env python3
"""
Simple validator for Mermaid diagram syntax.
Checks basic syntax patterns and common issues.
"""

import re
import os
from pathlib import Path

def validate_mermaid_syntax(content):
    """
    Basic validation of mermaid diagram syntax.
    Returns (is_valid, issues) tuple.
    """
    issues = []
    lines = content.split('\n')
    
    # Check for basic mermaid structure
    mermaid_blocks = []
    in_mermaid_block = False
    current_block = []
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        
        if line.startswith('```mermaid'):
            if in_mermaid_block:
                issues.append(f"Line {i}: Nested mermaid block detected")
            in_mermaid_block = True
            current_block = []
            continue
            
        if line == '```' and in_mermaid_block:
            mermaid_blocks.append('\n'.join(current_block))
            in_mermaid_block = False
            current_block = []
            continue
            
        if in_mermaid_block:
            current_block.append(line)
    
    if in_mermaid_block:
        issues.append("Unclosed mermaid block detected")
    
    # Validate each mermaid block
    for block_idx, block in enumerate(mermaid_blocks):
        block_issues = validate_mermaid_block(block, block_idx + 1)
        issues.extend(block_issues)
    
    return len(issues) == 0, issues

def validate_mermaid_block(block, block_num):
    """Validate individual mermaid block."""
    issues = []
    lines = [line.strip() for line in block.split('\n') if line.strip()]
    
    if not lines:
        issues.append(f"Block {block_num}: Empty mermaid block")
        return issues
    
    # Check for valid diagram type
    first_line = lines[0]
    valid_types = [
        'graph', 'flowchart', 'sequenceDiagram', 'classDiagram', 
        'stateDiagram', 'erDiagram', 'journey', 'gantt', 'pie',
        'gitgraph', 'mindmap', 'timeline', 'quadrantChart'
    ]
    
    has_valid_type = any(first_line.startswith(t) for t in valid_types)
    if not has_valid_type:
        issues.append(f"Block {block_num}: Unknown diagram type: {first_line}")
    
    # Check for common syntax issues
    for line_idx, line in enumerate(lines, 1):
        # Check for unmatched brackets
        if line.count('[') != line.count(']'):
            issues.append(f"Block {block_num}, Line {line_idx}: Unmatched square brackets")
        
        if line.count('(') != line.count(')'):
            issues.append(f"Block {block_num}, Line {line_idx}: Unmatched parentheses")
        
        if line.count('{') != line.count('}'):
            issues.append(f"Block {block_num}, Line {line_idx}: Unmatched curly braces")
        
        # Check for proper arrow syntax
        arrow_patterns = ['->', '-->', '-.>', '==>', '-.-.>', '~~>', '<->', '<-->', '<-.>', '<==>', '<-.-.>', '<~~>']
        if any(arrow in line for arrow in ['--', '->', '<-']) and not any(pattern in line for pattern in arrow_patterns):
            # This might be a malformed arrow, but it's not always an error
            pass
    
    return issues

def check_files():
    """Check all mermaid-related files in the current directory."""
    files_to_check = [
        'neuroflow_demo_mermaid.md',
        'neuroflow_demo_compact.md',
        'README_neuroflow_mermaid.md'
    ]
    
    all_valid = True
    
    for filename in files_to_check:
        if os.path.exists(filename):
            print(f"\n🔍 Checking {filename}...")
            
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            
            is_valid, issues = validate_mermaid_syntax(content)
            
            if is_valid:
                print(f"✅ {filename}: Valid mermaid syntax")
            else:
                print(f"❌ {filename}: Issues found:")
                for issue in issues:
                    print(f"   - {issue}")
                all_valid = False
        else:
            print(f"⚠️  {filename}: File not found")
    
    # Check HTML file for mermaid content
    html_file = 'neuroflow_mermaid_viewer.html'
    if os.path.exists(html_file):
        print(f"\n🔍 Checking {html_file}...")
        
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract mermaid diagrams from HTML
        mermaid_pattern = r'<div class="mermaid">(.*?)</div>'
        matches = re.findall(mermaid_pattern, content, re.DOTALL)
        
        if matches:
            print(f"   Found {len(matches)} mermaid diagrams in HTML")
            html_valid = True
            
            for i, match in enumerate(matches, 1):
                # Clean up the HTML content
                cleaned = re.sub(r'^\s+', '', match, flags=re.MULTILINE)
                cleaned = cleaned.strip()
                
                block_issues = validate_mermaid_block(cleaned, i)
                if block_issues:
                    print(f"   ❌ Diagram {i} issues:")
                    for issue in block_issues:
                        print(f"      - {issue}")
                    html_valid = False
            
            if html_valid:
                print(f"✅ {html_file}: All mermaid diagrams valid")
            else:
                all_valid = False
        else:
            print(f"⚠️  {html_file}: No mermaid diagrams found")
    
    print("\n" + "="*50)
    if all_valid:
        print("🎉 All mermaid diagrams passed validation!")
    else:
        print("❌ Some issues found. Please review and fix.")
    
    return all_valid

if __name__ == "__main__":
    print("🧪 Mermaid Diagram Validator")
    print("="*50)
    
    success = check_files()
    
    print("\n📋 Validation Summary:")
    print("- Checked markdown files for mermaid syntax")
    print("- Verified diagram types and basic structure") 
    print("- Looked for common syntax errors")
    print("- Validated HTML embedded diagrams")
    
    if success:
        print("\n✅ Ready to use! All diagrams should render correctly.")
    else:
        print("\n⚠️  Please fix issues before using the diagrams.")