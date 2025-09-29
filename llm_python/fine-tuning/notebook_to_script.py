#!/usr/bin/env python
"""
Convert Jupyter notebook to Python script with YAML config support.

Usage:
    uv run python notebook_to_script.py notebook.ipynb [--config config.yaml] [--output script.py]
"""

import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_code_cells(notebook_path: str, skip_config_cell: bool = False) -> List[str]:
    """Extract code cells from notebook, optionally skipping config cell."""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    code_blocks = []
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Skip the config cell if requested (now defaults to False)
            if skip_config_cell and i == 1 and 'Config' in source and '=' in source:
                continue
            
            # Skip empty cells or cells with only comments
            if source.strip() and not all(line.strip().startswith('#') or not line.strip() 
                                         for line in source.split('\n')):
                code_blocks.append(source)
    
    return code_blocks




def convert_notebook_to_script(
    notebook_path: str,
    output_path: str = None
) -> str:
    """Convert notebook to executable Python script."""
    
    # Generate output filename if not provided
    if output_path is None:
        nb_path = Path(notebook_path)
        output_path = nb_path.with_suffix('.py')
    
    # Start building the script
    script_parts = [
        "#!/usr/bin/env python",
        '"""',
        f"Generated from: {notebook_path}",
        '"""',
        "",
    ]
    
    # Extract all code cells from notebook
    code_cells = extract_code_cells(notebook_path, skip_config_cell=False)
    
    for i, cell_code in enumerate(code_cells):
        script_parts.append("# ---------------------------------------------------------------------")
        script_parts.append(f"# Cell {i + 1}")
        script_parts.append("# ---------------------------------------------------------------------")
        script_parts.append(cell_code)
        script_parts.append("")
    
    # Write the script
    script_content = '\n'.join(script_parts)
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Converted notebook to: {output_path}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Convert Jupyter notebook to Python script')
    parser.add_argument('notebook', help='Path to the Jupyter notebook')
    parser.add_argument('--output', help='Output Python script path')
    
    args = parser.parse_args()
    
    # Check if notebook exists
    if not Path(args.notebook).exists():
        print(f"❌ Notebook {args.notebook} not found!")
        sys.exit(1)
    
    # Convert the notebook (uses notebook's own config cell)
    output_file = convert_notebook_to_script(
        notebook_path=args.notebook,
        output_path=args.output
    )
    
    print(f"\n🚀 Run with: uv run python {output_file} --config config.yaml")


if __name__ == "__main__":
    import sys
    main()