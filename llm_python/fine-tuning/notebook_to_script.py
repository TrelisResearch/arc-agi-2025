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


def extract_code_cells(notebook_path: str, skip_config_cell: bool = True) -> List[str]:
    """Extract code cells from notebook, optionally skipping config cell."""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    code_blocks = []
    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Skip the config cell if requested
            if skip_config_cell and i == 1 and 'Config' in source and '=' in source:
                continue
            
            # Skip empty cells or cells with only comments
            if source.strip() and not all(line.strip().startswith('#') or not line.strip() 
                                         for line in source.split('\n')):
                code_blocks.append(source)
    
    return code_blocks


def generate_config_loader() -> str:
    """Generate Python code to load configuration from YAML."""
    return '''# ---------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------
import yaml
import sys
from pathlib import Path

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file."""
    if not Path(config_path).exists():
        print(f"Config file {config_path} not found!")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Flatten nested configuration
    flat_config = {}
    
    # Handle test_run override
    test_run = config.get('test_run', False)
    flat_config['test_run'] = test_run
    
    # Environment configs
    flat_config['is_local'] = config.get('is_local', False)
    flat_config['local_model_dir'] = config.get('local_model_dir', '/kaggle/working/')
    
    # Model configs
    model_config = config.get('model', {})
    flat_config['model_slug'] = model_config.get('slug', 'Qwen/Qwen3-4B')
    flat_config['model_max_length'] = model_config.get('max_length', 32768)
    flat_config['lora_rank'] = model_config.get('lora_rank', 128)
    
    # Training configs
    training_config = config.get('training', {})
    flat_config['batch_size_global'] = training_config.get('batch_size_global', 4)
    flat_config['train_slug'] = training_config.get('dataset_slug', 'Trelis/arc-agi-2-perfect-50')
    flat_config['enable_thinking'] = training_config.get('enable_thinking', False)
    
    # Handle max_rows with test_run override
    if test_run:
        overrides = config.get('overrides', {})
        flat_config['max_rows'] = overrides.get('test_run_max_rows', 128)
    else:
        flat_config['max_rows'] = training_config.get('max_rows')
    
    return flat_config

# Load configuration
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config.yaml', help='Path to config file')
args, unknown = parser.parse_known_args()

config = load_config(args.config)

# Inject config variables into global namespace
for key, value in config.items():
    globals()[key] = value
    print(f"Config: {key} = {value}")

print("\\nConfiguration loaded successfully!")
print("-" * 70)
'''


def convert_notebook_to_script(
    notebook_path: str,
    config_path: str = None,
    output_path: str = None,
    include_config_loader: bool = True
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
        f"Config file: {config_path or 'config.yaml'}",
        '"""',
        "",
    ]
    
    # Add config loader if requested
    if include_config_loader and config_path:
        script_parts.append(generate_config_loader())
        script_parts.append("")
    
    # Add code cells
    code_cells = extract_code_cells(notebook_path, skip_config_cell=include_config_loader)
    
    for i, cell_code in enumerate(code_cells):
        script_parts.append(f"# ---------------------------------------------------------------------")
        script_parts.append(f"# Cell {i + 1}")
        script_parts.append(f"# ---------------------------------------------------------------------")
        script_parts.append(cell_code)
        script_parts.append("")
    
    # Write the script
    script_content = '\n'.join(script_parts)
    
    with open(output_path, 'w') as f:
        f.write(script_content)
    
    print(f"✅ Converted notebook to: {output_path}")
    if config_path:
        print(f"📝 Using config file: {config_path}")
    
    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Convert Jupyter notebook to Python script with YAML config')
    parser.add_argument('notebook', help='Path to the Jupyter notebook')
    parser.add_argument('--config', default='config.yaml', help='Path to YAML config file')
    parser.add_argument('--output', help='Output Python script path')
    parser.add_argument('--no-config-loader', action='store_true', 
                       help='Do not include config loader code')
    
    args = parser.parse_args()
    
    # Check if notebook exists
    if not Path(args.notebook).exists():
        print(f"❌ Notebook {args.notebook} not found!")
        sys.exit(1)
    
    # Convert the notebook
    output_file = convert_notebook_to_script(
        notebook_path=args.notebook,
        config_path=args.config if not args.no_config_loader else None,
        output_path=args.output,
        include_config_loader=not args.no_config_loader
    )
    
    print(f"\n🚀 Run with: uv run python {output_file} --config {args.config}")


if __name__ == "__main__":
    import sys
    main()