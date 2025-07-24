#!/usr/bin/env python3

from pathlib import Path
from typing import Dict

class PromptLoader:
    """Load versioned prompt strings from the prompt-strings directory"""
    
    def __init__(self, base_dir: str = "prompt-strings"):
        self.base_dir = Path(base_dir)
        self._prompt_cache = {}
    
    def load_prompt(self, prompt_type: str, version: str = "v1") -> str:
        """Load a prompt from file, with caching"""
        cache_key = f"{prompt_type}_{version}"
        
        if cache_key not in self._prompt_cache:
            prompt_file = self.base_dir / prompt_type / f"{version}.txt"
            
            if not prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
            
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_content = f.read().strip()
            
            self._prompt_cache[cache_key] = prompt_content
        
        return self._prompt_cache[cache_key]
    
    def get_system_message(self, version: str = "v1") -> str:
        """Get the system message prompt"""
        return self.load_prompt("system", version)
    
    def get_initial_turn_prompt(self, version: str = "v1") -> str:
        """Get the initial turn prompt template"""
        return self.load_prompt("initial-turn", version)
    
    def get_subsequent_turn_prompt(self, version: str = "v1") -> str:
        """Get the subsequent turn prompt"""
        return self.load_prompt("subsequent-turn", version)
    
    def get_code_request_prompt(self, version: str = "v1") -> str:
        """Get the code request prompt"""
        return self.load_prompt("code-request", version)
    
    def list_available_versions(self, prompt_type: str) -> list:
        """List all available versions for a prompt type"""
        prompt_dir = self.base_dir / prompt_type
        if not prompt_dir.exists():
            return []
        
        versions = []
        for file_path in prompt_dir.glob("*.txt"):
            versions.append(file_path.stem)
        
        return sorted(versions)
    
    def clear_cache(self):
        """Clear the prompt cache (useful for testing or reloading)"""
        self._prompt_cache.clear() 