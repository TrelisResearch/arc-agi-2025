#!/usr/bin/env python3

import json
import numpy as np
from typing import Any, Dict, List


class ResponseSerializer:
    """Unified JSON serialization for ARC task responses and data structures"""
    
    @staticmethod
    def ensure_json_serializable(obj: Any) -> Any:
        """Convert any iterators or non-serializable objects to JSON-safe formats"""
        if obj is None:
            return None
        
        # Handle specific problematic types first
        if type(obj).__name__ == 'list_reverseiterator':
            if hasattr(obj, '__len__'):
                print(f"⚠️  Converting list_reverseiterator with {len(obj)} items to list")
            else:
                print(f"⚠️  Converting list_reverseiterator to list")
            return list(obj)
        elif type(obj).__name__ in ('map', 'filter', 'enumerate', 'zip'):
            print(f"⚠️  Converting {type(obj).__name__} iterator to list")
            return list(obj)
        elif hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        
        # Try JSON serialization test for other objects
        try:
            json.dumps(obj)
            return obj  # Already serializable
        except (TypeError, ValueError):
            # Not serializable, try to convert
            if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, dict)):
                try:
                    return list(obj)
                except (TypeError, ValueError):
                    pass
            # Final fallback to string representation
            return str(obj)
    
    @classmethod
    def serialize_response(cls, response) -> Dict:
        """Convert OpenAI response to JSON-serializable format"""
        if not response:
            return None
        
        try:
            choices = []
            response_choices = getattr(response, 'choices', [])
            # Ensure choices is a list, not an iterator
            response_choices = cls.ensure_json_serializable(response_choices)
            
            for choice in response_choices:
                message_data = {
                    'role': cls.ensure_json_serializable(getattr(choice.message, 'role', None)) if hasattr(choice, 'message') else None,
                    'content': cls.ensure_json_serializable(getattr(choice.message, 'content', None)) if hasattr(choice, 'message') else None,
                }
                
                # Capture reasoning content from different model types and standardize to "reasoning" field
                reasoning_content = None
                
                # Check for Qwen reasoning_content field first
                if hasattr(choice, 'message') and hasattr(choice.message, 'reasoning_content'):
                    reasoning_content = cls.ensure_json_serializable(getattr(choice.message, 'reasoning_content', None))
                
                # Check for Gemini reasoning field
                if hasattr(choice, 'message') and hasattr(choice.message, 'reasoning'):
                    reasoning_content = cls.ensure_json_serializable(getattr(choice.message, 'reasoning', None))
                
                if reasoning_content:
                    message_data['reasoning'] = reasoning_content
                
                # Check for o1 reasoning_details field
                if hasattr(choice, 'message') and hasattr(choice.message, 'reasoning_details'):
                    message_data['reasoning_details'] = cls.ensure_json_serializable(getattr(choice.message, 'reasoning_details', None))
                
                choice_data = {
                    'index': cls.ensure_json_serializable(getattr(choice, 'index', None)),
                    'message': message_data,
                    'finish_reason': cls.ensure_json_serializable(getattr(choice, 'finish_reason', None)),
                }
                choices.append(choice_data)
            
            return {
                'id': cls.ensure_json_serializable(getattr(response, 'id', None)),
                'model': cls.ensure_json_serializable(getattr(response, 'model', None)),
                'usage': {
                    'prompt_tokens': cls.ensure_json_serializable(getattr(response.usage, 'prompt_tokens', 0)) if hasattr(response, 'usage') and response.usage else 0,
                    'completion_tokens': cls.ensure_json_serializable(getattr(response.usage, 'completion_tokens', 0)) if hasattr(response, 'usage') and response.usage else 0,
                    'total_tokens': cls.ensure_json_serializable(getattr(response.usage, 'total_tokens', 0)) if hasattr(response, 'usage') and response.usage else 0,
                },
                'choices': choices,
            }
        except Exception as e:
            print(f"⚠️  Response serialization error: {e}")
            return None
    
    @classmethod
    def make_json_safe(cls, obj: Any) -> Any:
        """Recursively make objects JSON-safe by handling complex nested structures"""
        if obj is None:
            return None
        
        # Handle sets
        if isinstance(obj, set):
            return list(obj)
        
        # Handle dictionaries
        if isinstance(obj, dict):
            safe_dict = {}
            for k, v in obj.items():
                # Make keys safe (convert to string if needed)
                new_k = str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k
                safe_dict[new_k] = cls.make_json_safe(v)
            return safe_dict
        
        # Handle lists
        if isinstance(obj, list):
            return [cls.make_json_safe(x) for x in obj]
        
        if isinstance(obj, tuple):
            return [cls.make_json_safe(x) for x in obj]
        
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return [cls.make_json_safe(x) for x in obj]
        
        # Try basic serialization test
        try:
            json.dumps(obj)
            return obj
        except (TypeError, ValueError):
            # Handle iterators/generators
            if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                try:
                    return [cls.make_json_safe(x) for x in list(obj)]
                except (TypeError, ValueError):
                    pass
            
            # Handle objects with custom attributes
            if hasattr(obj, '__dict__'):
                try:
                    return [cls.make_json_safe(x) for x in list(obj)]
                except (TypeError, ValueError):
                    pass
            
            # Final fallback - convert to string
            return str(obj)
    
    @classmethod
    def safe_json_dumps(cls, data: Any, indent: int = 2) -> str:
        """Safely dump any data structure to JSON string"""
        try:
            safe_data = cls.make_json_safe(data)
            return json.dumps(safe_data, indent=indent)
        except Exception as e:
            print(f"⚠️  JSON serialization error: {e}")
            return json.dumps({"error": f"Failed to serialize: {str(e)}"}, indent=indent)