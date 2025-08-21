#!/usr/bin/env python3
"""
Shared configuration for RunPod scripts
"""

# Default data centers for RunPod instances
# Ordered by preference: US regions first, then EU
DEFAULT_DATA_CENTERS = [
    "US-IL-1",  # US-Illinois-1
    "US-TX-1",  # US-Texas-1  
    "US-TX-3",  # US-Texas-3
    "US-KS-2",  # US-Kansas-2
    "EU-RO-1",  # EU-Romania-1
    "EU-SE-1",   # EU-Sweden-1
    "US-NC-1",
    # "AP-JP-1"
]

# Alternative region sets for specific use cases
REGIONS = {
    "default": DEFAULT_DATA_CENTERS,
    "us_only": ["US-IL-1", "US-TX-1", "US-TX-3", "US-KS-2"],
    "eu_only": ["EU-RO-1", "EU-SE-1"]
}

def get_regions(region_set="default"):
    """Get a list of regions by name
    
    Args:
        region_set: One of "default", "us_only", "eu_only", "low_latency", "cost_optimized"
        
    Returns:
        List of region IDs
    """
    return REGIONS.get(region_set, DEFAULT_DATA_CENTERS)