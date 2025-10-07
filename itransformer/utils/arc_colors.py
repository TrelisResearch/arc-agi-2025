"""
ARC standard color palette for visualizations
"""
from matplotlib.colors import ListedColormap

# Standard ARC color mapping
ARC_COLORS = [
    '#000000',  # 0: Black (background)
    '#0074D9',  # 1: Blue
    '#FF4136',  # 2: Red
    '#2ECC40',  # 3: Green
    '#FFDC00',  # 4: Yellow
    '#AAAAAA',  # 5: Gray
    '#F012BE',  # 6: Magenta
    '#FF851B',  # 7: Orange
    '#7FDBFF',  # 8: Light Blue
    '#870C25',  # 9: Maroon/Brown
]

def get_arc_cmap():
    """Get the standard ARC colormap for visualizations"""
    return ListedColormap(ARC_COLORS[:10])

# Create a global instance for convenience
arc_cmap = get_arc_cmap()