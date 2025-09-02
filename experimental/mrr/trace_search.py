import jax
import jax.numpy as jnp
# Import the original scipy for the 'label' function
import scipy.ndimage as ndi
from functools import partial
from typing import Tuple

# By decorating with @partial(jax.jit), JAX will compile these functions
# to highly optimized machine code. We use partial to allow for static arguments
# in some functions if needed later.

# --- Helper Function ---
def _scipy_label_wrapper(mask):
    """A pure Python wrapper for the scipy label function."""
    return ndi.label(mask)

@partial(jax.jit)
def get_objects(grid: jnp.ndarray) -> jnp.ndarray:
    """
    Identifies connected components (objects) in a grid.
    Returns a grid where each object is labeled with a unique integer.
    The background (color 0) is ignored.
    """
    # Create a binary mask where any non-zero pixel is True
    binary_mask = grid > 0
    
    # Since jax.scipy.ndimage.label is deprecated, we use jax.pure_callback
    # to call the original scipy function. This runs on the CPU but can be
    # embedded within a JIT-compiled JAX function.
    result_shape_dtypes = (
        jax.ShapeDtypeStruct(binary_mask.shape, jnp.int32), # Labeled grid
        jax.ShapeDtypeStruct((), jnp.int32)                 # Number of features
    )
    labeled_grid, _ = jax.pure_callback(
        _scipy_label_wrapper, result_shape_dtypes, binary_mask
    )
    return labeled_grid

# --- Relational / Copying Primitives ---

@partial(jax.jit)
def copy_rectangle(source_grid: jnp.ndarray, dest_grid: jnp.ndarray, 
                   from_rect: Tuple[int, int, int, int], 
                   to_pos: Tuple[int, int]) -> jnp.ndarray:
    """
    Copies a rectangular region from a source grid to a destination grid.
    
    Args:
        source_grid: The grid to copy from.
        dest_grid: The grid to copy to (the canvas).
        from_rect: (y, x, h, w) of the source rectangle.
        to_pos: (y, x) of the top-left corner on the destination grid.
    
    Returns:
        A new destination grid with the copied rectangle.
    """
    y, x, h, w = from_rect
    to_y, to_x = to_pos
    
    # Extract the slice from the source
    source_slice = jax.lax.dynamic_slice(source_grid, (y, x), (h, w))
    
    # Paste the slice onto the destination grid
    return jax.lax.dynamic_update_slice(dest_grid, source_slice, (to_y, to_x))

@partial(jax.jit)
def isolate_object_by_color(grid: jnp.ndarray, color: int) -> jnp.ndarray:
    """
    Creates a new grid containing only the objects that have the specified color.
    """
    # Find all objects in the grid
    labeled_grid = get_objects(grid)
    
    # Find which object labels correspond to the target color
    # Note: jnp.unique is not supported on non-array inputs inside JIT. 
    # This might need adjustment if used in a more complex JIT context.
    # For now, it works with the current setup.
    object_labels_with_color = jnp.unique(labeled_grid * (grid == color))
    
    # Create a mask for all objects that have the target color
    output_mask = jnp.isin(labeled_grid, object_labels_with_color)
    
    # Apply the mask to the original grid. Also ensure background is kept.
    return grid * (output_mask | (labeled_grid == 0))


# --- Geometric Transform Primitives ---

@partial(jax.jit)
def rotate_90(grid: jnp.ndarray, k: int = 1) -> jnp.ndarray:
    """Rotates a grid 90 degrees clockwise k times."""
    return jnp.rot90(grid, k=k, axes=(1, 0))

@partial(jax.jit)
def flip(grid: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
    """Flips a grid along a given axis (0 for vertical, 1 for horizontal)."""
    return jnp.flip(grid, axis=axis)

# --- Generative Primitives ---

@partial(jax.jit)
def draw_rectangle(grid: jnp.ndarray, rect: Tuple[int, int, int, int], color: int) -> jnp.ndarray:
    """
    Draws a filled rectangle of a specific color onto a grid.
    
    Args:
        grid: The canvas grid to draw on.
        rect: (y, x, h, w) of the rectangle to draw.
        color: The integer color to fill with.
        
    Returns:
        A new grid with the rectangle drawn.
    """
    y, x, h, w = rect
    # Create a patch of the correct color and size
    patch = jnp.full((h, w), fill_value=color, dtype=grid.dtype)
    # Update the slice on the grid
    return jax.lax.dynamic_update_slice(grid, patch, (y, x))

# --- Canvas Management Primitives ---

@partial(jax.jit)
def resize_canvas(grid: jnp.ndarray, new_height: int, new_width: int) -> jnp.ndarray:
    """
    Resizes the canvas, cropping or padding as needed. Content is anchored top-left.
    """
    current_height, current_width = grid.shape
    new_grid = jnp.zeros((new_height, new_width), dtype=grid.dtype)
    
    # Determine the slice sizes for copying
    copy_h = jnp.minimum(current_height, new_height)
    copy_w = jnp.minimum(current_width, new_width)
    
    # Copy the relevant part of the old grid to the new grid
    return new_grid.at[:copy_h, :copy_w].set(grid[:copy_h, :copy_w])


if __name__ == '__main__':
    # --- Example Usage ---
    # Note: JAX arrays are immutable. These functions return NEW arrays.
    
    key = jax.random.PRNGKey(0)
    
    # Create a sample 6x6 input grid
    input_grid = jnp.array([
        [1, 1, 0, 0, 0, 0],
        [1, 1, 0, 2, 2, 0],
        [0, 0, 0, 2, 2, 0],
        [0, 3, 0, 0, 0, 0],
        [0, 3, 0, 0, 4, 4],
        [0, 0, 0, 0, 4, 4],
    ], dtype=jnp.int32)
    
    print("--- Original Grid ---")
    print(input_grid)

    # 1. Copying
    canvas = jnp.zeros_like(input_grid)
    copied = copy_rectangle(input_grid, canvas, from_rect=(1, 3, 2, 2), to_pos=(0, 0))
    print("\n--- copy_rectangle (copy 2s to top-left) ---")
    print(copied)
    
    # 2. Isolate Object
    isolated = isolate_object_by_color(input_grid, color=1)
    print("\n--- isolate_object_by_color (keep only 1s) ---")
    print(isolated)

    # 3. Rotate
    rotated = rotate_90(input_grid)
    print("\n--- rotate_90 ---")
    print(rotated)

    # 4. Draw
    drawn_on = draw_rectangle(input_grid, rect=(0, 4, 2, 2), color=8)
    print("\n--- draw_rectangle (draw 8s in top-right) ---")
    print(drawn_on)

    # 5. Resize
    resized = resize_canvas(input_grid, new_height=4, new_width=4)
    print("\n--- resize_canvas (crop to 4x4) ---")
    print(resized)

