import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter as gaussian_filter

# JAX provides its own implementations for these functions
from jax.scipy.ndimage import gaussian_filter as jax_gaussian_filter


def grid_distance(
    grid1: np.ndarray, grid2: np.ndarray, spatial_sigma: float = 1.0
) -> float:
    """
    Computes a sophisticated distance metric between two grids using NumPy/SciPy.

    This version is CPU-based and provides a reference implementation for the
    distance metric, including a proper color-channel blurring step.

    Args:
        grid1: The first grid, as a 3D NumPy array (H, W, C).
        grid2: The second grid, with the same shape as grid1.
        spatial_sigma: The standard deviation for the Gaussian kernel.

    Returns:
        A float representing the distance.
    """
    if grid1.shape != grid2.shape:
        raise ValueError("Input grids must have the same shape.")

    # --- Step 1: Spatial Blurring using a Gaussian Kernel ---
    # We don't blur the channel axis, so its sigma is 0.
    sigmas = [spatial_sigma, spatial_sigma, 0]
    grid1_spatial_blurred = gaussian_filter(grid1, sigma=sigmas, mode="wrap")
    grid2_spatial_blurred = gaussian_filter(grid2, sigma=sigmas, mode="wrap")

    # --- Step 3: Flatten and Calculate Cosine Similarity ---
    vec1 = grid1_spatial_blurred.flatten()
    vec2 = grid2_spatial_blurred.flatten()

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    epsilon = 1e-8
    similarity = dot_product / np.maximum(norm1 * norm2, epsilon)

    distance = 1.0 - similarity

    return distance


# By decorating with @jax.jit, JAX will compile this entire function
# to highly optimized machine code (e.g., for GPU) the first time it's called.
@jax.jit
def grid_distance_jax(
    grid1: jnp.ndarray, grid2: jnp.ndarray, spatial_sigma: float = 1.0
) -> jnp.ndarray:
    """
    Computes a sophisticated distance metric between two grids using JAX.

    This JAX version is JIT-compilable for high performance, making it suitable
    for use as a heuristic in a fast search algorithm.

    Args:
        grid1: The first grid, as a 3D JAX array (H, W, C).
        grid2: The second grid, with the same shape as grid1.
        spatial_sigma: The standard deviation for the Gaussian kernel.
        color_blur_kernel: The 1D kernel for blurring across color channels.

    Returns:
        A scalar JAX array representing the distance.
    """
    # Note: JAX functions are often stricter about shapes. We assume validation
    # happens outside this performance-critical kernel.

    # --- Step 1: Spatial Blurring using a Gaussian Kernel ---
    # JAX's gaussian_filter expects a list of sigmas for each axis.
    # We don't blur the channel axis, so its sigma is 0.
    sigmas = [spatial_sigma, spatial_sigma, 0]
    grid1_spatial_blurred = jax_gaussian_filter(grid1, sigma=sigmas, mode="wrap")
    grid2_spatial_blurred = jax_gaussian_filter(grid2, sigma=sigmas, mode="wrap")

    # --- Step 3: Flatten and Calculate Cosine Similarity ---
    vec1 = grid1_spatial_blurred.flatten()
    vec2 = grid2_spatial_blurred.flatten()

    dot_product = jnp.dot(vec1, vec2)
    norm1 = jnp.linalg.norm(vec1)
    norm2 = jnp.linalg.norm(vec2)

    epsilon = 1e-8
    similarity = dot_product / jnp.maximum(norm1 * norm2, epsilon)

    distance = 1.0 - similarity

    return distance
