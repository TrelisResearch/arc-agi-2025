import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import os
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

from llm_python.utils.task_loader import get_task_loader


@dataclass
class SegmentedObject:
    """
    A dataclass to hold a segmented object and its properties for sorting.
    The sorting order determines the layering (simplest first).
    """

    y: int
    x: int
    color: int
    mask: np.ndarray  # Mask is cropped to the bounding box

    def mask_sum(self) -> int:
        """Returns the number of pixels in the object (sum of mask)."""
        return int(np.sum(self.mask))

    def __lt__(self, other: "SegmentedObject") -> bool:
        # Sort by mask sum (ascending), then y, then x
        return (self.mask_sum(), -self.y, -self.x) < (other.mask_sum(), -other.y, -other.x)


@dataclass
class TokenizedObject:
    """A JSON-serializable representation of a segmented object."""

    x: int
    y: int
    c: int
    shape_cell: Optional[bool] = None
    shape_rect: Optional[Dict[str, int]] = None
    shape_mask: Optional[List[List[int]]] = None

    def __post_init__(self):
        shapes = ["shape_cell", "shape_rect", "shape_mask"]
        present_shapes = [s for s in shapes if getattr(self, s) is not None]
        if len(present_shapes) != 1:
            raise ValueError(
                f"Exactly one of {shapes} must be provided. Found: {present_shapes}"
            )


def find_connected_components(grid: np.ndarray, color: int) -> List[np.ndarray]:
    """
    Finds all distinct  connected components of a single color in a grid.
    This function correctly handles the color 0.
    """
    h, w = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            if grid[r, c] == color and not visited[r, c]:
                # Start of a new component
                component_mask = np.zeros_like(grid, dtype=bool)
                q = [(r, c)]
                visited[r, c] = True
                component_mask[r, c] = True

                head = 0
                while head < len(q):
                    row, col = q[head]
                    head += 1

                    # 8-way connectivity: all neighbors including diagonals
                    # neighbours = [
                    #     (-1, -1), (-1, 0), (-1, 1),
                    #     (0, -1),           (0, 1),
                    #     (1, -1),  (1, 0),  (1, 1)
                    # ]
                    # 4-way connectivity: only up, down, left, right
                    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]

                    for dr, dc in neighbours:
                        nr, nc = row + dr, col + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            if grid[nr, nc] == color and not visited[nr, nc]:
                                visited[nr, nc] = True
                                component_mask[nr, nc] = True
                                q.append((nr, nc))
                components.append(component_mask)
    return components


def get_bounding_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Calculates the (y, x, h, w) of the bounding box for a boolean mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows):  # Empty mask
        return 0, 0, 0, 0

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return ymin, xmin, ymax - ymin + 1, xmax - xmin + 1


def segment_grid(grid: np.ndarray) -> List[SegmentedObject]:
    """
    Segments a grid into a layered list of single-color, 8-way connected objects.
    It finds all objects, sorts them by simplicity, and then resolves containment
    by updating masks based on the layering order.

    Args:
        grid: A 2D numpy array representing the ARC grid.

    Returns:
        A list of SegmentedObject instances, sorted from most complex (background)
        to simplest (foreground).
    """

    # A special value to mark pixels that have already been assigned to an object
    REMOVED_PIXEL = -1

    segmented_objects: List[SegmentedObject] = []

    # --- Pass 1: Find all atomic objects on the original grid ---
    for color in np.unique(grid):
        components = find_connected_components(grid, color)
        for component_mask in components:
            y, x, h, w = get_bounding_box(component_mask)

            if h == 0 or w == 0:
                continue

            # Crop the mask to its bounding box
            cropped_mask = component_mask[y : y + h, x : x + w]

            segmented_obj = SegmentedObject(y=y, x=x, color=color, mask=cropped_mask)
            segmented_objects.append(segmented_obj)

    # Sort all found objects by simplicity (smallest bbox first)
    segmented_objects.sort()

    # --- Pass 2: Apply layering logic to handle containment ---
    mutable_grid = np.copy(grid)
    for obj in segmented_objects:
        h, w = obj.mask.shape
        y, x = obj.y, obj.x

        # Get the slice of the mutable_grid corresponding to the object's bounding box
        grid_slice = mutable_grid[y : y + h, x : x + w]

        # Store the original mask before potentially modifying it
        original_mask = obj.mask.copy()

        # Find where pixels have already been removed (-1) within this slice
        wildcard_mask = grid_slice == REMOVED_PIXEL

        # Update the object's mask to include these wildcard areas
        obj.mask = original_mask | wildcard_mask

        # "Erase" the original object's pixels from the mutable grid for the next iteration.
        # Use the ORIGINAL mask for erasure to not erase already-erased pixels.
        grid_slice[original_mask] = REMOVED_PIXEL

    # Reverse the order to get background-first layering
    segmented_objects.reverse()

    final_segmentation = segmented_objects
    # This is the final layered representation, from background to foreground
    return final_segmentation


def reconstruct_grid(layered_objects: List[SegmentedObject]) -> np.ndarray:
    max_x, max_y = 0, 0
    for obj in layered_objects:
        max_x = max(max_x, obj.x + obj.mask.shape[1])
        max_y = max(max_y, obj.y + obj.mask.shape[0])

    reconstructed_grid = np.full((max_y, max_x), -2)  # Start with a placeholder

    # Iterate forward through the list (from most complex/background to simplest/foreground)
    for obj in layered_objects:
        h, w = obj.mask.shape
        # Create a patch of the object's color
        color_patch = np.full((h, w), obj.color)
        # Place the patch onto the grid, using the mask to only overwrite the object's pixels
        grid_slice = reconstructed_grid[obj.y : obj.y + h, obj.x : obj.x + w]
        np.copyto(grid_slice, color_patch, where=obj.mask)

    return reconstructed_grid


def tokenize_object(obj: SegmentedObject) -> TokenizedObject:
    """
    Converts a single SegmentedObject into a TokenizedObject.
    """
    token_data = {"x": int(obj.x), "y": int(obj.y), "c": int(obj.color)}
    h, w = obj.mask.shape

    # Check for simple shape optimizations
    if h == 1 and w == 1 and obj.mask[0, 0]:
        token_data["shape_cell"] = True
    elif np.all(obj.mask):  # The mask is a solid rectangle
        token_data["shape_rect"] = {"w": int(w), "h": int(h)}
    else:  # Fallback to the full mask
        token_data["shape_mask"] = obj.mask.astype(int).tolist()

    return TokenizedObject(**token_data)


def tokenize_objects(objs: List[SegmentedObject]) -> List[TokenizedObject]:
    """
    Converts a list of SegmentedObject instances into a list of TokenizedObject instances.
    """
    return [tokenize_object(obj) for obj in objs]


def detokenize_object(token: TokenizedObject) -> SegmentedObject:
    """
    Converts a single TokenizedObject back into a SegmentedObject instance.
    """
    x, y, color = token.x, token.y, token.c

    mask: Optional[np.ndarray] = None
    if token.shape_cell:
        mask = np.ones((1, 1), dtype=bool)
    elif token.shape_rect:
        w, h = token.shape_rect["w"], token.shape_rect["h"]
        mask = np.ones((h, w), dtype=bool)
    elif token.shape_mask:
        mask = np.array(token.shape_mask, dtype=bool)

    if mask is None:
        raise ValueError(f"Invalid token format, no shape found: {token}")

    return SegmentedObject(y=y, x=x, color=color, mask=mask)


def detokenize_objects(tokens: List[TokenizedObject]) -> List[SegmentedObject]:
    """
    Converts a list of TokenizedObject instances back into a list of SegmentedObject instances.
    """
    return [detokenize_object(token) for token in tokens]


def visualize_segmentation(
    grid: np.ndarray, segmentation: List[SegmentedObject]
) -> Image.Image:
    """
    Visualizes a segmentation by creating a plot with the original grid
    and up to 63 segmentation layers.

    Args:
        grid: The original 2D numpy array of the grid.
        segmentation: A list of SegmentedObject instances.

    Returns:
        A PIL Image object of the visualization.
    """
    # ARC colors
    arc_cmap = ListedColormap(
        [
            "#000000",  # 0
            "#0074D9",  # 1
            "#FF4136",  # 2
            "#2ECC40",  # 3
            "#FFDC00",  # 4
            "#AAAAAA",  # 5
            "#F012BE",  # 6
            "#FF851B",  # 7
            "#7FDBFF",  # 8
            "#870C25",  # 9
        ]
    )

    fig, axes = plt.subplots(8, 8, figsize=(16, 16))
    axes = axes.ravel()

    # Plot original grid
    axes[0].imshow(grid, cmap=arc_cmap, vmin=0, vmax=9)
    axes[0].set_title("Original")
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot segmentation layers
    for i, obj in enumerate(segmentation[:63]):
        ax_idx = i + 1
        layer_grid = np.full(grid.shape, -1)  # Use -1 for background
        y, x, h, w = get_bounding_box(obj.mask)

        # Get the full mask, not the cropped one
        full_mask = np.zeros_like(grid, dtype=bool)

        # Find where the object was originally
        original_y = obj.y
        original_x = obj.x

        full_mask[original_y : original_y + h, original_x : original_x + w] = obj.mask

        layer_grid[full_mask] = obj.color

        # Create a custom colormap for this layer
        # Background is light grey, object color is from arc_cmap
        custom_cmap = ListedColormap([(0.8, 0.8, 0.8)] + arc_cmap.colors)

        # Normalize to map -1 to index 0, and 0-9 to indices 1-10
        norm = plt.Normalize(vmin=-1, vmax=9)

        axes[ax_idx].imshow(layer_grid, cmap=custom_cmap, norm=norm)
        axes[ax_idx].set_title(f"L{i} C{obj.color}")
        axes[ax_idx].set_xticks([])
        axes[ax_idx].set_yticks([])

    # Hide unused plots
    for i in range(len(segmentation) + 1, 64):
        axes[i].axis("off")

    fig.tight_layout()

    # Convert to PIL Image
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    img = Image.frombuffer("RGBA", fig.canvas.get_width_height(), buf)
    img = img.convert("RGB")
    plt.close(fig)
    return img


def create_prompt(
    examples: List[Tuple[List[TokenizedObject], List[TokenizedObject]]],
    test_input: List[TokenizedObject],
) -> str:
    """
    Generates a prompt for an LLM to predict the output TokenizedObject.

    Args:
        examples: A list of (input, output) pairs of TokenizedObject.
        test_input: The test input TokenizedObject.

    Returns:
        A string prompt for the LLM.
    """
    prompt = """
You are an expert in solving abstract reasoning problems.
You will be provided with several examples of object-centric representations of input output grids.
Grids are 2D grids of colored cells, where each cell has a color represented by an integer from 0 to 9.
Your goal is to determine what the transformation rule is based on the examples, and then apply that rule to a new test input to produce the correct output in the same representation format.
The intention is you think about the transformation in these object centric representations, rather than the raw grid.
The grids are represented as a list of layers, where each layer is a colored shape.
The layers are stacked in order, with the first layer at the bottom, and subsequent layers override previous layers where they overlap.
The shapes can be rectangles, single cells, or arbitrary masks - prefer to produce rectangles and single cells where possible where it makes sense.

Follow the schema of the examples to produce the output for the test case.
The output must be a single JSON object enclosed in ```json ... ```.

Here are the examples:
"""

    def dict_factory(data):
        return {k: v for k, v in data if v is not None}

    def format_json_list(obj_list: List[TokenizedObject]) -> str:
        if not obj_list:
            return "[]"
        lines = [
            "  " + json.dumps(asdict(o, dict_factory=dict_factory)) for o in obj_list
        ]
        return "[\n" + ",\n".join(lines) + "\n]"

    for i, (inp, outp) in enumerate(examples):
        prompt += f"--- Example {i + 1} ---\n"
        prompt += "Input:\n"
        prompt += "```json\n"
        prompt += format_json_list(inp) + "\n"
        prompt += "```\n\n"
        prompt += "Output:\n"
        prompt += "```json\n"
        prompt += format_json_list(outp) + "\n"
        prompt += "```\n\n"

    prompt += "--- Test Case ---\n"
    prompt += "Input:\n"
    prompt += "```json\n"
    prompt += format_json_list(test_input) + "\n"
    prompt += "```\n\n"
    
    prompt += "Provide any reasoning, then the representation of the predicted test case output in the json block at the end.\n"

    return prompt


def parse_llm_output(response: str) -> Optional[List[TokenizedObject]]:
    """
    Parses the LLM's response to extract the TokenizedObject.

    Args:
        response: The full response from the LLM.

    Returns:
        A TokenizedObjectList if parsing is successful, otherwise None.
    """
    try:
        # Find the JSON block
        start = response.find("```json")
        if start == -1:
            return None
        start += len("```json")

        end = response.find("```", start)
        if end == -1:
            return None

        json_str = response[start:end].strip()
        data = json.loads(json_str)
        # This assumes the output is a single object, not a list
        if isinstance(data, list):
            return [TokenizedObject(**item) for item in data]
        else:
            raise ValueError("Expected a list of TokenizedObject")
    except (json.JSONDecodeError, Exception):
        return None


if __name__ == "__main__":
    # --- Example Usage ---
    # Create a sample grid with a few objects to test the logic.
    # It includes:
    # - A single red pixel (should be simplest)
    # - A 2x1 blue line (next simplest)
    # - A 2x2 green square (next)
    # - A hollow 4x4 gray square (more complex)
    # - A black background (most complex)
    sample_grid = np.zeros((10, 10), dtype=np.int32)
    sample_grid[1, 1] = 2  # Red pixel
    sample_grid[3, 5:7] = 1  # Blue line
    sample_grid[7:9, 2:4] = 3  # Green square

    # Hollow gray square. The iterative process should now see this as two objects:
    # 1. The inner black square (volume 4)
    # 2. The outer gray square (volume 16)
    sample_grid[2:6, 2:6] = 5
    sample_grid[3:5, 3:5] = 0

    print("--- Original Grid ---")
    print(sample_grid)
    print("-" * 20)

    # Get the layered list of objects
    layered_objects = segment_grid(sample_grid)

    print(
        f"--- Detected and Layered {len(layered_objects)} Objects (Background First) ---"
    )
    for i, obj in enumerate(layered_objects):
        print(f"Layer {i}:")
        print(
            f"  Properties: Color={obj.color}, Pos=({obj.x},{obj.y}), Shape={obj.mask.shape}"
        )
        print(obj.mask.astype(np.uint8))

    # --- Reconstruct the grid to verify the layering ---
    reconstructed_grid = reconstruct_grid(layered_objects)

    print("\n--- Reconstructed Grid (from background to foreground) ---")
    print(reconstructed_grid)

    assert np.array_equal(sample_grid, reconstructed_grid), "Reconstruction failed!"
    print("\nVerification successful: Reconstructed grid matches the original.")

    task_loader = get_task_loader()
    tasks = task_loader.get_subset_tasks("arc-prize-2024/evaluation")

    # Select a random sample of 20 tasks
    random.seed(42)
    sampled_tasks = random.sample(tasks, 20)

    output_dir = "segmentation_vis"
    os.makedirs(output_dir, exist_ok=True)

    for task_id, task in sampled_tasks:
        print(f"Task ID: {task_id}")
        for i, example in enumerate(task["train"][:1]):
            input_grid = np.array(example["input"], dtype=np.int32)
            output_grid = np.array(example["output"], dtype=np.int32)

            segmented_input = segment_grid(input_grid)
            segmented_output = segment_grid(output_grid)

            reconstructed_input = reconstruct_grid(
                detokenize_objects(tokenize_objects(segmented_input))
            )
            reconstructed_output = reconstruct_grid(
                detokenize_objects(tokenize_objects(segmented_output))
            )
            assert np.array_equal(input_grid, reconstructed_input), (
                "Input reconstruction failed!"
            )
            assert np.array_equal(output_grid, reconstructed_output), (
                "Output reconstruction failed!"
            )

            print(f" Example {i}:")
            print(f"  Input Grid:\n{input_grid}")
            print(f"  Segmented into {len(segmented_input)} objects.")
            for j, obj in enumerate(segmented_input):
                print(tokenize_object(obj))

            print(f"  Output Grid:\n{output_grid}")
            print(f"  Segmented into {len(segmented_output)} objects.")
            for j, obj in enumerate(segmented_output):
                print(tokenize_object(obj))

            # Save segmentation visualizations
            input_vis = visualize_segmentation(input_grid, segmented_input)
            input_vis.save(os.path.join(output_dir, f"{task_id}_input.png"))
            print(f"Saved {task_id}_input.png")

            output_vis = visualize_segmentation(output_grid, segmented_output)
            output_vis.save(os.path.join(output_dir, f"{task_id}_output.png"))
            print(f"Saved {task_id}_output.png")

    print("Done.")
