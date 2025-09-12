import time
import jax
import jax.numpy as jnp
import equinox as eqx
from matplotlib import pyplot as plt
import optax
from typing import List, Tuple, Dict
from functools import partial
import math
import os

# Assuming 'llm_python' is a local module you have.
# Replace this with your actual ARC data loading utility.
from llm_python.utils.task_loader import get_task_loader
import numpy as np

# --- 1. Constants and Configuration ---

PADDING_TOKEN: int = 10
VOCAB_SIZE: int = 11  # 0-9 for colors, 10 for padding
MAX_GRID_DIM: int = 30
NUM_SLOTS: int = 20 # The number of object "slots" the segmenter can use

# --- 2. New Object-Centric Architecture ---

class ConvBlock(eqx.Module):
    """A standard block of Conv -> GroupNorm -> Activation."""
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    norm1: eqx.nn.GroupNorm
    norm2: eqx.nn.GroupNorm
    
    def __init__(self, in_channels: int, out_channels: int, *, key: jax.Array):
        key1, key2 = jax.random.split(key)
        self.conv1 = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, key=key1)
        self.norm1 = eqx.nn.GroupNorm(groups=8, channels=out_channels)
        self.conv2 = eqx.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, key=key2)
        self.norm2 = eqx.nn.GroupNorm(groups=8, channels=out_channels)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x is (C, H, W) - NOT batch-aware
        h = self.conv1(x)
        h = self.norm1(h)
        h = jax.nn.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        h = jax.nn.silu(h)
        return h

class ObjectSegmenter(eqx.Module):
    """A U-Net that takes a single grid and outputs segmentation masks."""
    down_block1: ConvBlock
    down_block2: ConvBlock
    mid_block: ConvBlock
    up_block1: ConvBlock
    up_block2: ConvBlock
    final_conv: eqx.nn.Conv2d
    pool: eqx.nn.MaxPool2d

    def __init__(self, base_channels: int, num_slots: int, *, key: jax.Array):
        keys = jax.random.split(key, 6)
        c1, c2 = base_channels, base_channels * 2
        
        self.down_block1 = ConvBlock(VOCAB_SIZE, c1, key=keys[0])
        self.down_block2 = ConvBlock(c1, c2, key=keys[1])
        self.mid_block = ConvBlock(c2, c2, key=keys[2])
        self.up_block1 = ConvBlock(c2 + c2, c1, key=keys[3]) 
        self.up_block2 = ConvBlock(c1 + c1, c1, key=keys[4])
        self.final_conv = eqx.nn.Conv2d(c1, num_slots, kernel_size=1, key=keys[5])
        self.pool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, grid: jnp.ndarray) -> jnp.ndarray:
        # grid is (H, W)
        x = jax.nn.one_hot(grid, VOCAB_SIZE)
        x = jnp.transpose(x, (2, 0, 1)) # (C, H, W)
        
        skip1 = self.down_block1(x)
        x = self.pool(skip1)
        skip2 = self.down_block2(x)
        x = self.pool(skip2)
        
        x = self.mid_block(x)
        
        x = jax.image.resize(x, (x.shape[0],) + skip2.shape[1:], "nearest")
        x = jnp.concatenate([x, skip2], axis=0)
        x = self.up_block1(x)
        
        x = jax.image.resize(x, (x.shape[0],) + skip1.shape[1:], "nearest")
        x = jnp.concatenate([x, skip1], axis=0)
        x = self.up_block2(x)
        
        slot_logits = self.final_conv(x) # (K, H, W)
        
        # Softmax across slots for each pixel
        return jax.nn.softmax(slot_logits, axis=0)

def reconstruct_from_segmentation(
    original_grid: jnp.ndarray, segmentation_masks: jnp.ndarray
) -> jnp.ndarray:
    """A non-learned function to reconstruct a grid from its segmentation."""
    # original_grid: (H, W), segmentation_masks: (K, H, W)
    
    original_one_hot = jax.nn.one_hot(original_grid, VOCAB_SIZE) # (H, W, C)
    
    sum_colors = jnp.einsum('khw,hwc->kc', segmentation_masks, original_one_hot)
    count_pixels = jnp.einsum('khw->k', segmentation_masks)
    avg_colors = sum_colors / jnp.maximum(count_pixels[:, None], 1) # (K, C)
    
    reconstructed_logits = jnp.einsum('khw,kc->hwc', segmentation_masks, avg_colors)
    
    return reconstructed_logits

class SegmentationAutoencoder(eqx.Module):
    segmenter: ObjectSegmenter

    def __init__(self, base_channels: int, num_slots: int, *, key: jax.Array):
        self.segmenter = ObjectSegmenter(base_channels, num_slots, key=key)

    def __call__(self, grid: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # This is a single-example model
        segmentation_masks = self.segmenter(grid)
        reconstructed_logits = reconstruct_from_segmentation(grid, segmentation_masks)
        return reconstructed_logits, segmentation_masks

# --- 3. Training and Evaluation Logic ---

def loss_fn(model: SegmentationAutoencoder, grid: jnp.ndarray, key: jax.Array, lambda_mask_size: float) -> jnp.ndarray:
    """Calculates the combined reconstruction and mask size loss."""
    pred_logits, segmentation_masks = model(grid)
    
    # 1. Reconstruction Loss (as before)
    content_mask = grid != PADDING_TOKEN
    recon_losses = optax.softmax_cross_entropy_with_integer_labels(pred_logits, grid)
    recon_loss = (recon_losses * content_mask).sum() / jnp.maximum(content_mask.sum(), 1)
    
    # 2. Mask Size Penalty (NEW)
    # Penalize the average activation across all masks to encourage sparsity and smaller objects.
    # This loss is low when masks are small and/or many slots are empty.
    
    
    # 3. Combine losses
    total_loss = recon_loss + lambda_mask_size * mask_size_loss
    return total_loss

@eqx.filter_jit
def train_step(
    model: SegmentationAutoencoder,
    optim_state: optax.OptState,
    batch: jnp.ndarray, # Batch is a tensor of grids
    key: jax.Array
) -> Tuple[SegmentationAutoencoder, optax.OptState, jnp.ndarray]:
    
    def compute_batch_loss(model, batch, key):
        # Use vmap with in_axes to correctly handle batching
        # Map over the grid (axis 0) and keys (axis 0), but not the model (None)
        loss_fn_vmapped = jax.vmap(loss_fn, in_axes=(None, 0, 0, None))
        keys = jax.random.split(key, batch.shape[0])
        return jnp.mean(loss_fn_vmapped(model, batch, keys, LAMBDA_CONTINUITY))

    loss_val, grads = eqx.filter_value_and_grad(compute_batch_loss)(model, batch, key)
    updates, new_optim_state = optimizer.update(grads, optim_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_optim_state, loss_val

def evaluate_metrics_single(model: SegmentationAutoencoder, grid: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculates loss and pixel accuracy for a single example."""
    pred_logits, _ = model(grid)
    pred_grid = jnp.argmax(pred_logits, axis=-1)
    
    content_mask = grid != PADDING_TOKEN
    
    # Loss
    losses = optax.softmax_cross_entropy_with_integer_labels(pred_logits, grid)
    masked_loss = (losses * content_mask).sum() / jnp.maximum(content_mask.sum(), 1)
    
    # Accuracy
    matches = (pred_grid == grid)
    correct_pixels = (matches & content_mask).sum()
    total_pixels = content_mask.sum()
    
    return masked_loss, correct_pixels, total_pixels

# --- 4. Visualization ---

def visualize_segmentation(model: SegmentationAutoencoder, grid: jnp.ndarray, save_path: str):
    model = eqx.tree_inference(model, value=True)
    
    # Get segmentation and reconstruction from the single-example model
    segmentation_masks = model.segmenter(grid)
    reconstructed_logits = reconstruct_from_segmentation(grid, segmentation_masks)
    reconstructed_grid = jnp.argmax(reconstructed_logits, axis=-1)

    # Plotting
    cmap = plt.cm.colors.ListedColormap([
        "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
        "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
        "#444444" # Padding
    ])
    norm = plt.cm.colors.Normalize(vmin=0, vmax=10)

    active_masks = [mask for mask in segmentation_masks if mask.sum() > 1]
    num_plots = 2 + len(active_masks)
    grid_size = math.ceil(math.sqrt(num_plots))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 3, grid_size * 3))
    axes = axes.flatten()

    axes[0].imshow(np.asarray(grid), cmap=cmap, norm=norm, interpolation='nearest')
    axes[0].set_title("Original")

    for i, mask in enumerate(active_masks):
        axes[i + 1].imshow(np.asarray(mask), cmap='gray', interpolation='nearest')
        axes[i + 1].set_title(f"Slot {i + 1}")

    axes[len(active_masks) + 1].imshow(np.asarray(reconstructed_grid), cmap=cmap, norm=norm, interpolation='nearest')
    axes[len(active_masks) + 1].set_title("Reconstructed")

    # Hide unused axes
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    for ax in axes[:num_plots]:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"Saved visualization to {save_path}")

def pad_grid(grid: np.ndarray, max_dim: int = MAX_GRID_DIM) -> np.ndarray:
    h, w = grid.shape
    padded_grid = np.full((max_dim, max_dim), PADDING_TOKEN, dtype=np.int32)
    padded_grid[:h, :w] = grid
    return padded_grid

# --- 5. Main function to run the experiment ---
if __name__ == "__main__":
    # --- Configuration ---
    BATCH_SIZE = 256
    LEARNING_RATE = 3e-4
    NUM_EPOCHS = 1000
    BASE_CHANNELS = 64
    LAMBDA_CONTINUITY = 0.5

    # --- Data Loading ---
    print("Loading and processing ARC data into grids...")
    task_loader = get_task_loader()
    training_tasks = task_loader.get_subset_tasks("arc-prize-2024/training")
    
    all_grids = []
    for _, task in training_tasks:
        for pair in task['train'] + task['test']:
            if pair["input"] is not None:
                all_grids.append(pad_grid(np.array(pair['input'])))
            if pair["output"] is not None:
                all_grids.append(pad_grid(np.array(pair['output'])))
    
    all_grids_np = np.array(all_grids)
    
    # Create train/test split
    key = jax.random.PRNGKey(42)
    indices = jax.random.permutation(key, all_grids_np.shape[0])
    split_idx = int(0.9 * len(indices))
    train_indices, eval_indices = indices[:split_idx], indices[split_idx:]
    
    train_dataset_cpu = all_grids_np[train_indices]
    eval_dataset_cpu = all_grids_np[eval_indices]

    print(f"Loaded {len(train_dataset_cpu)} training grids and {len(eval_dataset_cpu)} evaluation grids.")

    # --- Model Initialization ---
    key = jax.random.PRNGKey(int(time.time()))
    model_key, train_key, eval_key, loader_key = jax.random.split(key, 4)
    model = SegmentationAutoencoder(base_channels=BASE_CHANNELS, num_slots=NUM_SLOTS, key=model_key)
    optimizer = optax.adam(LEARNING_RATE)
    optim_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def count_params(model: eqx.Module):
        return sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"Model has {count_params(model):,} trainable parameters.")

    print(f"Training Autoencoder for {NUM_EPOCHS} epochs with batch size {BATCH_SIZE}...")
    num_train_samples = train_dataset_cpu.shape[0]
    time_since_eval = time.time()
    
    for epoch in range(NUM_EPOCHS):
        loader_key, perm_key = jax.random.split(loader_key)
        indices = jax.random.permutation(perm_key, num_train_samples)
        
        shuffled_train_dataset_cpu = train_dataset_cpu[indices]
        
        for i in range(0, num_train_samples, BATCH_SIZE):
            batch_cpu = shuffled_train_dataset_cpu[i:i+BATCH_SIZE]
            if batch_cpu.shape[0] != BATCH_SIZE:
                continue

            batch_gpu = jnp.array(batch_cpu)
            train_key, step_key = jax.random.split(train_key)
            model, optim_state, train_loss = train_step(model, optim_state, batch_gpu, step_key)

        if time.time() - time_since_eval > 10 or epoch == NUM_EPOCHS - 1:
            time_since_eval = time.time()
            
            model_for_eval = eqx.tree_inference(model, value=True)
            
            # JIT the vmapped evaluation function for performance
            jitted_eval_vmapped = eqx.filter_jit(jax.vmap(partial(evaluate_metrics_single, model_for_eval)))

            # Evaluate on both train and eval sets
            train_dataset_gpu = jnp.array(train_dataset_cpu)
            eval_dataset_gpu = jnp.array(eval_dataset_cpu)
            
            train_losses, train_corrects, train_totals = jitted_eval_vmapped(train_dataset_gpu)
            eval_losses, eval_corrects, eval_totals = jitted_eval_vmapped(eval_dataset_gpu)
            
            train_loss_val = train_losses.sum() / len(train_dataset_gpu)
            eval_loss_val = eval_losses.sum() / len(eval_dataset_gpu)
            train_acc = 100 * train_corrects.sum() / train_totals.sum()
            eval_acc = 100 * eval_corrects.sum() / eval_totals.sum()
            
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS} | Train Loss: {train_loss_val:.4f} | Eval Loss: {eval_loss_val:.4f} | Train Px Acc: {train_acc:.2f}% | Eval Px Acc: {eval_acc:.2f}%")

    # Final visualization
    print("\n--- Final Visualization ---")
    model_for_eval = eqx.tree_inference(model, value=True)
    vis_key = jax.random.PRNGKey(42)
    num_visualizations = 10
    
    if not os.path.exists("segmentation_vis"):
        os.makedirs("segmentation_vis")
        
    for i in range(num_visualizations):
        vis_key, choice_key = jax.random.split(vis_key)
        idx = jax.random.choice(choice_key, eval_dataset_cpu.shape[0])
        grid_to_vis = jnp.array(eval_dataset_cpu[idx])
        
        save_path = f"segmentation_vis/example_{i}.png"
        visualize_segmentation(model_for_eval, grid_to_vis, save_path)

